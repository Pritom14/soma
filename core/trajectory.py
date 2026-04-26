"""
core/trajectory.py — Auto-generation of LoRA training trajectories.

Every task SOMA runs writes a JSONL record in Hermes agent format to
data/trajectories/YYYYMMDD.jsonl. The dream cycle collects these and
triggers a LoRA fine-tune when enough new ones accumulate.

Usage:
    traj = TrajectoryRecorder(domain="code", model_name="qwen2.5-coder:14b")
    traj.record_step(human_turn=task, gpt_turn=response, tool_name="llm_ask")
    traj.finish(success=True)   # async write, returns immediately
"""
from __future__ import annotations

import fcntl
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

from config import BASE_DIR

_TRAJECTORIES_DIR = BASE_DIR / "data" / "trajectories"
_WATERMARK_FILE = _TRAJECTORIES_DIR / ".last_finetune"


class TrajectoryRecorder:
    """
    Records one task's conversation arc and writes it to a daily JSONL file.
    All file I/O happens in a daemon thread — finish() never blocks the caller.
    """

    TRAJECTORIES_DIR = _TRAJECTORIES_DIR

    def __init__(self, domain: str, model_name: str):
        self._domain = domain
        self._model_name = model_name
        self._trajectory_id = f"{domain}_{datetime.utcnow().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"
        self._conversations: list[dict] = []
        self._tool_stats: dict[str, dict] = {}
        self._api_calls: int = 0
        self._started_at: str = datetime.utcnow().isoformat()

    def record_step(self, human_turn: str, gpt_turn: str,
                    tool_name: str = "", tool_success: bool = True) -> None:
        """
        Record one human→gpt exchange. Optionally tag it with the tool used.
        This method is synchronous and in-memory only — no I/O.
        """
        self._conversations.append({"from": "human", "value": human_turn})
        self._conversations.append({"from": "gpt", "value": gpt_turn})
        self._api_calls += 1

        if tool_name:
            if tool_name not in self._tool_stats:
                self._tool_stats[tool_name] = {"count": 0, "success": 0, "failure": 0}
            self._tool_stats[tool_name]["count"] += 1
            if tool_success:
                self._tool_stats[tool_name]["success"] += 1
            else:
                self._tool_stats[tool_name]["failure"] += 1

    def finish(self, success: bool = True) -> None:
        """
        Finalise the trajectory and write it to disk in a background thread.
        Returns immediately — caller is never blocked.
        Thread is non-daemon so it completes even if main thread exits soon after.
        """
        if not self._conversations:
            return  # nothing to save

        record = self._build_record(success)
        t = threading.Thread(target=self._write_async, args=(record,), daemon=False)
        t.start()

    def _build_record(self, success: bool) -> dict:
        has_content = len(self._conversations) > 0
        return {
            "conversations": self._conversations,
            "metadata": {
                "batch": datetime.utcnow().strftime("%Y%m%d"),
                "timestamp": datetime.utcnow().isoformat(),
                "model_name": self._model_name,
                "domain": self._domain,
                "trajectory_id": self._trajectory_id,
            },
            "completed": success,
            "partial": has_content and not success,
            "tool_stats": self._tool_stats,
            "tool_error_counts": {
                k: v["failure"] for k, v in self._tool_stats.items()
            },
            "toolsets_used": [self._domain],
            "api_calls": self._api_calls,
            "failed": not success,
        }

    def _write_async(self, record: dict) -> None:
        """Daemon thread target. Appends one JSONL line atomically. Never raises."""
        try:
            _TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
            date_str = datetime.utcnow().strftime("%Y%m%d")
            path = _TRAJECTORIES_DIR / f"{date_str}.jsonl"
            line = json.dumps(record, separators=(",", ":")) + "\n"
            with open(path, "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    fh.write(line)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        except Exception:
            pass  # trajectory loss is acceptable; must never crash caller

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @staticmethod
    def stats() -> dict:
        """
        Scan TRAJECTORIES_DIR for all *.jsonl files and return aggregate stats.
        Also counts records newer than the .last_finetune watermark.
        """
        if not _TRAJECTORIES_DIR.exists():
            return {
                "total_files": 0,
                "total_trajectories": 0,
                "by_date": {},
                "success_rate": 0.0,
                "since_last_finetune": 0,
            }

        # Read watermark
        watermark_mtime: float = 0.0
        if _WATERMARK_FILE.exists():
            watermark_mtime = _WATERMARK_FILE.stat().st_mtime

        total = 0
        success_count = 0
        by_date: dict[str, int] = {}
        since_last = 0

        for f in sorted(_TRAJECTORIES_DIR.glob("*.jsonl")):
            file_mtime = f.stat().st_mtime
            newer_than_watermark = file_mtime > watermark_mtime
            date_key = f.stem  # filename is YYYYMMDD
            by_date[date_key] = 0

            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    total += 1
                    by_date[date_key] += 1
                    if not rec.get("failed", True):
                        success_count += 1
                    if newer_than_watermark:
                        since_last += 1

        return {
            "total_files": len(list(_TRAJECTORIES_DIR.glob("*.jsonl"))),
            "total_trajectories": total,
            "by_date": by_date,
            "success_rate": round(success_count / total, 3) if total else 0.0,
            "since_last_finetune": since_last,
        }
