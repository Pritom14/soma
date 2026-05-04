"""
core/finetuner.py — LoRA fine-tune coordinator for SOMA.

Responsibilities:
  - Prepare Alpaca-format training datasets from recorded trajectories
  - Track fine-tune runs in a persistent log (finetune_log.json)
  - Decide whether enough new successful trajectories have accumulated
    to warrant a new fine-tune run

This module does NOT call any training API (Ollama, Anthropic, MLX-LM, etc.).
It handles only data preparation and state tracking.  Actual training is
invoked by bootstrap/dream_cycle.py → _run_finetune() when mlx_lm is present.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import BASE_DIR

_TRAJ_DIR = BASE_DIR / "data" / "trajectories"
_FINETUNE_LOG = BASE_DIR / "data" / "finetune_log.json"
_WATERMARK_FILE = _TRAJ_DIR / ".last_finetune"


# ---------------------------------------------------------------------------
# Internal helpers (reused from scripts/train_lora.py logic but kept here so
# core/ has no dependency on scripts/)
# ---------------------------------------------------------------------------


def _load_trajectories(traj_dir: Path) -> list[dict]:
    """Parse every *.jsonl file in *traj_dir* and return all records."""
    records: list[dict] = []
    for path in sorted(traj_dir.glob("*.jsonl")):
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return records


def _is_successful(record: dict) -> bool:
    """Return True if a trajectory record represents a completed, non-failed run."""
    if record.get("failed", False):
        return False
    if record.get("partial", False):
        return False
    # ``completed`` defaults to True for records that pre-date the field
    if not record.get("completed", True):
        return False
    return True


def _record_to_alpaca(record: dict) -> list[dict]:
    """
    Convert one trajectory record to Alpaca instruction-tuning examples.

    Each human→gpt exchange becomes one dict::

        {
            "instruction": <human turn text>,
            "input":       "",
            "output":      <gpt turn text>,
        }
    """
    conversations = record.get("conversations", [])
    metadata = record.get("metadata", {})
    examples: list[dict] = []

    i = 0
    while i < len(conversations) - 1:
        turn_a = conversations[i]
        turn_b = conversations[i + 1]
        if turn_a.get("from") == "human" and turn_b.get("from") == "gpt":
            instruction = turn_a.get("value", "").strip()
            output = turn_b.get("value", "").strip()
            if instruction and output:
                examples.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                        "_domain": metadata.get("domain", ""),
                        "_model": metadata.get("model_name", ""),
                        "_trajectory_id": metadata.get("trajectory_id", ""),
                    }
                )
            i += 2
        else:
            i += 1

    return examples


# ---------------------------------------------------------------------------
# FineTuner
# ---------------------------------------------------------------------------


class FineTuner:
    """
    Coordinates LoRA dataset preparation and fine-tune state tracking.

    Parameters
    ----------
    traj_dir:
        Directory containing trajectory *.jsonl files.
        Defaults to ``data/trajectories/`` under BASE_DIR.
    log_path:
        JSON file that persists fine-tune run history.
        Defaults to ``data/finetune_log.json`` under BASE_DIR.
    """

    MIN_TRAJECTORIES: int = 10  # class-level default threshold

    def __init__(
        self,
        traj_dir: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self._traj_dir: Path = traj_dir or _TRAJ_DIR
        self._log_path: Path = log_path or _FINETUNE_LOG
        self._watermark_file: Path = self._traj_dir / ".last_finetune"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_dataset(self, min_trajectories: int = MIN_TRAJECTORIES) -> Optional[list[dict]]:
        """
        Build a list of Alpaca-format training examples from successful trajectories.

        Returns
        -------
        list[dict]
            Training examples when ``len(examples) >= min_trajectories``.
        None
            When there are fewer successful trajectories than the threshold.
        """
        if not self._traj_dir.exists():
            return None

        records = _load_trajectories(self._traj_dir)
        successful = [r for r in records if _is_successful(r)]

        if len(successful) < min_trajectories:
            return None

        examples: list[dict] = []
        for rec in successful:
            examples.extend(_record_to_alpaca(rec))

        return examples if examples else None

    def export_dataset(self, examples: list[dict], output_path: Path) -> None:
        """
        Write *examples* as newline-delimited JSON to *output_path*.

        Creates parent directories as needed.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def should_finetune(self, threshold: int = MIN_TRAJECTORIES) -> bool:
        """
        Return True when there are at least *threshold* successful trajectories
        recorded since the last fine-tune watermark.

        The watermark is the mtime of ``data/trajectories/.last_finetune``.
        If the file does not exist every successful trajectory counts.
        """
        if not self._traj_dir.exists():
            return False

        watermark_mtime: float = 0.0
        if self._watermark_file.exists():
            watermark_mtime = self._watermark_file.stat().st_mtime

        import os

        new_successful = 0
        for path in sorted(self._traj_dir.glob("*.jsonl")):
            file_mtime = os.path.getmtime(path)
            if file_mtime <= watermark_mtime:
                continue
            try:
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if _is_successful(rec):
                            new_successful += 1
            except OSError:
                continue

        return new_successful >= threshold

    def record_finetune(self, run_id: str, example_count: int) -> None:
        """
        Append a fine-tune run entry to the persistent log.

        The log file is a JSON array; it is created if absent.

        Parameters
        ----------
        run_id:
            Unique identifier for the run (caller may use any string, e.g. a
            UUID or timestamp).
        example_count:
            Number of training examples included in this run.
        """
        log: list[dict] = []
        if self._log_path.exists():
            try:
                log = json.loads(self._log_path.read_text())
                if not isinstance(log, list):
                    log = []
            except (json.JSONDecodeError, OSError):
                log = []

        entry = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "example_count": example_count,
        }
        log.append(entry)

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(json.dumps(log, indent=2) + "\n")

    def last_run(self) -> Optional[dict]:
        """Return the most recent log entry, or None if no runs have been recorded."""
        if not self._log_path.exists():
            return None
        try:
            log = json.loads(self._log_path.read_text())
            if isinstance(log, list) and log:
                return log[-1]
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def generate_run_id(self) -> str:
        """Generate a unique run identifier."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short = str(uuid.uuid4())[:8]
        return f"finetune_{ts}_{short}"
