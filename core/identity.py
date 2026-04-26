from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from config import BASE_DIR, BELIEFS_DIR

SOUL_DIR = BELIEFS_DIR / "self"


class IdentityStore:
    SOUL_FIELDS = ["purpose", "values", "style", "capabilities", "limitations", "non_negotiables"]

    DEFAULT_SOUL = {
        "purpose": "I am SOMA. I contribute to open-source projects, solve coding tasks, and learn from every interaction across all domains.",
        "values": "Honesty over confidence. Evidence over assumption. Simplicity over cleverness. Verify before shipping.",
        "style": "Direct and concise. I show reasoning on novel problems. I flag uncertainty explicitly.",
        "capabilities": "Code generation and editing, OSS contribution, debugging, research synthesis, self-testing beliefs, autonomous work loops.",
        "limitations": "More reliable in domains with high-confidence beliefs. Should verify more in unfamiliar territory. Cannot guarantee correctness without verification.",
        "non_negotiables": "Never hallucinate when uncertain. Never skip verification on novel tasks. Always record what I learn.",
        "last_updated": "",
    }

    def __init__(self):
        SOUL_DIR.mkdir(parents=True, exist_ok=True)
        self.path = SOUL_DIR / "soul.json"
        self._soul = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return dict(self.DEFAULT_SOUL)
        try:
            data = json.loads(self.path.read_text())
            # backfill any missing keys from DEFAULT_SOUL
            for k, v in self.DEFAULT_SOUL.items():
                data.setdefault(k, v)
            return data
        except Exception:
            return dict(self.DEFAULT_SOUL)

    def _save(self, soul: dict):
        self.path.write_text(json.dumps(soul, indent=2))
        self._soul = soul

    def get_soul(self) -> dict:
        return dict(self._soul)

    def get_system_context(self) -> str:
        """First-person identity paragraph injected into every LLM system prompt. Max 200 words."""
        s = self._soul
        if not any(s.get(k) for k in self.SOUL_FIELDS):
            return ""
        parts = []
        if s.get("purpose"):
            parts.append(s["purpose"])
        if s.get("values"):
            parts.append(f"My values: {s['values']}")
        if s.get("style"):
            parts.append(f"My style: {s['style']}")
        if s.get("limitations"):
            parts.append(f"My limitations: {s['limitations']}")
        if s.get("non_negotiables"):
            parts.append(f"My non-negotiables: {s['non_negotiables']}")
        text = " ".join(parts)
        # Hard cap at 200 words
        words = text.split()
        if len(words) > 200:
            words = words[:200]
            text = " ".join(words)
            # truncate at last sentence boundary
            for end in (".", "!", "?"):
                idx = text.rfind(end)
                if idx > len(text) // 2:
                    text = text[: idx + 1]
                    break
        return text

    def update_from_introspection(self, new_soul: dict):
        """Rewrite soul from introspection evidence. Logs to brain timeline."""
        # Validate required fields present
        for field in self.SOUL_FIELDS:
            if field not in new_soul:
                new_soul[field] = self._soul.get(field, self.DEFAULT_SOUL.get(field, ""))
        new_soul["last_updated"] = datetime.utcnow().isoformat()
        self._save(new_soul)
        # Log to brain timeline
        try:
            from core.brain import BrainStore
            brain = BrainStore()
            brain.add_timeline(
                slug="soma-identity",
                entity_type="pattern",
                entity_name="SOMA identity",
                event="Identity updated from introspection",
                impact="learning",
            )
        except Exception:
            pass
