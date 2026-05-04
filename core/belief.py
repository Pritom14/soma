import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime

from config import BELIEFS_DIR


@dataclass
class Belief:
    id: str
    domain: str
    statement: str
    confidence: float
    evidence_count: int
    experience_ids: list
    created_at: str
    last_verified: str
    is_actionable: bool = True  # False means stale, re-verify before acting


class BeliefStore:
    def __init__(self, domain: str):
        self.domain = domain
        self.path = BELIEFS_DIR / f"{domain}.json"
        BELIEFS_DIR.mkdir(parents=True, exist_ok=True)
        self.beliefs: dict[str, Belief] = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text())
            return {k: Belief(**v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupted JSON — fall back to empty, leave file intact for inspection
            return {}

    def _save(self):
        """Atomically persist beliefs to disk: write .tmp then rename."""
        data = {k: asdict(v) for k, v in self.beliefs.items()}
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(self.path)

    def flush(self):
        """Explicit flush — useful after batch updates (e.g. dream cycle Step 7b)."""
        self._save()

    def get_stale(self) -> list["Belief"]:
        """Return all beliefs that are marked not actionable (stale)."""
        return [b for b in self.beliefs.values() if not b.is_actionable]

    def crystallize(
        self, experience_id: str, statement: str, confidence: float, domain: str
    ) -> "Belief":
        # Reinforce if matching belief already exists
        for belief in self.beliefs.values():
            if belief.statement.lower().strip() == statement.lower().strip():
                belief.confidence = round(min(0.99, (belief.confidence + confidence) / 2 + 0.05), 4)
                belief.evidence_count += 1
                if experience_id not in belief.experience_ids:
                    belief.experience_ids.append(experience_id)
                belief.last_verified = datetime.utcnow().isoformat()
                belief.is_actionable = True
                self._save()
                return belief

        belief = Belief(
            id=str(uuid.uuid4())[:8],
            domain=domain,
            statement=statement,
            confidence=confidence,
            evidence_count=1,
            experience_ids=[experience_id],
            created_at=datetime.utcnow().isoformat(),
            last_verified=datetime.utcnow().isoformat(),
        )
        self.beliefs[belief.id] = belief
        self._save()
        return belief

    def get_relevant(self, context: str, limit: int = 5) -> list["Belief"]:
        words = set(context.lower().split())
        scored = []
        for belief in self.beliefs.values():
            b_words = set(belief.statement.lower().split())
            overlap = len(words & b_words) / max(len(words | b_words), 1)
            if overlap > 0.05:
                scored.append((overlap, belief))
        scored.sort(key=lambda x: (-x[0], -x[1].confidence))
        return [b for _, b in scored[:limit]]

    def update_from_experiment(self, belief_id: str, confirmed: bool):
        """Update confidence from a self-test experiment outcome."""
        belief = self.beliefs.get(belief_id)
        if not belief:
            return
        if confirmed:
            belief.confidence = round(min(0.99, belief.confidence + 0.025), 4)
            belief.evidence_count += 1
        else:
            belief.confidence = round(max(0.01, belief.confidence * 0.85), 4)
            belief.is_actionable = belief.confidence >= 0.4
        belief.last_verified = datetime.utcnow().isoformat()
        self._save()

    def update_from_pr(self, belief_id: str, merged: bool):
        """Update confidence from real-world OSS PR outcome - strongest signal."""
        belief = self.beliefs.get(belief_id)
        if not belief:
            return
        if merged:
            belief.confidence = round(min(0.99, belief.confidence + 0.05), 4)
            belief.evidence_count += 1
        else:
            belief.confidence = round(max(0.01, belief.confidence * 0.75), 4)
            belief.is_actionable = belief.confidence >= 0.4
        belief.last_verified = datetime.utcnow().isoformat()
        self._save()

    def record_contradiction(self, belief_id: str, experience_id: str):
        belief = self.beliefs.get(belief_id)
        if belief:
            belief.is_actionable = False
            ref = f"contra:{experience_id}"
            if ref not in belief.experience_ids:
                belief.experience_ids.append(ref)
            self._save()

    def mark_stale(self, belief_id: str):
        if belief_id in self.beliefs:
            self.beliefs[belief_id].is_actionable = False
            self._save()

    def all(self) -> list["Belief"]:
        return list(self.beliefs.values())
