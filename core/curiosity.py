from __future__ import annotations
import math
from datetime import datetime
from core.belief import Belief, BeliefStore
from core.experience import ExperienceStore
from config import BELIEF_DECAY_DAYS


class CuriosityEngine:
    def __init__(self, belief_store: BeliefStore, experience_store: ExperienceStore):
        self.beliefs = belief_store
        self.store = experience_store

    def score(self, belief: Belief) -> float:
        return round(
            self._staleness(belief)
            * self._uncertainty(belief)
            * self._age_weight(belief)
            + self._contradiction_bonus(belief),
            4,
        )

    def select_candidates(self, limit: int = 3, min_score: float = 0.2) -> list[Belief]:
        scored = [(self.score(b), b) for b in self.beliefs.all()]
        scored.sort(key=lambda x: -x[0])
        return [b for s, b in scored if s >= min_score][:limit]

    def scores(self) -> list[tuple[float, Belief]]:
        scored = [(self.score(b), b) for b in self.beliefs.all()]
        scored.sort(key=lambda x: -x[0])
        return scored

    # ------------------------------------------------------------------

    def _staleness(self, belief: Belief) -> float:
        try:
            last = datetime.fromisoformat(belief.last_verified)
        except ValueError:
            return 3.0
        days = (datetime.utcnow() - last).total_seconds() / 86400
        return min(3.0, days / BELIEF_DECAY_DAYS)

    def _uncertainty(self, belief: Belief) -> float:
        # Peak at 0.5 confidence, near-zero at extremes
        return 1.0 - abs(belief.confidence - 0.5) * 2

    def _age_weight(self, belief: Belief) -> float:
        if belief.evidence_count == 1:
            return 1.3
        if belief.evidence_count >= 5:
            return 0.8
        return 1.0

    def _contradiction_bonus(self, belief: Belief) -> float:
        negation = {"not", "never", "dont", "avoid", "without", "no"}
        b_words = set(belief.statement.lower().split())
        b_neg = bool(b_words & negation)
        bonus = 0.0
        for other in self.beliefs.all():
            if other.id == belief.id:
                continue
            o_words = set(other.statement.lower().split())
            overlap = len(b_words & o_words) / max(len(b_words | o_words), 1)
            o_neg = bool(o_words & negation)
            if overlap > 0.4 and b_neg != o_neg:
                bonus = 0.5
                break
        return bonus
