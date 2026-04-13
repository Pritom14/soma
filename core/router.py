from dataclasses import dataclass
from typing import Optional

from core.experience import Experience
from config import (
    TIER_1_MODEL, TIER_2_MODEL, TIER_3_MODEL,
    TIER_1_THRESHOLD, TIER_2_THRESHOLD, MIN_TESTS_FOR_TIER1,
)


@dataclass
class RouteDecision:
    tier: int
    model: str
    reason: str
    best_experience: Optional[Experience]
    confidence: float


def route(context: str, domain: str, similar: list[Experience]) -> RouteDecision:
    if not similar:
        return RouteDecision(
            tier=3, model=TIER_3_MODEL,
            reason="No prior experience - deep reasoner",
            best_experience=None, confidence=0.0,
        )

    best = max(similar, key=lambda e: e.confidence)

    if best.confidence >= TIER_1_THRESHOLD and best.test_count >= MIN_TESTS_FOR_TIER1:
        return RouteDecision(
            tier=1, model=TIER_1_MODEL,
            reason=f"High confidence ({best.confidence:.0%}, {best.test_count}x tested) - fast local",
            best_experience=best, confidence=best.confidence,
        )

    if best.confidence >= TIER_2_THRESHOLD:
        return RouteDecision(
            tier=2, model=TIER_2_MODEL,
            reason=f"Medium confidence ({best.confidence:.0%}) - mid local",
            best_experience=best, confidence=best.confidence,
        )

    return RouteDecision(
        tier=3, model=TIER_3_MODEL,
        reason=f"Low confidence ({best.confidence:.0%}) - deep cloud",
        best_experience=best, confidence=best.confidence,
    )
