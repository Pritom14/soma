"""core/model_router.py — Complexity-aware model selection for SOMA.

Bridges TaskComplexityScorer output (0.0-1.0 float + task_type string) to a
concrete Ollama/Anthropic model name.  ModelRouter is stateless and cheap to
instantiate; create one per call-site or keep a long-lived instance.

Routing table
-------------
score < 0.3          TIER_3_MODEL  — fast 7b, simple tasks (direct edits)
0.3 <= score < 0.6   TIER_2_MODEL  — mid 32b, moderate tasks
0.6 <= score < 0.9   TIER_1_MODEL  — quality 32b, decomposition / complex edits
score >= 0.9         CLAUDE_MODEL  — cloud best-judgment (dangerous/novel tasks)

Task-type overrides (applied *after* score routing)
----------------------------------------------------
"self_modify"   → forces at least TIER_1_MODEL (escalates to CLAUDE_MODEL if
                  score >= 0.9, but never falls below TIER_1_MODEL)
"dream_cycle"   → always TIER_1_MODEL regardless of score
"""

from __future__ import annotations

import os

from config import CLAUDE_MODEL, TIER_1_MODEL, TIER_2_MODEL, TIER_3_MODEL

# Ordered from cheapest/fastest (index 0) to most capable (index 3).
# Used to enforce "at least tier X" semantics for task-type overrides.
_MODEL_RANK: list[str] = [TIER_3_MODEL, TIER_2_MODEL, TIER_1_MODEL, CLAUDE_MODEL]


def _rank(model: str) -> int:
    """Return the capability rank of *model* (higher = more capable)."""
    try:
        return _MODEL_RANK.index(model)
    except ValueError:
        # Unknown model: treat as highest rank so it is never downgraded.
        return len(_MODEL_RANK)


def _at_least(model: str, minimum: str) -> str:
    """Return *model* if its rank >= *minimum*'s rank, else return *minimum*."""
    return model if _rank(model) >= _rank(minimum) else minimum


class ModelRouter:
    """Select the best model for a task given its complexity score and type.

    Parameters
    ----------
    allow_claude:
        When False the router will never return CLAUDE_MODEL even for scores
        above 0.9 — it falls back to TIER_1_MODEL instead.  Useful when no
        ANTHROPIC_API_KEY is configured.  Defaults to auto-detect from env.
    """

    # Complexity thresholds (exclusive upper bound for each band)
    TIER_3_MAX = 0.3
    TIER_2_MAX = 0.6
    TIER_1_MAX = 0.9
    # Above TIER_1_MAX → CLAUDE_MODEL

    def __init__(self, allow_claude: bool | None = None) -> None:
        if allow_claude is None:
            allow_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._allow_claude = allow_claude

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, complexity_score: float, task_type: str = "") -> str:
        """Return the model name best suited for *complexity_score* and *task_type*.

        Parameters
        ----------
        complexity_score:
            Float in [0.0, 1.0] from TaskComplexityScorer.score().score.
        task_type:
            Optional string hint from the call-site (e.g. ``"self_modify"``,
            ``"dream_cycle"``, ``"code_edit"``).  Empty string = no override.

        Returns
        -------
        str
            A model identifier from config.py (e.g. ``"qwen2.5-coder:32b"``).
        """
        base = self._score_to_model(complexity_score)
        final = self._apply_task_type_override(base, task_type)
        return final

    def routing_policy(self) -> dict:
        """Return a human-readable description of the current routing policy.

        Intended for dream-cycle introspection so SOMA can reason about its
        own model-selection strategy.

        Returns
        -------
        dict with keys:
            ``bands``       — list of {range, model, description} dicts
            ``task_type_overrides`` — dict of task_type -> override rule
            ``claude_enabled``      — bool, whether CLAUDE_MODEL is reachable
            ``models``      — dict of tier -> model-name for quick reference
        """
        return {
            "bands": [
                {
                    "range": f"score < {self.TIER_3_MAX}",
                    "model": TIER_3_MODEL,
                    "description": "Fast 7b — simple/direct tasks with no decomposition",
                },
                {
                    "range": f"{self.TIER_3_MAX} <= score < {self.TIER_2_MAX}",
                    "model": TIER_2_MODEL,
                    "description": "Mid 32b — moderate complexity, standard edits",
                },
                {
                    "range": f"{self.TIER_2_MAX} <= score < {self.TIER_1_MAX}",
                    "model": TIER_1_MODEL,
                    "description": "Quality 32b — complex tasks, decomposition needed",
                },
                {
                    "range": f"score >= {self.TIER_1_MAX}",
                    "model": CLAUDE_MODEL if self._allow_claude else TIER_1_MODEL,
                    "description": (
                        "Cloud best-judgment — dangerous/novel tasks requiring broad reasoning"
                        if self._allow_claude
                        else "Fallback to quality 32b (ANTHROPIC_API_KEY not set)"
                    ),
                },
            ],
            "task_type_overrides": {
                "self_modify": (
                    f"Minimum {TIER_1_MODEL}; escalates to "
                    f"{CLAUDE_MODEL if self._allow_claude else TIER_1_MODEL} if score >= {self.TIER_1_MAX}"
                ),
                "dream_cycle": f"Always {TIER_1_MODEL}, regardless of score",
            },
            "claude_enabled": self._allow_claude,
            "models": {
                "tier_1": TIER_1_MODEL,
                "tier_2": TIER_2_MODEL,
                "tier_3": TIER_3_MODEL,
                "claude": CLAUDE_MODEL,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_to_model(self, score: float) -> str:
        """Map raw complexity score to a model name (no task-type logic)."""
        if score < self.TIER_3_MAX:
            return TIER_3_MODEL
        if score < self.TIER_2_MAX:
            return TIER_2_MODEL
        if score < self.TIER_1_MAX:
            return TIER_1_MODEL
        # score >= 0.9
        if self._allow_claude:
            return CLAUDE_MODEL
        return TIER_1_MODEL  # graceful fallback when API key absent

    def _apply_task_type_override(self, base_model: str, task_type: str) -> str:
        """Enforce task-type override rules on top of *base_model*."""
        tt = (task_type or "").lower().strip()

        if tt == "self_modify":
            # Must be at least TIER_1_MODEL; score-based escalation to Claude preserved.
            return _at_least(base_model, TIER_1_MODEL)

        if tt == "dream_cycle":
            # Always TIER_1_MODEL — not cloud, not tier-3.
            return TIER_1_MODEL

        return base_model
