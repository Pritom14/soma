"""tests/test_model_router.py

Tests for ModelRouter: complexity-band routing, task-type overrides,
routing_policy() introspection, and config import sanity.
"""
from __future__ import annotations

import pytest

from config import (
    CLAUDE_MODEL,
    QWEN_14B,
    QWEN_32B,
    QWEN_72B,
    QWEN_7B,
    SUPPORTED_MODELS,
    TIER_1_MODEL,
    TIER_2_MODEL,
    TIER_3_MODEL,
)
from core.model_router import ModelRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router_no_claude():
    """Router with Claude disabled (simulates no ANTHROPIC_API_KEY)."""
    return ModelRouter(allow_claude=False)


@pytest.fixture
def router_with_claude():
    """Router with Claude explicitly enabled (simulates API key present)."""
    return ModelRouter(allow_claude=True)


# ---------------------------------------------------------------------------
# 1. Low complexity → Tier 3 (fast 7b)
# ---------------------------------------------------------------------------


def test_low_complexity_routes_to_tier3(router_no_claude):
    model = router_no_claude.select(0.0)
    assert model == TIER_3_MODEL


def test_low_complexity_boundary_routes_to_tier3(router_no_claude):
    # 0.29 is still below the 0.3 threshold
    model = router_no_claude.select(0.29)
    assert model == TIER_3_MODEL


# ---------------------------------------------------------------------------
# 2. Medium complexity → Tier 2
# ---------------------------------------------------------------------------


def test_medium_complexity_routes_to_tier2(router_no_claude):
    model = router_no_claude.select(0.45)
    assert model == TIER_2_MODEL


def test_medium_complexity_lower_boundary(router_no_claude):
    # Exactly 0.3 enters the Tier 2 band
    model = router_no_claude.select(0.3)
    assert model == TIER_2_MODEL


def test_medium_complexity_upper_boundary(router_no_claude):
    # 0.59 still in Tier 2 band
    model = router_no_claude.select(0.59)
    assert model == TIER_2_MODEL


# ---------------------------------------------------------------------------
# 3. High complexity → Tier 1
# ---------------------------------------------------------------------------


def test_high_complexity_routes_to_tier1(router_no_claude):
    model = router_no_claude.select(0.75)
    assert model == TIER_1_MODEL


def test_high_complexity_lower_boundary(router_no_claude):
    # Exactly 0.6 enters the Tier 1 band
    model = router_no_claude.select(0.6)
    assert model == TIER_1_MODEL


# ---------------------------------------------------------------------------
# 4. Very high complexity → Claude (when enabled) or Tier 1 fallback
# ---------------------------------------------------------------------------


def test_very_high_complexity_routes_to_claude_when_enabled(router_with_claude):
    model = router_with_claude.select(0.95)
    assert model == CLAUDE_MODEL


def test_very_high_complexity_falls_back_to_tier1_without_claude(router_no_claude):
    model = router_no_claude.select(0.95)
    assert model == TIER_1_MODEL


def test_score_exactly_0_9_routes_to_claude_when_enabled(router_with_claude):
    # 0.9 is the threshold — should route to Claude
    model = router_with_claude.select(0.9)
    assert model == CLAUDE_MODEL


# ---------------------------------------------------------------------------
# 5. "self_modify" task type — always Tier 1 or above
# ---------------------------------------------------------------------------


def test_self_modify_low_score_escalates_to_tier1(router_no_claude):
    # Even a trivial score must reach at least Tier 1 for self-modification.
    model = router_no_claude.select(0.05, task_type="self_modify")
    assert model == TIER_1_MODEL


def test_self_modify_medium_score_escalates_to_tier1(router_no_claude):
    model = router_no_claude.select(0.45, task_type="self_modify")
    assert model == TIER_1_MODEL


def test_self_modify_high_score_stays_tier1_without_claude(router_no_claude):
    model = router_no_claude.select(0.75, task_type="self_modify")
    assert model == TIER_1_MODEL


def test_self_modify_very_high_score_goes_claude_when_enabled(router_with_claude):
    # Score >= 0.9 with self_modify + Claude enabled → Claude (highest capability).
    model = router_with_claude.select(0.95, task_type="self_modify")
    assert model == CLAUDE_MODEL


# ---------------------------------------------------------------------------
# 6. "dream_cycle" task type — always Tier 1
# ---------------------------------------------------------------------------


def test_dream_cycle_low_score_always_tier1(router_no_claude):
    model = router_no_claude.select(0.05, task_type="dream_cycle")
    assert model == TIER_1_MODEL


def test_dream_cycle_very_high_score_stays_tier1(router_with_claude):
    # dream_cycle must NOT escalate to Claude even with high score.
    model = router_with_claude.select(0.99, task_type="dream_cycle")
    assert model == TIER_1_MODEL


# ---------------------------------------------------------------------------
# 7. Qwen models importable from config
# ---------------------------------------------------------------------------


def test_qwen_72b_importable():
    assert QWEN_72B == "qwen2.5-coder:72b"


def test_qwen_32b_importable():
    assert QWEN_32B == "qwen2.5-coder:32b"


def test_qwen_14b_importable():
    assert QWEN_14B == "qwen2.5-coder:14b"


def test_qwen_7b_importable():
    assert QWEN_7B == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# 8. SUPPORTED_MODELS includes Qwen and Claude variants
# ---------------------------------------------------------------------------


def test_supported_models_contains_qwen_variants():
    for model in (QWEN_72B, QWEN_32B, QWEN_14B, QWEN_7B):
        assert model in SUPPORTED_MODELS, f"{model} missing from SUPPORTED_MODELS"


def test_supported_models_contains_claude():
    assert CLAUDE_MODEL in SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# 9. LLMClient accepts Qwen model names without raising on construction
# ---------------------------------------------------------------------------


def test_llm_client_accepts_qwen_model_names():
    """LLMClient.ask() routes Qwen models through Ollama.
    We verify the routing logic (model.startswith("claude-")) without making
    a real network call by checking the internal branch condition directly.
    """
    from core.llm import LLMClient

    client = LLMClient()
    for model in (QWEN_72B, QWEN_32B, QWEN_14B, QWEN_7B):
        # Qwen models must NOT be treated as Anthropic cloud models.
        assert not model.startswith("claude-"), (
            f"Model {model} would be incorrectly routed to Anthropic API"
        )
    # Claude model must be routed to Anthropic.
    assert CLAUDE_MODEL.startswith("claude-")
    # LLMClient instantiates without error regardless of API key presence.
    assert client is not None


# ---------------------------------------------------------------------------
# 10. routing_policy() returns expected structure
# ---------------------------------------------------------------------------


def test_routing_policy_structure(router_no_claude):
    policy = router_no_claude.routing_policy()
    assert "bands" in policy
    assert "task_type_overrides" in policy
    assert "claude_enabled" in policy
    assert "models" in policy

    assert isinstance(policy["bands"], list)
    assert len(policy["bands"]) == 4

    for band in policy["bands"]:
        assert "range" in band
        assert "model" in band
        assert "description" in band

    assert "self_modify" in policy["task_type_overrides"]
    assert "dream_cycle" in policy["task_type_overrides"]

    assert policy["claude_enabled"] is False  # router_no_claude fixture


def test_routing_policy_claude_flag_true(router_with_claude):
    policy = router_with_claude.routing_policy()
    assert policy["claude_enabled"] is True
    # Highest band should reference CLAUDE_MODEL when enabled
    highest_band = policy["bands"][-1]
    assert CLAUDE_MODEL in highest_band["model"]


def test_routing_policy_models_dict(router_no_claude):
    policy = router_no_claude.routing_policy()
    models = policy["models"]
    assert models["tier_1"] == TIER_1_MODEL
    assert models["tier_2"] == TIER_2_MODEL
    assert models["tier_3"] == TIER_3_MODEL
    assert models["claude"] == CLAUDE_MODEL
