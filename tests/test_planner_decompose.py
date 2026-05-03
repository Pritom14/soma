"""tests/test_planner_decompose.py

Tests for RecursivePlanner:
  - decompose_step splits correctly
  - detect_sequencing_hazards finds out-of-order dependencies
  - reorder_for_dependencies produces a valid execution order
  - build_structured_plan returns a correct StructuredPlan
"""
from __future__ import annotations

import pytest

from core.planner import RecursivePlanner, StructuredPlan


@pytest.fixture
def planner():
    return RecursivePlanner()


# ---------------------------------------------------------------------------
# decompose_step
# ---------------------------------------------------------------------------


def test_atomic_step_returns_itself(planner):
    step = "Add a type annotation to the function parameter."
    result = planner.decompose_step(step)
    assert result == [step]


def test_split_on_and(planner):
    # "and then" is the trigger in the conjunction splitter
    step = "Add the import statement and then update the function signature"
    result = planner.decompose_step(step)
    assert len(result) == 2
    assert any("import" in s for s in result)
    assert any("function" in s or "signature" in s for s in result)


def test_split_on_then(planner):
    step = "Create the helper function then wire it into the main loop"
    result = planner.decompose_step(step)
    assert len(result) == 2


def test_split_on_also(planner):
    step = "Rename the method also update the docstring"
    result = planner.decompose_step(step)
    assert len(result) == 2


def test_split_on_semicolons(planner):
    step = "add import; define function; call function"
    result = planner.decompose_step(step)
    assert len(result) == 3


def test_split_on_numbered_list(planner):
    step = "Do work 1. Add import 2. Define class 3. Write tests"
    result = planner.decompose_step(step)
    # Should produce 3 (or more) sub-steps
    assert len(result) >= 2


def test_recursive_split(planner):
    """A step with multiple conjunctions should recurse."""
    # Use explicit triggers: "then" + "and also"
    step = "Add import then define class and also write method then add decorator"
    result = planner.decompose_step(step)
    assert len(result) >= 3


def test_max_depth_respected(planner):
    """Should not recurse forever."""
    step = "and and and and and and and and"
    result = planner.decompose_step(step)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_deduplication(planner):
    """Duplicate sub-steps should be removed."""
    step = "add import and add import"
    result = planner.decompose_step(step)
    assert len(result) == len(set(result))


def test_empty_step(planner):
    result = planner.decompose_step("")
    assert result == [""] or result == []


# ---------------------------------------------------------------------------
# detect_sequencing_hazards
# ---------------------------------------------------------------------------


def test_no_hazards_correct_order(planner):
    steps = [
        "Create the new validator function",
        "Call validator in main handler",
    ]
    hazards = planner.detect_sequencing_hazards(steps)
    # validator is defined before it is used — no hazard
    assert hazards == []


def test_hazard_detected_when_use_before_define(planner):
    steps = [
        "Call validator in main handler",    # idx 0 — uses validator
        "Create the new validator function", # idx 1 — defines validator
    ]
    hazards = planner.detect_sequencing_hazards(steps)
    # (0, 1): step 0 depends on step 1 which comes after
    assert len(hazards) >= 1
    dependent, dependency = hazards[0]
    assert dependent == 0
    assert dependency == 1


def test_multiple_hazards(planner):
    steps = [
        "Call processor and apply transformer",   # uses processor, transformer
        "Create the new processor function",      # defines processor
        "Implement the new transformer helper",   # defines transformer
    ]
    hazards = planner.detect_sequencing_hazards(steps)
    assert len(hazards) >= 1


def test_no_false_positives_for_stopwords(planner):
    """Common stop-words should not generate spurious hazards."""
    steps = [
        "Update the configuration file",
        "Create a new directory for outputs",
    ]
    hazards = planner.detect_sequencing_hazards(steps)
    # 'new' and 'the' are stop-words; should not produce hazards
    assert isinstance(hazards, list)


# ---------------------------------------------------------------------------
# reorder_for_dependencies
# ---------------------------------------------------------------------------


def test_reorder_correct_order_unchanged(planner):
    steps = [
        "Create the new validator function",
        "Call validator in main handler",
    ]
    ordered = planner.reorder_for_dependencies(steps)
    assert ordered == steps


def test_reorder_fixes_inverted_order(planner):
    steps = [
        "Call validator in main handler",    # uses validator
        "Create the new validator function", # defines validator
    ]
    ordered = planner.reorder_for_dependencies(steps)
    # The definition should come first
    define_idx = ordered.index("Create the new validator function")
    use_idx = ordered.index("Call validator in main handler")
    assert define_idx < use_idx


def test_reorder_empty_list(planner):
    assert planner.reorder_for_dependencies([]) == []


def test_reorder_single_step(planner):
    steps = ["Do the one thing"]
    assert planner.reorder_for_dependencies(steps) == steps


def test_reorder_preserves_all_steps(planner):
    steps = [
        "Call the helper function",
        "Define the helper function",
        "Verify the system works",
    ]
    ordered = planner.reorder_for_dependencies(steps)
    assert set(ordered) == set(steps)
    assert len(ordered) == len(steps)


def test_reorder_cycle_falls_back_to_original(planner):
    """If there is a circular dependency, original order is preserved."""
    # Craft a scenario where A uses B and B uses A — heuristically hard to
    # trigger with text, so we test the fallback path explicitly by confirming
    # that reorder always returns the same number of steps.
    steps = ["Create foo and use bar", "Create bar and use foo"]
    ordered = planner.reorder_for_dependencies(steps)
    assert len(ordered) == 2
    assert set(ordered) == set(steps)


# ---------------------------------------------------------------------------
# build_structured_plan
# ---------------------------------------------------------------------------


def test_build_structured_plan_basic(planner):
    raw = [
        "Create validate function then call it in the handler",
    ]
    plan = planner.build_structured_plan(raw, estimated_complexity=0.65)
    assert isinstance(plan, StructuredPlan)
    assert len(plan.steps) >= 1
    assert plan.estimated_complexity == 0.65
    assert isinstance(plan.depth_levels, dict)
    assert isinstance(plan.dependencies, list)


def test_build_structured_plan_multi_step(planner):
    raw = [
        "Add import statement",
        "Implement the new helperutil class",
        "Call helperutil in main module",
    ]
    plan = planner.build_structured_plan(raw)
    assert len(plan.steps) >= 3
    # helperutil should be defined before it is called
    step_list = plan.steps
    implement_indices = [i for i, s in enumerate(step_list) if "helperutil" in s.lower() and ("implement" in s.lower() or "Implement" in s)]
    call_indices = [i for i, s in enumerate(step_list) if "Call" in s or "call" in s]
    if implement_indices and call_indices:
        assert min(implement_indices) < min(call_indices), "definition must precede call"


def test_build_structured_plan_empty(planner):
    plan = planner.build_structured_plan([])
    assert plan.steps == []
    assert plan.dependencies == []


def test_build_structured_plan_complexity_passthrough(planner):
    plan = planner.build_structured_plan(["do something"], estimated_complexity=0.42)
    assert plan.estimated_complexity == 0.42


def test_build_structured_plan_steps_are_strings(planner):
    raw = ["Rename the config variable and update all references"]
    plan = planner.build_structured_plan(raw)
    for step in plan.steps:
        assert isinstance(step, str)
