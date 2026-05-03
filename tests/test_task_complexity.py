"""tests/test_task_complexity.py

Tests for TaskComplexityScorer: each scoring factor, threshold boundaries,
and real task strings.
"""
from __future__ import annotations

import pytest

from core.task_complexity import ComplexityScore, TaskComplexityScorer


@pytest.fixture
def scorer():
    return TaskComplexityScorer()


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------


def test_complexity_score_fields():
    cs = ComplexityScore(
        score=0.5,
        file_count=2,
        operation_count=3,
        nesting_depth=1,
        special_char_penalty=0.1,
        recommendation="decompose",
        reasons=["test reason"],
    )
    assert cs.score == 0.5
    assert cs.recommendation == "decompose"
    assert "test reason" in cs.reasons


# ---------------------------------------------------------------------------
# Individual scoring factors
# ---------------------------------------------------------------------------


def test_file_count_factor_above_threshold(scorer):
    """file_count > 3 adds 0.3 to score."""
    file_paths = ["a.py", "b.py", "c.py", "d.py"]  # 4 files
    cs = scorer.score("update each file", file_paths=file_paths)
    assert cs.file_count == 4
    # At least the file_count contribution should be present
    assert cs.score >= 0.3
    assert any("file" in r for r in cs.reasons)


def test_file_count_factor_at_threshold_triggered(scorer):
    """file_count == 3 now adds 0.3 (threshold is >= 3)."""
    file_paths = ["a.py", "b.py", "c.py"]
    cs = scorer.score("simple task", file_paths=file_paths)
    # file_count penalty IS triggered since count is >= 3
    assert cs.file_count == 3
    assert cs.score >= 0.3
    assert any("file" in r for r in cs.reasons)


def test_operation_count_factor(scorer):
    """More than 5 distinct operation verbs adds 0.25."""
    task = "add the function then remove the old class then rename and replace and insert and delete some code"
    cs = scorer.score(task)
    assert cs.operation_count > 5
    assert cs.score >= 0.25
    assert any("operation" in r for r in cs.reasons)


def test_operation_count_at_boundary_not_triggered(scorer):
    """Exactly 5 distinct operations does NOT add 0.25."""
    task = "add remove replace insert delete the item"
    cs = scorer.score(task)
    # 5 unique ops → no penalty
    assert cs.operation_count <= 5
    assert not any("operation" in r for r in cs.reasons)


def test_nesting_depth_factor_class_in_class(scorer):
    """Detecting class inside class bumps nesting depth."""
    task = "In class Outer add a class Inner with a method"
    cs = scorer.score(task)
    assert cs.nesting_depth >= 1


def test_nesting_depth_factor_nested_def(scorer):
    """Detecting def inside def bumps nesting depth."""
    task = "def outer(): then def inner(): — wire them together"
    cs = scorer.score(task)
    assert cs.nesting_depth >= 1


def test_special_char_triple_quotes(scorer):
    """Triple-quoted strings add 0.1 to penalty."""
    task = 'Add a docstring: """This is a docstring""" to the function'
    cs = scorer.score(task)
    assert cs.special_char_penalty >= 0.1
    assert any("triple" in r for r in cs.reasons)


def test_special_char_decorator(scorer):
    """Decorator @ symbols add 0.1 to penalty."""
    task = "Apply @property decorator to the getter"
    cs = scorer.score(task)
    assert cs.special_char_penalty >= 0.1
    assert any("decorator" in r for r in cs.reasons)


def test_special_char_metaclass(scorer):
    """metaclass keyword adds 0.1 to penalty."""
    task = "Change metaclass=ABCMeta on the base class"
    cs = scorer.score(task)
    assert cs.special_char_penalty >= 0.1
    assert any("metaclass" in r for r in cs.reasons)


def test_task_length_factor(scorer):
    """Task longer than 500 chars adds 0.1."""
    long_task = "update something " * 35  # > 500 chars
    assert len(long_task) > 500
    cs = scorer.score(long_task)
    assert cs.score >= 0.1
    assert any("500" in r or "length" in r for r in cs.reasons)


def test_keyword_refactor(scorer):
    """'refactor' keyword adds 0.15."""
    cs = scorer.score("refactor the entire authentication module")
    assert cs.score >= 0.15
    assert any("refactor" in r for r in cs.reasons)


def test_keyword_migrate(scorer):
    """'migrate' keyword adds 0.15."""
    cs = scorer.score("migrate the database schema to Postgres")
    assert cs.score >= 0.15


def test_keyword_rename_across(scorer):
    """'rename across' adds 0.15."""
    cs = scorer.score("rename across all files from foo to bar")
    assert cs.score >= 0.15


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------


def test_simple_task_is_direct(scorer):
    cs = scorer.score("add a comment to main.py", file_paths=["main.py"])
    assert cs.recommendation == "direct"
    assert not scorer.should_decompose(cs)
    assert not scorer.should_reject(cs)


def test_decompose_threshold_boundary(scorer):
    """A score >= 0.6 should recommend decompose."""
    # file_count > 3 (0.3) + operation_count > 5 (0.25) + keyword refactor (0.15) = 0.70
    task = "refactor: add remove replace insert delete rename change the module"
    cs = scorer.score(task, file_paths=["a.py", "b.py", "c.py", "d.py"])
    assert cs.score >= TaskComplexityScorer.DECOMPOSE_THRESHOLD
    assert cs.recommendation in ("decompose", "reject")
    assert scorer.should_decompose(cs) or scorer.should_reject(cs)


def test_reject_threshold_boundary(scorer):
    """A score >= 0.9 should recommend reject."""
    # Stack all factors:
    #   file_count > 3      (0.30)
    #   op_count > 5        (0.25)
    #   nesting_depth > 2   (0.20)
    #   triple_quotes       (0.10)
    #   decorator           (0.10)
    #   keyword refactor    (0.15)
    # Total = 1.10 → clamped to 1.0
    task = (
        'refactor: add remove replace insert delete rename change '
        'class Outer: class Inner: def foo(): def bar(): '
        '@property @staticmethod """docstring"""'
    )
    cs = scorer.score(task, file_paths=["a.py", "b.py", "c.py", "d.py", "e.py"])
    assert cs.score >= TaskComplexityScorer.REJECT_THRESHOLD, (
        f"Expected score >= 0.9, got {cs.score}. Reasons: {cs.reasons}"
    )
    assert cs.recommendation == "reject"
    assert scorer.should_reject(cs)
    assert not scorer.should_decompose(cs)


def test_score_clamped_at_one(scorer):
    """Score never exceeds 1.0."""
    huge_task = (
        'migrate rename across refactor class Foo: class Bar: def x(): def y(): '
        '@property @staticmethod metaclass=Meta """doc""" '
        * 10
    )
    cs = scorer.score(huge_task, file_paths=["a.py", "b.py", "c.py", "d.py", "e.py"])
    assert cs.score <= 1.0


# ---------------------------------------------------------------------------
# Real-world task strings
# ---------------------------------------------------------------------------


def test_three_file_task_routes_to_decompose(scorer):
    """A 3-file refactor task should score >= 0.6 and route to decomposition."""
    # 3 files (0.3) + refactor keyword (0.15) + operations (0.25) = 0.70 → decompose
    task = "refactor: add remove replace insert delete and rename across modules"
    cs = scorer.score(task, file_paths=["a.py", "b.py", "c.py"])
    assert cs.file_count == 3
    assert cs.score >= TaskComplexityScorer.DECOMPOSE_THRESHOLD
    assert cs.recommendation in ("decompose", "reject")


def test_two_file_task_routes_to_direct(scorer):
    """A 2-file simple task should route to direct."""
    task = "update both files"
    cs = scorer.score(task, file_paths=["a.py", "b.py"])
    assert cs.file_count == 2
    assert cs.recommendation == "direct"


def test_real_simple_task(scorer):
    cs = scorer.score(
        "In core/utils.py, add a helper function `slugify(text: str) -> str`.",
        file_paths=["core/utils.py"],
    )
    assert cs.recommendation == "direct"


def test_real_moderate_task(scorer):
    cs = scorer.score(
        "rename across all callers: rename `route()` to `dispatch()`, "
        "update the import, replace the old alias, remove dead code, "
        "add a deprecation warning, insert a shim for compat, "
        "and delete the legacy test in orchestrator.py and main.py.",
        file_paths=["core/router.py", "core/executor.py", "orchestrator.py", "main.py"],
    )
    # 4 files (0.3) + rename across keyword (0.15) + ops > 5 (0.25) = 0.70 → decompose
    assert cs.recommendation in ("decompose", "reject"), (
        f"Expected decompose/reject, got {cs.recommendation!r} "
        f"(score={cs.score}, ops={cs.operation_count}, reasons={cs.reasons})"
    )


def test_real_complex_migration(scorer):
    cs = scorer.score(
        "Migrate the entire codebase from the old BeliefStore API to the new BeliefIndex API. "
        "Rename all usages, update imports across every module, refactor the confidence "
        "calculation, and add metaclass=ABCMeta to the base belief class. "
        "Also update the @property decorators on BeliefStore attributes.",
        file_paths=["core/belief.py", "core/belief_index.py", "orchestrator.py",
                    "core/router.py", "core/planner.py"],
    )
    assert cs.recommendation in ("decompose", "reject")
    assert cs.score >= 0.6


# ---------------------------------------------------------------------------
# Legacy API backward-compat
# ---------------------------------------------------------------------------


def test_legacy_score_task():
    from core.task_complexity import score_task, is_complex

    score, reason = score_task("add a helper", {})
    assert isinstance(score, int)
    assert isinstance(reason, str)


def test_legacy_is_complex_simple():
    from core.task_complexity import is_complex

    assert not is_complex("add a helper", {})


def test_legacy_is_complex_triggered():
    from core.task_complexity import is_complex

    # 4 files touched → (4-1)*10 = 30; "rename" + "replace" + "add" + "remove" = 4 distinct ops
    # → (4-2)*5 = 10; total = 40 > 30 threshold
    assert is_complex(
        "rename and replace and add and remove things across all modules",
        {"a.py": "", "b.py": "", "c.py": "", "d.py": ""},
    )
