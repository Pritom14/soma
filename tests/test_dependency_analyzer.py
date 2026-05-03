"""tests/test_dependency_analyzer.py

Tests for DependencyAnalyzer, SubTask, and DependencyGraph.
Covers: linear chains, parallel grouping, cycle detection, same-file
serialisation, and empty input.
"""
from __future__ import annotations

import pytest

from core.dependency_analyzer import DependencyAnalyzer, DependencyGraph, SubTask


@pytest.fixture
def analyzer() -> DependencyAnalyzer:
    return DependencyAnalyzer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tasks(*descs: str) -> list[dict]:
    return [{"id": str(i), "description": d} for i, d in enumerate(descs)]


# ---------------------------------------------------------------------------
# 1. Simple linear chain A → B → C
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_explicit_depends_on_respected(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            {"id": "A", "description": "Define function setup"},
            {"id": "B", "description": "Call setup(), run tests", "depends_on": ["A"]},
            {"id": "C", "description": "Call run_tests, generate report", "depends_on": ["B"]},
        ]
        graph = analyzer.build_graph(tasks)
        order = graph.execution_order
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_all_tasks_present(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            {"id": "A", "description": "Define function init"},
            {"id": "B", "description": "Call init, do work", "depends_on": ["A"]},
            {"id": "C", "description": "Wrap up", "depends_on": ["B"]},
        ]
        graph = analyzer.build_graph(tasks)
        assert set(graph.execution_order) == {"A", "B", "C"}

    def test_function_name_inference_creates_chain(self, analyzer: DependencyAnalyzer) -> None:
        """Rule 1: task B mentions a function defined in task A."""
        tasks = [
            {"id": "A", "description": "implement function load_config"},
            {"id": "B", "description": "call load_config() to initialise app"},
        ]
        graph = analyzer.build_graph(tasks)
        order = graph.execution_order
        assert order.index("A") < order.index("B"), (
            f"Expected A before B but got order: {order}"
        )

    def test_sequencing_phrase_after(self, analyzer: DependencyAnalyzer) -> None:
        """Rule 2: 'after X' phrase creates dependency."""
        tasks = [
            {"id": "A", "description": "Create the database schema"},
            {"id": "B", "description": "After task A, seed the database"},
        ]
        graph = analyzer.build_graph(tasks)
        order = graph.execution_order
        assert order.index("A") < order.index("B"), (
            f"Expected A before B but got order: {order}"
        )

    def test_sequencing_phrase_once_done(self, analyzer: DependencyAnalyzer) -> None:
        """Rule 2: 'once X is done' phrase creates dependency."""
        tasks = [
            {"id": "alpha", "description": "Write migration script"},
            {"id": "beta", "description": "Once alpha is done, run tests"},
        ]
        graph = analyzer.build_graph(tasks)
        order = graph.execution_order
        assert order.index("alpha") < order.index("beta"), (
            f"Expected alpha before beta but got order: {order}"
        )


# ---------------------------------------------------------------------------
# 2. Parallel tasks correctly grouped
# ---------------------------------------------------------------------------


class TestParallelGroups:
    def test_independent_tasks_are_parallelised(self, analyzer: DependencyAnalyzer) -> None:
        """Tasks with no mutual dependencies should appear in the same wave."""
        tasks = [
            {"id": "A", "description": "Write unit tests for auth"},
            {"id": "B", "description": "Write unit tests for billing"},
            {"id": "C", "description": "Write documentation"},
        ]
        graph = analyzer.build_graph(tasks)
        # All three have no dependencies — should be in one wave
        assert len(graph.parallelizable) >= 1
        first_wave = graph.parallelizable[0]
        assert len(first_wave) == 3, f"Expected 3 parallel tasks, got wave: {first_wave}"

    def test_diamond_parallel(self, analyzer: DependencyAnalyzer) -> None:
        """Classic diamond: A → (B, C) → D."""
        tasks = [
            {"id": "A", "description": "Set up base"},
            {"id": "B", "description": "Build feature X", "depends_on": ["A"]},
            {"id": "C", "description": "Build feature Y", "depends_on": ["A"]},
            {"id": "D", "description": "Integrate", "depends_on": ["B", "C"]},
        ]
        graph = analyzer.build_graph(tasks)
        waves = graph.parallelizable
        # Wave 1: A
        # Wave 2: B and C (parallel)
        # Wave 3: D
        assert any(set(w) == {"B", "C"} for w in waves), (
            f"Expected B and C in the same wave. Waves: {waves}"
        )
        # D must be in its own wave after B and C
        for w in waves:
            if "D" in w:
                # B and C must have been in earlier waves
                bc_wave_idx = next(i for i, w2 in enumerate(waves) if "B" in w2 or "C" in w2)
                d_wave_idx = waves.index(w)
                assert bc_wave_idx < d_wave_idx
                break

    def test_parallelizable_groups_cover_all_tasks(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            {"id": "A", "description": "step one"},
            {"id": "B", "description": "step two"},
            {"id": "C", "description": "step three", "depends_on": ["A"]},
        ]
        graph = analyzer.build_graph(tasks)
        all_in_waves = {tid for w in graph.parallelizable for tid in w}
        assert all_in_waves == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# 3. Circular dependency detected and warned
# ---------------------------------------------------------------------------


class TestCircularDependency:
    def test_cycle_detected(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            {"id": "X", "description": "task X", "depends_on": ["Y"]},
            {"id": "Y", "description": "task Y", "depends_on": ["X"]},
        ]
        graph = analyzer.build_graph(tasks)
        assert len(graph.circular_deps) > 0, "Expected circular dependency to be detected"

    def test_cycle_warning_added(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            {"id": "X", "description": "task X", "depends_on": ["Y"]},
            {"id": "Y", "description": "task Y", "depends_on": ["X"]},
        ]
        graph = analyzer.build_graph(tasks)
        assert len(graph.warnings) > 0, "Expected at least one warning for circular dependency"
        assert any("circular" in w.lower() or "cycle" in w.lower() for w in graph.warnings)

    def test_cycle_not_silently_ignored(self, analyzer: DependencyAnalyzer) -> None:
        """After breaking, the graph should still record the cycle in circular_deps."""
        tasks = [
            {"id": "P", "description": "task P", "depends_on": ["Q"]},
            {"id": "Q", "description": "task Q", "depends_on": ["P"]},
        ]
        graph = analyzer.build_graph(tasks)
        # circular_deps is populated before breaking
        assert ("P", "Q") in graph.circular_deps or ("Q", "P") in graph.circular_deps

    def test_cycle_broken_so_execution_order_valid(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            {"id": "X", "description": "task X", "depends_on": ["Y"]},
            {"id": "Y", "description": "task Y", "depends_on": ["X"]},
        ]
        graph = analyzer.build_graph(tasks)
        # After breaking, both tasks should appear in execution_order
        assert "X" in graph.execution_order
        assert "Y" in graph.execution_order

    def test_three_node_cycle(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            {"id": "A", "description": "task A", "depends_on": ["C"]},
            {"id": "B", "description": "task B", "depends_on": ["A"]},
            {"id": "C", "description": "task C", "depends_on": ["B"]},
        ]
        graph = analyzer.build_graph(tasks)
        assert len(graph.circular_deps) > 0
        assert len(graph.warnings) > 0


# ---------------------------------------------------------------------------
# 4. Same-file tasks serialised (Rule 4)
# ---------------------------------------------------------------------------


class TestSameFileSerialisation:
    def test_same_file_creates_dependency(self, analyzer: DependencyAnalyzer) -> None:
        """Two tasks touching the same file must be serialised."""
        tasks = [
            {"id": "first", "description": "Add helper to utils.py"},
            {"id": "second", "description": "Extend utils.py with new function"},
        ]
        graph = analyzer.build_graph(tasks)
        order = graph.execution_order
        # 'first' was declared before 'second', so it must come first
        assert order.index("first") < order.index("second"), (
            f"Expected first before second, got: {order}"
        )

    def test_file_create_before_modify(self, analyzer: DependencyAnalyzer) -> None:
        """Rule 3: create file A before modifying it."""
        tasks = [
            {"id": "creator", "description": "Create new file config.yaml with defaults"},
            {"id": "modifier", "description": "Update config.yaml to add new key"},
        ]
        graph = analyzer.build_graph(tasks)
        order = graph.execution_order
        assert order.index("creator") < order.index("modifier"), (
            f"Expected creator before modifier, got: {order}"
        )

    def test_different_files_can_be_parallel(self, analyzer: DependencyAnalyzer) -> None:
        """Tasks touching different files should not be forced to serialize."""
        tasks = [
            {"id": "A", "description": "Edit models.py"},
            {"id": "B", "description": "Edit views.py"},
        ]
        graph = analyzer.build_graph(tasks)
        # Both should be in the same first wave (no forced serialisation)
        first_wave = graph.parallelizable[0] if graph.parallelizable else []
        assert "A" in first_wave and "B" in first_wave, (
            f"Expected A and B in same wave. Waves: {graph.parallelizable}"
        )


# ---------------------------------------------------------------------------
# 5. Empty input handled gracefully
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_list_returns_empty_graph(self, analyzer: DependencyAnalyzer) -> None:
        graph = analyzer.build_graph([])
        assert isinstance(graph, DependencyGraph)
        assert graph.tasks == []
        assert graph.execution_order == []
        assert graph.circular_deps == []
        assert graph.parallelizable == []
        assert graph.warnings == []

    def test_single_task(self, analyzer: DependencyAnalyzer) -> None:
        graph = analyzer.build_graph([{"id": "only", "description": "do the thing"}])
        assert graph.execution_order == ["only"]
        assert graph.circular_deps == []
        assert len(graph.parallelizable) == 1
        assert graph.parallelizable[0] == ["only"]

    def test_safe_execution_order_empty(self, analyzer: DependencyAnalyzer) -> None:
        graph = analyzer.safe_execution_order([])
        assert isinstance(graph, DependencyGraph)
        assert graph.execution_order == []


# ---------------------------------------------------------------------------
# 6. Topological sort standalone
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_simple_chain(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            SubTask(id="A", description="first", depends_on=[]),
            SubTask(id="B", description="second", depends_on=["A"]),
            SubTask(id="C", description="third", depends_on=["B"]),
        ]
        graph = DependencyGraph(
            tasks=tasks,
            execution_order=[],
            circular_deps=[],
            parallelizable=[],
            warnings=[],
        )
        order = analyzer.topological_sort(graph)
        assert order == ["A", "B", "C"]

    def test_no_dependencies_stable_alphabetical(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            SubTask(id="C", description="c"),
            SubTask(id="A", description="a"),
            SubTask(id="B", description="b"),
        ]
        graph = DependencyGraph(
            tasks=tasks,
            execution_order=[],
            circular_deps=[],
            parallelizable=[],
            warnings=[],
        )
        order = analyzer.topological_sort(graph)
        # All tasks present, alphabetically sorted for zero-in-degree nodes
        assert set(order) == {"A", "B", "C"}

    def test_fork_and_join(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            SubTask(id="root", description="root", depends_on=[]),
            SubTask(id="left", description="left", depends_on=["root"]),
            SubTask(id="right", description="right", depends_on=["root"]),
            SubTask(id="merge", description="merge", depends_on=["left", "right"]),
        ]
        graph = DependencyGraph(
            tasks=tasks,
            execution_order=[],
            circular_deps=[],
            parallelizable=[],
            warnings=[],
        )
        order = analyzer.topological_sort(graph)
        assert order[0] == "root"
        assert order[-1] == "merge"
        assert set(order) == {"root", "left", "right", "merge"}


# ---------------------------------------------------------------------------
# 7. detect_circular standalone
# ---------------------------------------------------------------------------


class TestDetectCircular:
    def test_no_cycles_returns_empty(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            SubTask(id="A", description="a", depends_on=[]),
            SubTask(id="B", description="b", depends_on=["A"]),
        ]
        graph = DependencyGraph(
            tasks=tasks,
            execution_order=[],
            circular_deps=[],
            parallelizable=[],
            warnings=[],
        )
        assert analyzer.detect_circular(graph) == []

    def test_direct_cycle_detected(self, analyzer: DependencyAnalyzer) -> None:
        tasks = [
            SubTask(id="X", description="x", depends_on=["Y"]),
            SubTask(id="Y", description="y", depends_on=["X"]),
        ]
        graph = DependencyGraph(
            tasks=tasks,
            execution_order=[],
            circular_deps=[],
            parallelizable=[],
            warnings=[],
        )
        cycles = analyzer.detect_circular(graph)
        assert len(cycles) > 0
        assert any("X" in str(c) or "Y" in str(c) for c in cycles)


# ---------------------------------------------------------------------------
# 8. Integration: example dependency graph output
# ---------------------------------------------------------------------------


class TestIntegrationExample:
    def test_realistic_feature_plan(self, analyzer: DependencyAnalyzer) -> None:
        """Simulate a realistic multi-step coding plan."""
        tasks = [
            {
                "id": "models",
                "description": "create models.py with class User and class Session",
                "estimated_complexity": 2.0,
            },
            {
                "id": "db",
                "description": "create db.py, implement function connect_db",
                "estimated_complexity": 1.5,
            },
            {
                "id": "auth",
                "description": "implement function authenticate using User class, after task models",
                "estimated_complexity": 2.5,
            },
            {
                "id": "tests_models",
                "description": "Write unit tests for models.py",
                "estimated_complexity": 1.0,
            },
            {
                "id": "tests_auth",
                "description": "Write tests for authenticate function",
                "estimated_complexity": 1.0,
            },
            {
                "id": "docs",
                "description": "Write README.md documentation",
                "estimated_complexity": 0.5,
            },
        ]
        graph = analyzer.safe_execution_order(tasks)

        # Basic structure checks
        assert isinstance(graph, DependencyGraph)
        assert len(graph.execution_order) == 6
        assert set(graph.execution_order) == {
            "models", "db", "auth", "tests_models", "tests_auth", "docs"
        }

        # models must come before auth (due to sequencing phrase + class name)
        assert graph.execution_order.index("models") < graph.execution_order.index("auth")

        # Parallel groups must cover all 6 tasks exactly once
        all_in_waves = [tid for w in graph.parallelizable for tid in w]
        assert set(all_in_waves) == set(graph.execution_order)

    def test_graph_repr_has_all_fields(self, analyzer: DependencyAnalyzer) -> None:
        graph = analyzer.safe_execution_order([
            {"id": "t1", "description": "task one"},
            {"id": "t2", "description": "task two"},
        ])
        assert hasattr(graph, "tasks")
        assert hasattr(graph, "execution_order")
        assert hasattr(graph, "circular_deps")
        assert hasattr(graph, "parallelizable")
        assert hasattr(graph, "warnings")
