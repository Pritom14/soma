from __future__ import annotations

"""
core/dependency_analyzer.py - DAG-based sub-task dependency analysis.

Builds a dependency graph from sub-task descriptions using purely lexical rules
(no LLM). Detects cycles via DFS, topologically sorts, and finds parallel groups.
"""

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    estimated_complexity: float = 1.0


@dataclass
class DependencyGraph:
    tasks: list[SubTask]
    execution_order: list[str]          # topologically sorted task ids
    circular_deps: list[tuple]          # (task_a, task_b) pairs forming cycles
    parallelizable: list[list[str]]     # groups that can run in parallel
    warnings: list[str]


# ---------------------------------------------------------------------------
# Legacy support dataclass (used by planner.py)
# ---------------------------------------------------------------------------


@dataclass
class DepNode:
    step_index: int
    defines: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DependencyAnalyzer
# ---------------------------------------------------------------------------


class DependencyAnalyzer:
    """
    Build and analyse DAG-based sub-task dependency graphs.

    Dependency detection rules (lexical, no LLM):
      1. If task B mentions a function/class name defined in task A → B depends on A.
      2. If task B contains sequencing phrases referencing task A's id or description
         fragment ("after X", "following X", "once X is done") → B depends on A.
      3. If task A creates a file that task B modifies → A must precede B.
      4. If two tasks touch the same file → serialize them (earlier listed first).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self, subtasks: list[dict]) -> DependencyGraph:
        """Convert raw dicts to SubTask objects and compute graph metadata."""
        tasks = self._parse_tasks(subtasks)
        self._infer_dependencies(tasks)

        graph = DependencyGraph(
            tasks=tasks,
            execution_order=[],
            circular_deps=[],
            parallelizable=[],
            warnings=[],
        )

        graph.circular_deps = self.detect_circular(graph)
        if graph.circular_deps:
            self._break_cycles(graph)

        graph.execution_order = self.topological_sort(graph)
        graph.parallelizable = self.find_parallel_groups(graph)
        return graph

    def detect_circular(self, graph: DependencyGraph) -> list[tuple]:
        """
        DFS cycle detection. Returns list of (task_a_id, task_b_id) pairs where
        the edge task_a → task_b completes a back edge.
        """
        adj = self._adjacency(graph.tasks)
        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles: list[tuple] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            for neighbour in adj.get(node, []):
                if neighbour not in visited:
                    dfs(neighbour)
                elif neighbour in rec_stack:
                    cycles.append((node, neighbour))
            rec_stack.discard(node)

        for task in graph.tasks:
            if task.id not in visited:
                dfs(task.id)

        return cycles

    def topological_sort(self, graph: DependencyGraph) -> list[str]:
        """
        Kahn's algorithm. Returns task ids in a valid execution order.
        If the graph is acyclic after cycle-breaking, returns full ordering.
        Falls back to declaration order if residual cycles remain.
        """
        task_ids = [t.id for t in graph.tasks]
        adj = self._adjacency(graph.tasks)

        # in_degree[X] = number of prerequisites X is still waiting for
        in_degree: dict[str, int] = {tid: 0 for tid in task_ids}
        for tid in task_ids:
            for dep in adj.get(tid, []):
                if dep in in_degree:
                    in_degree[tid] += 1  # tid depends on dep, so tid's count rises

        queue = sorted([tid for tid, deg in in_degree.items() if deg == 0])
        order: list[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            # Reduce in_degree of every task that listed `node` as a dependency
            for candidate in sorted(task_ids):
                if node in adj.get(candidate, []):
                    in_degree[candidate] -= 1
                    if in_degree[candidate] == 0:
                        queue.append(candidate)
                        queue.sort()

        if len(order) < len(task_ids):
            # Residual cycle — append remaining in declaration order
            seen = set(order)
            order += [tid for tid in task_ids if tid not in seen]

        return order

    def find_parallel_groups(self, graph: DependencyGraph) -> list[list[str]]:
        """
        Return execution waves: tasks in the same wave have no mutual dependencies
        and no shared files, and all their prerequisites are satisfied in earlier waves.

        Example result: [[A, B], [C], [D, E]] means A and B can run concurrently,
        then C, then D and E concurrently.
        """
        task_map = {t.id: t for t in graph.tasks}
        adj = self._adjacency(graph.tasks)          # id → [ids it depends on]
        file_map = self._file_map(graph.tasks)       # task_id → set[files]

        remaining = set(graph.execution_order)
        completed: set[str] = set()
        waves: list[list[str]] = []

        while remaining:
            # Tasks whose dependencies are all satisfied
            ready = [
                tid for tid in graph.execution_order
                if tid in remaining
                and all(dep in completed for dep in adj.get(tid, []))
            ]
            if not ready:
                # Safety: avoid infinite loop on residual cycles
                waves.append(sorted(remaining))
                break

            # Further split ready tasks by shared-file conflicts
            wave = self._wave_no_file_conflicts(ready, file_map)
            waves.append(wave)
            completed.update(wave)
            remaining -= set(wave)

        return waves

    def safe_execution_order(self, subtasks: list[dict]) -> DependencyGraph:
        """
        End-to-end: build graph, detect cycles, sort, find parallel groups.
        The returned DependencyGraph has all fields populated.
        """
        return self.build_graph(subtasks)

    # ------------------------------------------------------------------
    # Legacy compatibility (used by core/planner.py)
    # ------------------------------------------------------------------

    def reorder(self, steps: list) -> list:
        """
        Reorder PlanStep / str objects using dependency inference.
        Preserved for backward compatibility with planner.py.
        """
        if not steps:
            return steps

        raw = [self._step_to_dict(i, s) for i, s in enumerate(steps)]
        graph = self.safe_execution_order(raw)

        id_to_step = {str(i): s for i, s in enumerate(steps)}
        reordered = [id_to_step[tid] for tid in graph.execution_order if tid in id_to_step]
        if len(reordered) != len(steps):
            return steps
        return reordered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_tasks(self, subtasks: list[dict]) -> list[SubTask]:
        tasks: list[SubTask] = []
        for i, raw in enumerate(subtasks):
            if isinstance(raw, dict):
                task = SubTask(
                    id=str(raw.get("id", i)),
                    description=str(raw.get("description", "")),
                    depends_on=list(raw.get("depends_on", [])),
                    estimated_complexity=float(raw.get("estimated_complexity", 1.0)),
                )
            else:
                task = SubTask(id=str(i), description=str(raw))
            tasks.append(task)
        return tasks

    def _infer_dependencies(self, tasks: list[SubTask]) -> None:
        """
        Mutate tasks in place by adding inferred dependencies according to the
        four detection rules.
        """
        # Rule 1: function/class name cross-references
        defines: dict[str, str] = {}  # name → task_id that defines it
        for task in tasks:
            for name in self._extract_defines(task.description):
                defines[name] = task.id

        for task in tasks:
            for name in self._extract_uses(task.description):
                if name in defines and defines[name] != task.id:
                    dep = defines[name]
                    if dep not in task.depends_on:
                        task.depends_on.append(dep)

        # Rule 2: sequencing phrases ("after X", "following X", "once X is done")
        id_to_task = {t.id: t for t in tasks}
        for task in tasks:
            for ref_id in self._extract_sequencing_refs(task.description, tasks):
                if ref_id != task.id and ref_id not in task.depends_on:
                    task.depends_on.append(ref_id)

        # Rules 3 & 4: file relationships
        file_map: dict[str, list[str]] = {}  # filename → [task_ids in declaration order]
        for task in tasks:
            for fname in self._extract_files(task.description):
                file_map.setdefault(fname, []).append(task.id)

        creates: dict[str, str] = {}  # filename → task_id that creates it
        for task in tasks:
            for fname in self._extract_creates(task.description):
                creates[fname] = task.id

        for task in tasks:
            for fname in self._extract_modifies(task.description):
                if fname in creates and creates[fname] != task.id:
                    dep = creates[fname]
                    if dep not in task.depends_on:
                        task.depends_on.append(dep)  # Rule 3

        # Rule 4: same file → serialize (earlier declaration comes first)
        for fname, tid_list in file_map.items():
            for i in range(1, len(tid_list)):
                earlier = tid_list[i - 1]
                later_task = id_to_task.get(tid_list[i])
                if later_task and earlier not in later_task.depends_on:
                    later_task.depends_on.append(earlier)

    def _extract_defines(self, text: str) -> list[str]:
        """Return function/class names defined in this description."""
        names: list[str] = []
        for m in re.finditer(r"(?:^|\s)(?:def|class)\s+(\w+)", text, re.MULTILINE | re.IGNORECASE):
            names.append(m.group(1))
        # Also match "implement/add/create function/class Foo"
        for m in re.finditer(
            r"(?:implement|add|create|write|define)\s+(?:function|class|method)\s+[`'\"]?(\w+)[`'\"]?",
            text,
            re.IGNORECASE,
        ):
            names.append(m.group(1))
        return list(set(names))

    def _extract_uses(self, text: str) -> list[str]:
        """Return names called/instantiated in this description."""
        names: list[str] = []
        for m in re.finditer(r"\b(\w+)\s*\(", text):
            names.append(m.group(1))
        # Also match "call/use/invoke Foo"
        for m in re.finditer(r"(?:call|use|invoke|instantiate)\s+[`'\"]?(\w+)[`'\"]?", text, re.IGNORECASE):
            names.append(m.group(1))
        return list(set(names))

    def _extract_sequencing_refs(self, text: str, tasks: list[SubTask]) -> list[str]:
        """
        Parse sequencing phrases and return referenced task ids.
        Patterns: "after <X>", "following <X>", "once <X> is done", "depends on <X>"
        where X is either a task id or a word that appears in another task's description.
        """
        refs: list[str] = []
        patterns = [
            r"after\s+(?:task\s+)?[`'\"]?(\w[\w\-]*)[`'\"]?",
            r"following\s+(?:task\s+)?[`'\"]?(\w[\w\-]*)[`'\"]?",
            r"once\s+[`'\"]?(\w[\w\-]*)[`'\"]?\s+is\s+done",
            r"depends\s+on\s+(?:task\s+)?[`'\"]?(\w[\w\-]*)[`'\"]?",
            r"requires\s+(?:task\s+)?[`'\"]?(\w[\w\-]*)[`'\"]?",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                token = m.group(1)
                # Direct task id match
                for task in tasks:
                    if task.id == token:
                        refs.append(task.id)
                        break
                else:
                    # Fuzzy: token appears in another task's description
                    for task in tasks:
                        if token.lower() in task.description.lower():
                            refs.append(task.id)
        return list(set(refs))

    def _extract_files(self, text: str) -> list[str]:
        """Return file paths mentioned in the description."""
        files: list[str] = []
        for m in re.finditer(r"[\w\-/]+\.(?:py|js|ts|json|yaml|yml|md|txt|cfg|toml|sh|html|css)\b", text):
            files.append(m.group(0))
        return list(set(files))

    def _extract_creates(self, text: str) -> list[str]:
        """Return files this task creates."""
        files: list[str] = []
        for m in re.finditer(
            r"(?:create|write|generate|new file)\s+[`'\"]?([\w\-/]+\.(?:py|js|ts|json|yaml|yml|md|txt|cfg|toml|sh|html|css))[`'\"]?",
            text,
            re.IGNORECASE,
        ):
            files.append(m.group(1))
        return list(set(files))

    def _extract_modifies(self, text: str) -> list[str]:
        """Return files this task modifies."""
        files: list[str] = []
        for m in re.finditer(
            r"(?:modify|update|edit|change|add to|append to)\s+[`'\"]?([\w\-/]+\.(?:py|js|ts|json|yaml|yml|md|txt|cfg|toml|sh|html|css))[`'\"]?",
            text,
            re.IGNORECASE,
        ):
            files.append(m.group(1))
        return list(set(files))

    def _adjacency(self, tasks: list[SubTask]) -> dict[str, list[str]]:
        """Return {task_id: [dep_ids]} mapping (what each task depends on)."""
        return {t.id: list(t.depends_on) for t in tasks}

    def _file_map(self, tasks: list[SubTask]) -> dict[str, set[str]]:
        """Return {task_id: set_of_files} mapping."""
        result: dict[str, set[str]] = {}
        for task in tasks:
            result[task.id] = set(self._extract_files(task.description))
        return result

    def _wave_no_file_conflicts(
        self, ready: list[str], file_map: dict[str, set[str]]
    ) -> list[str]:
        """
        From the ready list, greedily select a subset that shares no files.
        Tasks excluded from this wave will be picked in the next wave.
        """
        wave: list[str] = []
        used_files: set[str] = set()
        for tid in ready:
            files = file_map.get(tid, set())
            if not files.intersection(used_files):
                wave.append(tid)
                used_files.update(files)
            # Tasks with no files can always join
            elif not files:
                wave.append(tid)
        return wave if wave else [ready[0]]

    def _break_cycles(self, graph: DependencyGraph) -> None:
        """
        Break cycles by removing the back-edge from the lower-complexity task.
        Adds a warning for each cycle found.
        """
        task_map = {t.id: t for t in graph.tasks}
        for (task_a, task_b) in graph.circular_deps:
            msg = f"Circular dependency detected: {task_a} -> {task_b}. Breaking cycle."
            graph.warnings.append(msg)
            ta = task_map.get(task_a)
            tb = task_map.get(task_b)
            if ta is None or tb is None:
                continue
            # Demote the lower-complexity task (remove its dependency on the other)
            if ta.estimated_complexity <= tb.estimated_complexity:
                if task_b in ta.depends_on:
                    ta.depends_on.remove(task_b)
            else:
                if task_a in tb.depends_on:
                    tb.depends_on.remove(task_a)

    def _step_to_dict(self, index: int, step: Any) -> dict:
        """Convert a PlanStep or string to a dict suitable for build_graph."""
        if hasattr(step, "description"):
            desc = step.description
        elif hasattr(step, "replace"):
            desc = step.replace or ""
        else:
            desc = str(step)
        # Include file reference in description if available for Rule 4
        if hasattr(step, "file") and step.file:
            desc = f"{desc} {step.file}"
        return {"id": str(index), "description": desc}

    # ------------------------------------------------------------------
    # Legacy helpers (kept for any external callers)
    # ------------------------------------------------------------------

    def extract_defines(self, text: str) -> list[str]:
        return self._extract_defines(text)

    def extract_uses(self, text: str) -> list[str]:
        return self._extract_uses(text)

    def build_nodes(self, steps: list) -> list[DepNode]:
        nodes = []
        for i, step in enumerate(steps):
            text = getattr(step, "replace", "") or getattr(step, "description", "") or str(step)
            node = DepNode(step_index=i)
            node.defines = self._extract_defines(text)
            node.uses = self._extract_uses(text)
            nodes.append(node)
        return nodes

    def detect_cycles(self, deps: dict) -> list[str]:
        """Legacy index-based cycle detection for planner.py fallback path."""
        visited: set = set()
        rec_stack: set = set()
        cycles: list[str] = []

        def dfs(node: Any) -> None:
            visited.add(node)
            rec_stack.add(node)
            for neighbour in deps.get(node, set()):
                if neighbour not in visited:
                    dfs(neighbour)
                elif neighbour in rec_stack:
                    cycles.append(f"cycle: {neighbour} -> {node}")
            rec_stack.discard(node)

        for node in deps:
            if node not in visited:
                dfs(node)
        return cycles
