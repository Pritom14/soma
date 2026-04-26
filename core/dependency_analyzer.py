from __future__ import annotations
import re
from dataclasses import dataclass, field

@dataclass
class DepNode:
    step_index: int
    defines: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)

class DependencyAnalyzer:

    def extract_defines(self, text: str) -> list[str]:
        # Extract function/class names defined in this step
        names = []
        for m in re.finditer(r'^\s*(?:def|class)\s+(\w+)', text, re.MULTILINE):
            names.append(m.group(1))
        return names

    def extract_uses(self, text: str) -> list[str]:
        # Extract names called/instantiated in this step
        names = []
        for m in re.finditer(r'\b(\w+)\s*\(', text):
            names.append(m.group(1))
        return list(set(names))

    def build_nodes(self, steps: list) -> list[DepNode]:
        nodes = []
        for i, step in enumerate(steps):
            text = getattr(step, 'replace', '') or getattr(step, 'description', '') or str(step)
            node = DepNode(step_index=i)
            node.defines = self.extract_defines(text)
            node.uses = self.extract_uses(text)
            nodes.append(node)
        return nodes

    def reorder(self, steps: list) -> list:
        # Topological sort: definitions before usages. Returns reordered steps.
        nodes = self.build_nodes(steps)
        all_defines = {}
        for node in nodes:
            for name in node.defines:
                all_defines[name] = node.step_index

        # Build adjacency: step A must come before step B if B uses something A defines
        deps = {i: set() for i in range(len(steps))}
        for node in nodes:
            for used in node.uses:
                if used in all_defines:
                    definer = all_defines[used]
                    if definer != node.step_index:
                        deps[node.step_index].add(definer)

        cycles = self.detect_cycles(deps)
        if cycles:
            return steps  # Cannot reorder safely, return as-is

        # Kahn topological sort
        in_degree = {i: len(deps[i]) for i in range(len(steps))}
        queue = [i for i in range(len(steps)) if in_degree[i] == 0]
        order = []
        while queue:
            queue.sort()
            node_i = queue.pop(0)
            order.append(node_i)
            for j in range(len(steps)):
                if node_i in deps[j]:
                    deps[j].discard(node_i)
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        if len(order) == len(steps):
            return [steps[i] for i in order]
        return steps  # Cycle detected fallback

    def detect_cycles(self, deps: dict) -> list[str]:
        # DFS cycle detection. Returns list of cycle descriptions.
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in deps.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    cycles.append(f'cycle: {neighbor} -> {node}')
            rec_stack.discard(node)

        for node in deps:
            if node not in visited:
                dfs(node)
        return cycles
