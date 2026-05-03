from __future__ import annotations

"""
core/planner.py - Pre-execution planning phase.
Generates a structured step-by-step plan before CodeAct execution.
Separates planning from execution to reduce hallucination and iteration errors.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.llm import LLMClient
from core.dependency_analyzer import DependencyAnalyzer


@dataclass
class PlanStep:
    step_id: int
    description: str
    file: str
    find: str
    replace: str
    verify_contains: str = ""
    verify_not_contains: str = ""


@dataclass
class ExecutionPlan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    notes: str = ""
    valid: bool = True
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class StructuredPlan:
    """Richer plan produced by the recursive decomposer."""

    steps: list[str]
    depth_levels: dict[str, int]  # step text -> nesting depth
    dependencies: list[tuple[int, int]]  # (dependent_idx, dependency_idx)
    estimated_complexity: float  # mirrors ComplexityScore.score


_PLANNER_SYSTEM = (
    "You are a precise code change planner. "
    "Given a task and file contents, output a JSON execution plan. "
    "Be exact — find strings will be matched literally against file contents. "
    "Output only valid JSON, no explanation."
)

_PLANNER_PROMPT = """Task: {task}

File contents:
{file_contents}

Generate a JSON execution plan with this structure:
{{
  "goal": "one-line summary of what this achieves",
  "notes": "ordering constraints or caveats",
  "steps": [
    {{
      "step_id": 1,
      "description": "what this step does",
      "file": "exact/file/path",
      "find": "exact string to find in the file",
      "replace": "exact replacement string",
      "verify_contains": "string that must exist in file after edit",
      "verify_not_contains": "string that must NOT exist after edit"
    }}
  ]
}}

Rules:
- Each step targets exactly one file
- find strings MUST be copied character-for-character from the file content shown above — do not paraphrase, reformat, or reconstruct from memory
- Select the shortest unique substring that identifies the exact location of the change
- If find is not found but replace is already present, that step is already done — skip it
- Order steps so no step depends on a later step — callee must be defined before caller
- One logical change per step
- If a step would add a call to a function that does not yet exist in the file, add the function definition as an earlier step first"""


def generate_plan(
    task: str,
    file_contexts: dict[str, str],
    llm: LLMClient,
    model: str,
) -> ExecutionPlan:
    """Ask the LLM to plan all steps before any execution begins."""
    file_summary = ""
    for path, content in list(file_contexts.items())[:6]:
        file_summary += f"\n--- {path} ---\n{content[:2000]}\n"

    prompt = _PLANNER_PROMPT.format(task=task, file_contents=file_summary)
    try:
        raw = llm.ask(model, prompt, system=_PLANNER_SYSTEM)
    except Exception:
        return ExecutionPlan(
            goal=task,
            valid=False,
            validation_errors=["Planner timed out — running without plan"],
        )

    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 2 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        steps = [PlanStep(**s) for s in data.get("steps", [])]
        return ExecutionPlan(
            goal=data.get("goal", task),
            steps=steps,
            notes=data.get("notes", ""),
        )
    except Exception as e:
        return ExecutionPlan(goal=task, valid=False, validation_errors=[f"Plan parse failed: {e}"])


def validate_plan(plan: ExecutionPlan, repo: Path) -> ExecutionPlan:
    """Check referenced files exist and find-strings are present."""
    if not plan.valid:
        return plan

    errors = []
    for step in plan.steps:
        fp = Path(step.file) if Path(step.file).is_absolute() else repo / step.file
        if not fp.exists():
            errors.append(f"Step {step.step_id}: file not found: {step.file}")
            continue
        try:
            content = fp.read_text(encoding="utf-8")
            if step.find and step.find not in content:
                if step.replace and step.replace in content:
                    step.description = f"[SKIP - already applied] {step.description}"
                else:
                    errors.append(f"Step {step.step_id}: find string not in {step.file}")
        except Exception as e:
            errors.append(f"Step {step.step_id}: could not read {step.file}: {e}")

    plan.validation_errors = errors
    plan.valid = len(errors) == 0
    return plan


def plan_to_context(plan: ExecutionPlan) -> str:
    """Convert plan to string for injection into executor prompt."""
    if not plan.steps:
        return ""
    lines = [f"Execution plan — {plan.goal} ({len(plan.steps)} steps):"]
    for step in plan.steps:
        lines.append(f"  Step {step.step_id}: {step.description} [{step.file}]")
        if step.find:
            lines.append(f"    find:    {repr(step.find[:100])}")
            lines.append(f"    replace: {repr(step.replace[:100])}")
    if plan.notes:
        lines.append(f"Notes: {plan.notes}")
    return "\n".join(lines)


# Split any complex plan steps into smaller atomic operations.
def decompose_complex_steps(plan: ExecutionPlan, llm: LLMClient, model: str) -> ExecutionPlan:
    if not plan.steps:
        return plan

    new_steps = []
    for step in plan.steps:
        if (
            len(step.replace) > 200
            or step.replace.count("def ") > 1
            or step.replace.count("class ") > 0
        ):
            try:
                sys_prompt = "You are a code change planner. Split this step into 2-4 atomic sub-steps. Output only a JSON array of objects with keys: step_id, description, file, find, replace. No explanation."
                user_prompt = (
                    "Step: "
                    + step.description
                    + " | File: "
                    + step.file
                    + " | Find: "
                    + step.find[:200]
                    + " | Replace: "
                    + step.replace[:200]
                )
                response = llm.ask(model, user_prompt, system=sys_prompt)
                sub_steps_raw = json.loads(response)
                parsed = [PlanStep(**s) for s in sub_steps_raw if isinstance(s, dict)]
                new_steps.extend(parsed if parsed else [step])
            except Exception:
                new_steps.append(step)
        else:
            new_steps.append(step)

    # Re-number all step_ids sequentially starting from 1
    for idx, step in enumerate(new_steps, start=1):
        step.step_id = idx

    plan.steps = DependencyAnalyzer().reorder(new_steps)
    return plan


# ---------------------------------------------------------------------------
# RecursivePlanner — high-level decomposer with hazard detection
# ---------------------------------------------------------------------------

# Verb-only patterns — used to find the start of a "define" or "use" region.
# _scan_window collects all identifiers in the following 80 chars.
_DEFINES_VERB_RE = re.compile(
    r"\b(?:add|create|define|implement|write|introduce)\b",
    re.IGNORECASE,
)

_USES_VERB_RE = re.compile(
    r"\b(?:use|call|invoke|import|apply|reference|update|modify|test|verify)\b",
    re.IGNORECASE,
)

# Legacy single-capture regexes kept for any direct callers (not used internally)
_DEFINES_RE = _DEFINES_VERB_RE
_USES_RE = _USES_VERB_RE

# Common English stop-words to skip when extracting names
_STOP_WORDS = {
    "the",
    "and",
    "with",
    "for",
    "from",
    "into",
    "that",
    "this",
    "its",
    "new",
    "all",
    "any",
    "each",
    "also",
    "then",
    "when",
    "file",
    "step",
    "code",
    "data",
    "value",
    "class",
    "function",
}


class RecursivePlanner:
    """
    Heuristic planner that:
      - decomposes complex step descriptions into atomic sub-steps
      - detects out-of-order dependencies between steps
      - reorders steps into a safe execution sequence
    """

    # Maximum recursion depth for decomposition
    MAX_DEPTH = 3

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def decompose_step(self, step: str, _depth: int = 0) -> list[str]:
        """
        Break a complex step description into atomic sub-steps.

        A step is considered complex if it contains multiple conjunctions
        ("and", "then", "also", ";") or multiple action verbs.  Recursion
        stops when the step is already atomic or MAX_DEPTH is reached.
        """
        if _depth >= self.MAX_DEPTH:
            return [step.strip()]

        parts = self._split_on_conjunctions(step)
        if len(parts) <= 1:
            # Try splitting on semicolons / numbered lists
            parts = self._split_on_punctuation(step)

        if len(parts) <= 1:
            return [step.strip()]

        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            sub = self.decompose_step(part, _depth + 1)
            result.extend(sub)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for s in result:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    def detect_sequencing_hazards(self, steps: list[str]) -> list[tuple[int, int]]:
        """
        Return pairs (i, j) where step[i] depends on something defined by step[j]
        but j > i (i.e. the dependency comes after the step that needs it).

        Each tuple is (dependent_index, dependency_index).
        """
        defines: dict[str, int] = {}  # name -> first step index that defines it
        for idx, step in enumerate(steps):
            for name in self._extract_defined_names(step):
                if name not in defines:
                    defines[name] = idx

        hazards: list[tuple[int, int]] = []
        for idx, step in enumerate(steps):
            for name in self._extract_used_names(step):
                if name in defines and defines[name] > idx:
                    pair = (idx, defines[name])
                    if pair not in hazards:
                        hazards.append(pair)

        return hazards

    def reorder_for_dependencies(self, steps: list[str]) -> list[str]:
        """
        Return steps in a safe execution order (definitions before usages).

        Uses a stable topological sort.  If a cycle is detected the original
        order is returned unchanged.
        """
        n = len(steps)
        if n == 0:
            return steps

        defines: dict[str, int] = {}
        for idx, step in enumerate(steps):
            for name in self._extract_defined_names(step):
                if name not in defines:
                    defines[name] = idx

        # Build dependency graph: deps[i] = set of indices that must come before i
        deps: dict[int, set[int]] = {i: set() for i in range(n)}
        for idx, step in enumerate(steps):
            for name in self._extract_used_names(step):
                if name in defines and defines[name] != idx:
                    deps[idx].add(defines[name])

        # Kahn's algorithm — stable (preserves relative order for tied nodes)
        in_degree = [len(deps[i]) for i in range(n)]
        # Use a list as a stable queue (sorted to preserve original order)
        ready = sorted(i for i in range(n) if in_degree[i] == 0)
        order: list[int] = []

        while ready:
            node = ready.pop(0)
            order.append(node)
            for j in range(n):
                if node in deps[j]:
                    deps[j].discard(node)
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        ready.append(j)
                        ready.sort()  # keep stable

        if len(order) != n:
            # Cycle detected — fall back to original order
            return steps

        return [steps[i] for i in order]

    def build_structured_plan(
        self,
        raw_steps: list[str],
        estimated_complexity: float = 0.0,
    ) -> StructuredPlan:
        """
        Decompose each step, detect hazards, reorder, and return a StructuredPlan.
        """
        # 1. Decompose each raw step
        decomposed: list[str] = []
        depth_levels: dict[str, int] = {}
        for step in raw_steps:
            subs = self.decompose_step(step)
            for sub in subs:
                decomposed.append(sub)
                # Depth is 0 for top-level; sub-steps inherit depth from recursion
                depth_levels[sub] = 0 if sub == step else 1

        # 2. Detect hazards before reordering
        hazards = self.detect_sequencing_hazards(decomposed)

        # 3. Reorder
        ordered = self.reorder_for_dependencies(decomposed)

        # Recompute depth_levels index for reordered list
        ordered_depths = {s: depth_levels.get(s, 0) for s in ordered}

        # Convert hazard step-text pairs to index pairs in the reordered list
        step_to_idx = {s: i for i, s in enumerate(ordered)}
        dep_pairs: list[tuple[int, int]] = []
        for dependent_txt, dependency_txt in [
            (decomposed[h[0]], decomposed[h[1]]) for h in hazards
        ]:
            if dependent_txt in step_to_idx and dependency_txt in step_to_idx:
                dep_pairs.append((step_to_idx[dependent_txt], step_to_idx[dependency_txt]))

        return StructuredPlan(
            steps=ordered,
            depth_levels=ordered_depths,
            dependencies=dep_pairs,
            estimated_complexity=estimated_complexity,
        )

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _split_on_conjunctions(self, text: str) -> list[str]:
        """Split on ' and ', ' then ', ' also ', ' after that '."""
        pattern = re.compile(
            r"\s+(?:and then|and also|then|also|after that|afterwards)\s+",
            re.IGNORECASE,
        )
        parts = pattern.split(text)
        return parts if len(parts) > 1 else [text]

    def _split_on_punctuation(self, text: str) -> list[str]:
        """Split on semicolons or numbered list markers like '1. ... 2. ...'."""
        # Try numbered list first
        numbered = re.split(r"\s+\d+\.\s+", text)
        if len(numbered) > 1:
            return [p for p in numbered if p.strip()]

        # Fall back to semicolons
        by_semi = [p for p in text.split(";") if p.strip()]
        return by_semi if len(by_semi) > 1 else [text]

    def _extract_defined_names(self, step: str) -> list[str]:
        return self._scan_window(step, _DEFINES_VERB_RE)

    def _extract_used_names(self, step: str) -> list[str]:
        return self._scan_window(step, _USES_VERB_RE)

    def _scan_window(self, step: str, verb_re: re.Pattern) -> list[str]:
        """
        Find all action verbs matched by verb_re.  For each match, collect
        every non-stop-word identifier (3+ chars) in the following 80 chars.
        """
        names: list[str] = []
        for m in verb_re.finditer(step):
            window = step[m.end() : m.end() + 80]
            for word in re.findall(r"\b(\w{3,})\b", window):
                w = word.lower()
                if w not in _STOP_WORDS:
                    names.append(w)
        return list(dict.fromkeys(names))  # deduplicate, preserve order
