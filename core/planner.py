from __future__ import annotations

"""
core/planner.py - Pre-execution planning phase.
Generates a structured step-by-step plan before CodeAct execution.
Separates planning from execution to reduce hallucination and iteration errors.
"""
import json
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
