from __future__ import annotations
"""
core/executor.py - CodeAct execution pattern.
Agent writes a Python edit script → run it → observe outcome → self-correct.
Max 3 iterations before surfacing to user.
"""
from dataclasses import dataclass, field
from pathlib import Path

from core.tools import run_python, read_file, RunResult
from core.llm import LLMClient

MAX_ITERATIONS = 3


@dataclass
class EditResult:
    success: bool
    iterations: int
    final_script: str
    output: str
    error: str = ""
    files_changed: list[str] = field(default_factory=list)


def execute_edit(
    task: str,
    file_contexts: dict[str, str],  # {filepath: content}
    repo_path: str | Path,
    llm: LLMClient,
    model: str,
    beliefs_context: str = "",
) -> EditResult:
    """
    CodeAct loop: generate edit script → run → if fails, self-correct.
    The agent writes Python code that edits the target files.
    Python is the action space (not JSON tool calls).
    """
    repo = Path(repo_path)
    history = []  # (script, result) pairs for self-correction context

    for iteration in range(1, MAX_ITERATIONS + 1):
        prompt = _build_prompt(task, file_contexts, history, beliefs_context, repo)
        script = llm.ask(model, prompt, system=_SYSTEM)

        # Strip markdown fences if model wraps in ```python
        script = _clean_script(script)

        result = run_python(script, cwd=repo, timeout=30)
        history.append((script, result))

        if result.success:
            changed = _detect_changed_files(script, repo)
            return EditResult(
                success=True,
                iterations=iteration,
                final_script=script,
                output=result.output,
                files_changed=changed,
            )

        if iteration == MAX_ITERATIONS:
            break

        # Self-correction: loop continues with error context

    return EditResult(
        success=False,
        iterations=MAX_ITERATIONS,
        final_script=history[-1][0] if history else "",
        output=history[-1][1].output if history else "",
        error=f"Failed after {MAX_ITERATIONS} iterations. Last error:\n{history[-1][1].stderr if history else ''}",
    )


def _build_prompt(task: str, file_contexts: dict, history: list,
                  beliefs: str, repo: Path) -> str:
    lines = [f"Task: {task}", ""]

    if beliefs:
        lines += ["Relevant beliefs from memory:", beliefs, ""]

    lines += [
        f"Repo root: {repo}",
        "",
        "Relevant file contents:",
    ]
    for filepath, content in list(file_contexts.items())[:4]:  # cap context
        lines += [f"\n--- {filepath} ---", content[:1500], ""]

    if history:
        last_script, last_result = history[-1]
        lines += [
            "Previous attempt failed. Script:",
            "```python",
            last_script,
            "```",
            f"Error output:\n{last_result.tail(20)}",
            "",
            "Fix the script based on the error above.",
        ]

    lines += [
        "",
        "Write a Python script that makes the required edit.",
        "CRITICAL RULES:",
        "- MODIFY existing functions/classes in place. NEVER add duplicate definitions.",
        "- Use str.replace() to swap old code for new code within the file content.",
        "- Read the full file, apply ONE targeted replacement, write back.",
        "- Do not append new functions if one with the same name already exists.",
        "- Use pathlib.Path for file operations. Do not import non-stdlib modules.",
        "- Print 'SUCCESS' at the end if the edit was applied correctly.",
        "Only output the Python script - no explanation.",
    ]

    return "\n".join(lines)


def _detect_changed_files(script: str, repo: Path) -> list[str]:
    """Heuristic: find file paths written to in the script."""
    import re
    paths = re.findall(r'["\']([^"\']+\.[a-z]{2,4})["\']', script)
    changed = []
    for p in paths:
        full = repo / p if not Path(p).is_absolute() else Path(p)
        if full.exists():
            changed.append(str(full))
    return list(set(changed))


def _clean_script(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1]
        elif len(parts) == 2:
            raw = parts[1]
        if raw.startswith("python"):
            raw = raw[6:]
    return raw.strip()


_SYSTEM = """You are a precise code editing agent.
You write Python scripts that make targeted edits to files.
Your scripts must be correct Python that runs without errors.
Never use external libraries beyond Python stdlib.
Always print SUCCESS at the end if the edit succeeds."""
