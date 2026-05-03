from __future__ import annotations

"""
core/executor.py - CodeAct execution pattern.
Agent writes a Python edit script → run it → observe outcome → self-correct.
Max 5 iterations before surfacing to user.
"""
from dataclasses import dataclass, field
from pathlib import Path
import subprocess as _sp

from core.tools import run_python, RunResult
from core.llm import LLMClient
from core.tool_registry import ToolRegistry
from core.snapshot import take_snapshot, restore_snapshot
from core.failure_analyzer import classify_failure, FailureAnalyzer, FailureAnalysis
from core.experience import FailureClass
from core.atomic_executor import AtomicExecutor

MAX_ITERATIONS = 5

# Config/orchestration file extensions that should use whole-file write
WHOLE_FILE_EXTENSIONS = {
    ".sh",
    ".bash",
    ".zsh",
    ".env",
    ".ini",
    ".cfg",
    ".toml",
    ".conf",
    ".config",
    ".properties",
    "Dockerfile",
}


def _has_config_files(file_contexts: dict) -> list[str]:
    """Detect config/orchestration files that should use whole-file write."""
    config_files = []
    for fp in file_contexts:
        p = Path(fp)
        if p.suffix in WHOLE_FILE_EXTENSIONS or p.name in WHOLE_FILE_EXTENSIONS:
            config_files.append(fp)
    return config_files


@dataclass
class EditResult:
    success: bool
    iterations: int
    final_script: str
    output: str
    error: str = ""
    files_changed: list[str] = field(default_factory=list)
    failure_class: str = FailureClass.NONE  # populated on failure for ExperienceStore


def execute_edit(
    task: str,
    file_contexts: dict[str, str],  # {filepath: content}
    repo_path: str | Path,
    llm: LLMClient,
    model: str,
    beliefs_context: str = "",
    tool_registry: ToolRegistry = None,
) -> EditResult:
    """
    CodeAct loop: generate edit script → run → if fails, self-correct.
    The agent writes Python code that edits the target files.
    Python is the action space (not JSON tool calls).
    """
    repo = Path(repo_path)
    history = []  # (script, result) pairs for self-correction context
    tools_ctx = tool_registry.to_prompt_context() if tool_registry else ""
    snapshot = take_snapshot(list(file_contexts.keys()))
    _analyzer = FailureAnalyzer()
    _last_analysis: FailureAnalysis | None = None
    _atomic = AtomicExecutor()

    # Build system prompt with tool discovery
    system_prompt = _build_system_prompt(tool_registry)

    for iteration in range(1, MAX_ITERATIONS + 1):
        prompt = _build_prompt(task, file_contexts, history, beliefs_context, repo, tools_ctx)
        script = llm.ask(model, prompt, system=system_prompt)

        # Strip markdown fences if model wraps in ```python
        script = _clean_script(script)

        pre_run_files = set(file_contexts.keys())
        # Wrap script execution atomically: snapshot all touched files before
        # running, restore on any exception so no file is left partially edited.
        _file_list = list(file_contexts.keys())
        _run_holder: list[RunResult] = []

        def _atomic_edit_fn() -> None:
            r = run_python(script, cwd=repo, timeout=30)
            _run_holder.append(r)
            # Propagate run failures as exceptions so the atomic wrapper rolls back
            if not r.success:
                raise RuntimeError(r.stderr or r.output or "run_python returned failure")

        _guard_paths = _file_list if len(_file_list) <= 3 else _file_list[:3]
        _atomic_result = _atomic.execute_atomic(_atomic_edit_fn, _guard_paths)

        if _atomic_result.restored:
            # AtomicExecutor rolled back — the run itself raised an exception.
            # Build a RunResult from whatever partial output was captured.
            _rollback_err = (
                f"[atomic_executor] Rollback triggered: {_atomic_result.error}\n"
                "All snapshotted files restored to pre-edit state."
            )
            if _run_holder:
                result = _run_holder[0]
            else:
                result = RunResult(returncode=1, stdout="", stderr=_rollback_err)
            _last_analysis = _analyzer.analyze(_rollback_err, script)
            if _last_analysis is not None:
                _last_analysis.failure_class = FailureClass.EDIT_SYNTAX_ERROR
            history.append((script, result))
            if iteration == MAX_ITERATIONS:
                break
            continue

        # Successful atomic execution — unwrap the captured RunResult.
        result = (
            _run_holder[0]
            if _run_holder
            else RunResult(
                returncode=1, stdout="", stderr="[atomic_executor] no run result captured"
            )
        )
        history.append((script, result))

        if result.success:
            # Enforce SUCCESS marker — silent failures must be caught
            if "SUCCESS" not in result.stdout:
                _err = "[executor] No SUCCESS marker — edit may not have applied"
                _last_analysis = _analyzer.analyze(_err, script)
                history.append(
                    (
                        script,
                        RunResult(
                            returncode=1,
                            stdout=result.stdout,
                            stderr=_err,
                        ),
                    )
                )
                continue

            changed = _detect_changed_files(script, repo)

            check_errors = _check_changed_files(changed, repo)
            if check_errors:
                restore_snapshot(snapshot)
                for _f in changed:
                    if _f not in pre_run_files and Path(_f).exists():
                        Path(_f).unlink()
                _err = "Maker-checker validation failed:\n" + "\n".join(check_errors)
                _last_analysis = _analyzer.analyze(_err, script)
                if iteration == MAX_ITERATIONS:
                    return EditResult(
                        success=False,
                        iterations=iteration,
                        final_script=script,
                        output=result.output,
                        error="Maker-checker failed on final iteration:\n"
                        + "\n".join(check_errors),
                        failure_class=_last_analysis.failure_class,
                    )
                history[-1] = (
                    script,
                    RunResult(
                        returncode=1,
                        stdout=result.stdout,
                        stderr=_err,
                    ),
                )
                continue

            # Run tests if available
            if tool_registry and iteration < MAX_ITERATIONS:
                test_tool = tool_registry.get("test")
                if test_tool:
                    test_result = test_tool.run(cwd=repo)
                    if not test_result.success:
                        restore_snapshot(snapshot)
                        _err = f"Tests failed:\n{test_result.output}"
                        _last_analysis = _analyzer.analyze(_err, script)
                        history[-1] = (
                            script,
                            RunResult(
                                success=False,
                                output=test_result.output,
                                stderr=_err,
                                returncode=1,
                            ),
                        )
                        continue

            return EditResult(
                success=True,
                iterations=iteration,
                final_script=script,
                output=result.output,
                files_changed=changed,
            )

        # Script execution itself failed — classify and record for next iteration
        _err = result.stderr if hasattr(result, "stderr") else result.output
        _last_analysis = _analyzer.analyze(_err, script)

        if iteration == MAX_ITERATIONS:
            break

        # Self-correction: loop continues with error context and recovery prompt injected
        # via _build_prompt which reads history[-1] and classify_failure

    restore_snapshot(snapshot)
    _fc = _last_analysis.failure_class if _last_analysis else FailureClass.NONE
    return EditResult(
        success=False,
        iterations=MAX_ITERATIONS,
        final_script=history[-1][0] if history else "",
        output=history[-1][1].output if history else "",
        error=f"Failed after {MAX_ITERATIONS} iterations. Last error:\n{history[-1][1].stderr if history else ''}",
        failure_class=_fc,
    )


def _build_prompt(
    task: str,
    file_contexts: dict,
    history: list,
    beliefs: str,
    repo: Path,
    tools_ctx: str = "",
) -> str:
    lines = [f"Task: {task}", ""]

    if beliefs:
        lines += ["Relevant beliefs from memory:", beliefs, ""]

    if tools_ctx:
        lines += ["Available tools:", tools_ctx, ""]

    lines += [
        f"Repo root: {repo}",
        "",
        "Relevant file contents:",
    ]
    for filepath, content in list(file_contexts.items())[:4]:  # cap context
        lines += [f"\n--- {filepath} ---", content[:1500], ""]

    lines += [
        "",
        "Write a Python script that makes the required edit.",
        "CRITICAL RULES:",
        "- For NEW files: use Path(filepath).write_text(content).",
        "- For EXISTING files: read with Path(filepath).read_text(), apply str.replace(), write back.",
        "- MODIFY existing functions/classes in place. NEVER add duplicate definitions.",
        "- Do not append new functions if one with the same name already exists.",
        "- Use pathlib.Path for file operations. Do not import non-stdlib modules.",
        "- Print 'SUCCESS' at the end if the edit was applied correctly.",
        "- IDEMPOTENCY: For each str.replace(old, new), check if old is present. If old is NOT found but new IS already present, that step is done — skip silently and continue to the NEXT replacement. If old is NOT found and new is NOT present either, raise AssertionError so the executor can retry. Only print SUCCESS after ALL replacements are processed (either applied or skipped). Never exit early.",
        "- NEVER apply the same replacement twice in one script.",
        "- For config/orchestration files (.env, .ini, .cfg, .sh, .bash, .zsh, Dockerfile, .toml, .conf, .config, .properties): ALWAYS use Path(filepath).read_text() then Path(filepath).write_text(new_content) to write the whole file. NEVER use str.replace() on these file types. Make changes mentally while reading, then write the complete corrected content in one write_text() call.",
        "- NEVER define a function or class if one with that exact name already exists in the file.",
        "Only output the Python script - no explanation.",
    ]

    # Inject harness meta-beliefs as executor warnings
    try:
        from core.belief import BeliefStore

        self_beliefs = BeliefStore("self")
        executor_warnings = [
            b for b in self_beliefs.all() if b.is_actionable and "executor" in b.statement.lower()
        ]
        if executor_warnings:
            lines += ["Known weaknesses (from self-analysis):"]
            for wb in executor_warnings[:3]:
                lines.append(f"  WARNING: {wb.statement}")
            lines.append("")
    except Exception:
        pass

    # Detect config/orchestration files and inject CRITICAL warning
    config_files = _has_config_files(file_contexts)
    if config_files:
        lines += [
            "CRITICAL: The following files are config/orchestration types that MUST be edited with whole-file write:",
            *[f"  - {f}" for f in config_files],
            "Do NOT use str.replace() on any of these files. Read the whole file, apply changes in memory, write back.",
            "",
        ]

    # Inject skill file context from previous fixes
    try:
        from core.failure_analyzer import SkillStore

        skill_store = SkillStore()
        matching_skill = skill_store.find_matching_skill(task)
        if matching_skill:
            lines += [
                "",
                "REFERENCE: A similar case was previously solved with this skill:",
                f"--- {matching_skill['name']} ---",
                matching_skill["content"][:500],  # First 500 chars of skill
                "",
            ]
    except Exception:
        pass  # Skill lookup is optional

    if history:
        last_script, last_result = history[-1]
        last_err = last_result.stderr if hasattr(last_result, "stderr") else ""
        # Structured recovery prompt from FailureAnalyzer (higher signal than raw error dump)
        _fa = FailureAnalyzer()
        _analysis = _fa.analyze(last_err, last_script)
        recovery_block = _fa.recovery_prompt(_analysis)
        lines += [
            recovery_block,
            "Previous attempt failed. Script:",
            "```python",
            last_script,
            "```",
            f"Error output:\n{last_result.tail(20)}",
            f"Legacy hint: {classify_failure(last_script, last_err, task).recovery_hint}",
            "",
            "Fix the script based on the recovery instructions above.",
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


def _check_changed_files(files: list[str], repo: Path) -> list[str]:
    """Maker-checker: validate changed files for obvious errors."""
    import json as _json

    errors = []
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            errors.append(f"{filepath}: file missing after edit")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"{filepath}: unreadable after edit: {e}")
            continue
        if not content.strip():
            errors.append(f"{filepath}: file is empty after edit")
            continue
        lines_set = set(content.splitlines())
        if any(l.startswith("<<<<<<<") or l.startswith(">>>>>>>") for l in lines_set):
            errors.append(f"{filepath}: merge conflict markers present")
            continue
        if filepath.endswith(".py"):
            r = _sp.run(
                ["python3", "-m", "py_compile", filepath],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode != 0:
                errors.append(f"{filepath}: syntax error — {r.stderr.strip()[:300]}")
        elif filepath.endswith(".json"):
            try:
                _json.loads(content)
            except Exception as e:
                errors.append(f"{filepath}: invalid JSON — {e}")
        elif filepath.endswith((".yaml", ".yml")):
            if "\t" in content:
                errors.append(f"{filepath}: YAML contains tabs")
        elif filepath.endswith((".sh", ".bash", ".zsh")):
            r = _sp.run(["bash", "-n", filepath], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                errors.append(f"{filepath}: shell syntax error — {r.stderr.strip()[:300]}")
    return errors


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


def _build_system_prompt(tool_registry: ToolRegistry = None) -> str:
    """
    Build system prompt with optional tool discovery.
    If tool_registry is provided, inject discovered tools into the prompt.
    """
    base_system = """You are a precise code editing agent.
You write Python scripts that make targeted edits to files.
Your scripts must be correct Python that runs without errors.
Never use external libraries beyond Python stdlib.
Always print SUCCESS at the end if the edit succeeds."""

    if not tool_registry:
        return base_system

    # Inject discovered tools into system prompt
    try:
        tools = list(tool_registry._tools.values())
        if tools:
            tools_list = "\n".join(f"  - {tool.name}: {tool.description}" for tool in tools)
            tool_section = f"\nAvailable tools in this repo:\n{tools_list}"
            return base_system + tool_section
    except Exception:
        pass

    return base_system


_SYSTEM = """You are a precise code editing agent.
You write Python scripts that make targeted edits to files.
Your scripts must be correct Python that runs without errors.
Never use external libraries beyond Python stdlib.
Always print SUCCESS at the end if the edit succeeds."""
