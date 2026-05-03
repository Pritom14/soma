# core/failure_analyzer.py - classify CodeAct iteration failures for targeted recovery
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.experience import FailureClass


# ---------------------------------------------------------------------------
# New structured interface: FailureAnalysis + FailureAnalyzer
# ---------------------------------------------------------------------------


@dataclass
class FailureAnalysis:
    """Structured result of classifying a single failed CodeAct iteration."""

    failure_class: str  # one of FailureClass constants
    confidence: float  # 0.0–1.0, how confident we are in the classification
    root_cause: str  # 1-sentence human-readable diagnosis
    recovery_instruction: str  # injected into the next CodeAct prompt iteration
    context: dict = field(default_factory=dict)  # structured data for logging


# Recovery instructions keyed by FailureClass constant
_RECOVERY_INSTRUCTIONS: dict[str, str] = {
    FailureClass.LOCALIZATION_MISS: (
        "The find_string was not found. Read the file first with the Read tool, "
        "copy the exact text including whitespace, then retry the replacement."
    ),
    FailureClass.EDIT_SYNTAX_ERROR: (
        "Your edit script has a syntax error. Check indentation and quote matching. "
        "Use triple-quoted strings for multiline content."
    ),
    FailureClass.VERIFY_BUILD_FAIL: (
        "The build or typecheck failed after your edit. Review the TypeScript/build "
        "errors carefully and fix the type annotations or import paths accordingly."
    ),
    FailureClass.VERIFY_TEST_FAIL: (
        "Tests failed after your edit. Read the failing test output, identify which "
        "assertion broke, and adjust the implementation to satisfy the test."
    ),
    FailureClass.CI_FAIL: (
        "CI checks failed. Review the workflow log for the specific step that failed "
        "and address the lint, type, or test error it reports."
    ),
    FailureClass.LLM_HALLUCINATION: (
        "The method/attribute you referenced does not exist. Read the file to confirm "
        "the actual method names and signatures before editing."
    ),
    FailureClass.PUSH_FAIL: (
        "The git push was rejected. Pull the latest changes, resolve any conflicts, "
        "then retry. Ensure you have write permissions to the branch."
    ),
    FailureClass.NONE: (
        "No specific failure pattern was identified. Review the full error output "
        "and try a more targeted single-operation approach."
    ),
}

# Ordered classification rules: (failure_class, confidence, patterns_any_of, root_cause_template)
# ORDER MATTERS — first match wins. More specific / less ambiguous rules come first.
_RULES: list[tuple[str, float, list[str], str]] = [
    # 1. Syntax errors — very distinctive tokens, no cross-class ambiguity
    (
        FailureClass.EDIT_SYNTAX_ERROR,
        0.90,
        [
            "syntaxerror",
            "indentationerror",
            "unexpected token",
            "unexpected indent",
            "unexpected eof",
            "invalid syntax",
        ],
        "The generated edit script contains a Python syntax or indentation error.",
    ),
    # 2. Build/typecheck — TypeScript-specific tokens before generic "failed"
    (
        FailureClass.VERIFY_BUILD_FAIL,
        0.88,
        [
            "typescript error",
            "tsc:",
            "build failed",
            "typecheck",
            "type error",
            "compilation error",
            "cannot find module",
        ],
        "Build or typecheck failed after the edit was applied.",
    ),
    # 3. LLM hallucination — AttributeError/NameError are specific to runtime
    #    attribute access on objects that don't have a method; comes before test/CI
    (
        FailureClass.LLM_HALLUCINATION,
        0.87,
        [
            "attributeerror",
            "nameerror",
            "importerror",
            "has no attribute",
            "modulenotfounderror",
            "is not defined",
        ],
        "The model referenced a method, name, or module that does not exist.",
    ),
    # 4. Push failures — very specific git vocabulary
    (
        FailureClass.PUSH_FAIL,
        0.90,
        ["rejected", "permission denied", "push failed", "non-fast-forward", "remote rejected"],
        "The git push was rejected due to conflicts or permission issues.",
    ),
    # 5. CI failures — workflow-specific vocabulary
    (
        FailureClass.CI_FAIL,
        0.85,
        [
            "github actions",
            "workflow",
            "checks failed",
            "action failed",
            "pipeline failed",
            "ci pipeline",
        ],
        "CI workflow checks failed on the pull request.",
    ),
    # 6. Test failures — require pytest-specific signals or test_ prefix context
    (
        FailureClass.VERIFY_TEST_FAIL,
        0.88,
        [
            "pytest",
            "test_",
            "1 failed",
            "tests failed",
            "assertion failed",
            "assert response",
            "assert result",
            "error in test",
        ],
        "Tests failed after the edit was applied.",
    ),
    # 7. Localization miss — find_string / replacement string absent in file
    #    Uses only localization-specific vocabulary (no generic "failed"/"error")
    (
        FailureClass.LOCALIZATION_MISS,
        0.92,
        [
            "find_string not in file",
            "no match for",
            "not in content",
            "not in file content",
            "string not found in file",
            "could not find",
            "old string not found",
        ],
        "The edit find-string was not present in the target file.",
    ),
]


def _is_localization_error(error_output: str) -> bool:
    """Secondary check: 'not found' alone is ambiguous; require file-edit context."""
    err = error_output.lower()
    # "not found" is valid for localization only when combined with file/content vocabulary
    localization_context = ("file", "content", "replace", "str.replace", "find_string", "edit")
    if "not found" in err:
        return any(ctx in err for ctx in localization_context)
    return False


class FailureAnalyzer:
    """Pattern-based classifier for CodeAct iteration failures.

    No LLM required — all classification is regex/substring matching.
    """

    def analyze(
        self,
        error_output: str,
        edit_script: str = "",
        file_content: str = "",
    ) -> FailureAnalysis:
        """Classify a failed iteration and return structured recovery data."""
        if not error_output and not edit_script:
            return FailureAnalysis(
                failure_class=FailureClass.NONE,
                confidence=0.0,
                root_cause="No error output provided.",
                recovery_instruction=_RECOVERY_INSTRUCTIONS[FailureClass.NONE],
                context={},
            )

        combined = (error_output + "\n" + edit_script).lower()

        for failure_class, base_confidence, patterns, root_cause in _RULES:
            matched_patterns = [p for p in patterns if p in combined]
            if not matched_patterns:
                continue

            # Confidence scales with how many patterns match, capped at base
            confidence = min(base_confidence + 0.02 * (len(matched_patterns) - 1), 0.99)

            # Extract a short relevant snippet from the error for context
            snippet = self._extract_snippet(error_output)

            return FailureAnalysis(
                failure_class=failure_class,
                confidence=round(confidence, 3),
                root_cause=root_cause,
                recovery_instruction=_RECOVERY_INSTRUCTIONS[failure_class],
                context={
                    "matched_patterns": matched_patterns,
                    "snippet": snippet,
                    "error_length": len(error_output),
                },
            )

        # Secondary check: "not found" with file-edit context → LOCALIZATION_MISS
        if _is_localization_error(error_output):
            snippet = self._extract_snippet(error_output)
            return FailureAnalysis(
                failure_class=FailureClass.LOCALIZATION_MISS,
                confidence=0.75,
                root_cause="The edit find-string was not present in the target file.",
                recovery_instruction=_RECOVERY_INSTRUCTIONS[FailureClass.LOCALIZATION_MISS],
                context={
                    "matched_patterns": ["not found"],
                    "snippet": snippet,
                    "error_length": len(error_output),
                },
            )

        return FailureAnalysis(
            failure_class=FailureClass.NONE,
            confidence=0.1,
            root_cause="Could not classify failure from error output.",
            recovery_instruction=_RECOVERY_INSTRUCTIONS[FailureClass.NONE],
            context={"error_length": len(error_output)},
        )

    def recovery_prompt(self, analysis: FailureAnalysis) -> str:
        """Return a formatted string to prepend to the next CodeAct prompt iteration."""
        lines = [
            f"[FAILURE RECOVERY — {analysis.failure_class}]",
            f"Root cause: {analysis.root_cause}",
            f"Action required: {analysis.recovery_instruction}",
        ]
        if analysis.context.get("snippet"):
            lines.append(f"Relevant error snippet: {analysis.context['snippet']}")
        lines.append("")  # trailing blank line before task prompt
        return "\n".join(lines)

    @staticmethod
    def _extract_snippet(error_output: str, max_len: int = 200) -> str:
        """Extract the most informative line(s) from raw error output."""
        if not error_output:
            return ""
        # Prefer lines with "Error", "assert", "FAILED", "not found"
        priority_keywords = ("error", "assert", "failed", "not found", "traceback")
        lines = error_output.splitlines()
        for line in reversed(lines):  # last occurrence is usually most specific
            if any(kw in line.lower() for kw in priority_keywords):
                return line.strip()[:max_len]
        # Fallback: last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()[:max_len]
        return error_output[:max_len]


# ---------------------------------------------------------------------------
# Legacy interface (kept for executor.py backwards compat)
# ---------------------------------------------------------------------------


class FailureType(Enum):
    FIND_STRING_MISMATCH = "find_string_mismatch"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_MISSING = "import_missing"
    FILE_NOT_FOUND = "file_not_found"
    OVERSIZED_TASK = "oversized_task"
    SEQUENCING_DEADLOCK = "sequencing_deadlock"
    UNKNOWN = "unknown"


@dataclass
class FailureDiagnosis:
    failure_type: FailureType
    detail: str
    recovery_hint: str

    def summary(self) -> str:
        detail_truncated = self.detail[:80]
        return f"[{self.failure_type.value}] {detail_truncated} => {self.recovery_hint}"


def classify_failure(script: str, error_output: str, task: str) -> FailureDiagnosis:
    # Classify a CodeAct iteration failure and return a targeted recovery hint.
    err = error_output.lower()

    # Check if this is a config/shell file edit failure
    config_extensions = (".sh", ".env", ".toml", "Dockerfile", ".conf", ".cfg", ".ini")
    script_has_config = any(ext in script for ext in config_extensions)

    if (
        "find string not" in err
        or "not in content" in err
        or "could not find" in err
        or "valueerror" in err
        and "not found" in err
    ):
        hint = "Read the file content first and copy the find string verbatim — do not paraphrase or reconstruct from memory."
        if script_has_config:
            hint = "This is a config/shell file — do NOT use str.replace(). Read the whole file, make changes in memory, and write the entire content back with Path.write_text()."
        return FailureDiagnosis(FailureType.FIND_STRING_MISMATCH, error_output[:200], hint)

    if "syntaxerror" in err or "unexpected" in err and "line" in err:
        return FailureDiagnosis(
            FailureType.SYNTAX_ERROR,
            error_output[:200],
            "Check the generated script for unmatched quotes, brackets, or indentation errors.",
        )

    if "nameerror" in err or "importerror" in err or "modulenotfounderror" in err:
        return FailureDiagnosis(
            FailureType.IMPORT_MISSING,
            error_output[:200],
            "Add the missing import at the top of the script before using the name.",
        )

    if "filenotfounderror" in err or "no such file" in err:
        return FailureDiagnosis(
            FailureType.FILE_NOT_FOUND,
            error_output[:200],
            "Verify the file path exists. Use Path(filepath).exists() before reading.",
        )

    if "already defined" in err or "duplicate" in err or "typeerror" in err and "argument" in err:
        return FailureDiagnosis(
            FailureType.SEQUENCING_DEADLOCK,
            error_output[:200],
            "Define the function or class before adding calls to it. Check for duplicate definitions.",
        )

    return FailureDiagnosis(
        FailureType.UNKNOWN,
        error_output[:200],
        "Review the full error output and try a more targeted single-operation approach.",
    )


def get_recovery_context(script: str, error_output: str, task: str) -> str:
    # Build a formatted recovery context string combining error tail and diagnosis hint.
    diagnosis = classify_failure(script, error_output, task)
    error_tail = error_output[-300:] if len(error_output) > 300 else error_output
    return f"Error:\n{error_tail}\nDiagnosis ({diagnosis.failure_type.value}): {diagnosis.recovery_hint}"


class SkillStore:
    """Manage skill files that codify repeated fixes."""

    def __init__(self, skills_dir: str | Path = None):
        if skills_dir is None:
            from config import BASE_DIR

            skills_dir = BASE_DIR / "skills"
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)

    def all_skills(self) -> list[dict]:
        """Return all skill files as dicts."""
        skills = []
        for skill_file in self.skills_dir.glob("*.md"):
            try:
                content = skill_file.read_text()
                skills.append(
                    {
                        "path": str(skill_file),
                        "name": skill_file.stem,
                        "content": content,
                    }
                )
            except Exception:
                pass
        return skills

    def find_matching_skill(self, task: str) -> dict | None:
        """Fuzzy match task description against skill files. Return best match."""
        import difflib

        task_lower = task.lower()
        best_match = None
        best_ratio = 0

        for skill in self.all_skills():
            # Check title and pattern sections
            skill_text = skill["content"].lower()
            ratio = difflib.SequenceMatcher(None, task_lower, skill_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = skill

        return best_match if best_ratio > 0.3 else None

    def emit_skill_file(self, pattern: str, failure_type: str, recovery_hint: str) -> str:
        """Write a skill file for a repeated pattern."""
        from datetime import datetime

        # Slug: lowercase, replace spaces with underscores, limit to 50 chars
        slug = re.sub(r"[^a-z0-9_]", "_", pattern.lower())[:50]
        slug = re.sub(r"_+", "_", slug).strip("_")
        skill_path = self.skills_dir / f"{slug}.md"

        content = f"""# Skill: {pattern}

## Pattern
{pattern}

## Failure Type
{failure_type}

## Recovery Hint
{recovery_hint}

## Metadata
- Created: {datetime.utcnow().isoformat()}
- Usage Count: 1
- Last Validated: {datetime.utcnow().isoformat()}
"""
        skill_path.write_text(content)
        return str(skill_path)


def _emit_skill_file(failure_diagnosis: FailureDiagnosis, task: str) -> None:
    """Emit a skill file when a pattern is identified."""
    try:
        store = SkillStore()
        pattern = f"{failure_diagnosis.failure_type.value}: {task[:80]}"
        store.emit_skill_file(
            pattern=pattern,
            failure_type=failure_diagnosis.failure_type.value,
            recovery_hint=failure_diagnosis.recovery_hint,
        )
    except Exception:
        pass  # Silent: skill emission is optional
