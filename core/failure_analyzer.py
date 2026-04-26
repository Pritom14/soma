# core/failure_analyzer.py - classify CodeAct iteration failures for targeted recovery
from __future__ import annotations
from enum import Enum
from pathlib import Path


class FailureType(Enum):
    FIND_STRING_MISMATCH = "find_string_mismatch"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_MISSING = "import_missing"
    FILE_NOT_FOUND = "file_not_found"
    OVERSIZED_TASK = "oversized_task"
    SEQUENCING_DEADLOCK = "sequencing_deadlock"
    UNKNOWN = "unknown"


from dataclasses import dataclass


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
        import re

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
