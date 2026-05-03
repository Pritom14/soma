# core/task_complexity.py - heuristic task complexity scoring before CodeAct
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Legacy constants kept for backward compatibility
COMPLEXITY_THRESHOLD = 30


@dataclass
class ComplexityScore:
    score: float  # 0.0 - 1.0
    file_count: int
    operation_count: int
    nesting_depth: int
    special_char_penalty: float  # triple-quotes, nested classes, decorators
    recommendation: str  # "direct" | "decompose" | "reject"
    reasons: list[str] = field(default_factory=list)


class TaskComplexityScorer:
    DECOMPOSE_THRESHOLD = 0.6
    REJECT_THRESHOLD = 0.9

    # Keywords that imply broad, multi-file refactoring
    _HIGH_COMPLEXITY_KEYWORDS = re.compile(
        r"\b(refactor|rename across|migrate|rewrite|restructure|overhaul)\b",
        re.IGNORECASE,
    )

    # Operation verbs that each represent a distinct code change
    _OP_PATTERN = re.compile(
        r"\b(add|remove|replace|insert|delete|rename|change|update|create|modify|extract|inline|move)\b",
        re.IGNORECASE,
    )

    def score(self, task: str, file_paths: list[str] | None = None) -> ComplexityScore:
        """Score a task on a 0.0–1.0 scale and return a ComplexityScore."""
        raw = 0.0
        reasons: list[str] = []

        # --- file_count factor ---
        file_count = len(file_paths) if file_paths else self._estimate_file_count(task)
        if file_count >= 3:
            raw += 0.3
            reasons.append(f"touches {file_count} files (>= 3)")

        # --- operation_count factor ---
        op_matches = self._OP_PATTERN.findall(task)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ops: list[str] = []
        for op in op_matches:
            key = op.lower()
            if key not in seen:
                seen.add(key)
                unique_ops.append(op)
        operation_count = len(unique_ops)
        if operation_count > 5:
            raw += 0.25
            reasons.append(f"{operation_count} distinct operations (> 5)")

        # --- nesting_depth factor ---
        nesting_depth = self._estimate_nesting_depth(task)
        if nesting_depth > 2:
            raw += 0.2
            reasons.append(f"nesting depth {nesting_depth} (> 2)")

        # --- special_char_penalty ---
        special_char_penalty = 0.0
        triple_count = task.count('"""') + task.count("'''")
        if triple_count > 0:
            special_char_penalty += 0.1
            reasons.append("contains triple-quoted strings")

        decorator_count = task.count("@")
        if decorator_count > 0:
            special_char_penalty += 0.1
            reasons.append(f"{decorator_count} decorator(s) in task")

        metaclass_count = len(re.findall(r"\bmetaclass\b", task, re.IGNORECASE))
        if metaclass_count > 0:
            special_char_penalty += 0.1
            reasons.append("metaclass usage in task")

        raw += special_char_penalty

        # --- task length factor ---
        if len(task) > 500:
            raw += 0.1
            reasons.append(f"task length {len(task)} chars (> 500)")

        # --- high-complexity keyword factor ---
        kw_matches = self._HIGH_COMPLEXITY_KEYWORDS.findall(task)
        if kw_matches:
            raw += 0.15
            reasons.append(f"high-complexity keyword(s): {', '.join(set(k.lower() for k in kw_matches))}")

        # Clamp to [0.0, 1.0]
        final_score = min(raw, 1.0)

        recommendation = self._recommend(final_score)

        return ComplexityScore(
            score=final_score,
            file_count=file_count,
            operation_count=operation_count,
            nesting_depth=nesting_depth,
            special_char_penalty=special_char_penalty,
            recommendation=recommendation,
            reasons=reasons if reasons else ["simple task"],
        )

    def should_decompose(self, score: ComplexityScore) -> bool:
        return self.DECOMPOSE_THRESHOLD <= score.score < self.REJECT_THRESHOLD

    def should_reject(self, score: ComplexityScore) -> bool:
        return score.score >= self.REJECT_THRESHOLD

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recommend(self, score: float) -> str:
        if score >= self.REJECT_THRESHOLD:
            return "reject"
        if score >= self.DECOMPOSE_THRESHOLD:
            return "decompose"
        return "direct"

    def _estimate_file_count(self, task: str) -> int:
        """Heuristically count file paths mentioned in the task text."""
        # Match anything that looks like a relative or absolute file path
        paths = re.findall(r"[\w./\\-]+\.\w{1,6}", task)
        return len(set(paths))

    def _estimate_nesting_depth(self, task: str) -> int:
        """Estimate structural nesting depth from code snippets in the task."""
        depth = 0
        # class inside class
        if re.search(r"\bclass\b.*\bclass\b", task, re.DOTALL):
            depth += 1
        # function inside function
        if re.search(r"\bdef\b.*\bdef\b", task, re.DOTALL):
            depth += 1
        # nested brackets: ((  or {{  or [[
        if re.search(r"[({[][({[]", task):
            depth += 1
        return depth


# ------------------------------------------------------------------
# Legacy API — kept for backwards compatibility with existing callers
# ------------------------------------------------------------------

def score_task(task: str, file_contexts: dict) -> tuple[int, str]:
    """Legacy integer-scale scorer. Returns (score, reason). Score > 30 means decompose."""
    score = 0
    reasons = []

    file_count = len(file_contexts)
    if file_count > 1:
        score += (file_count - 1) * 10
        reasons.append(f"{file_count} files touched")

    op_keywords = re.findall(
        r"\b(add|remove|replace|insert|delete|rename|change|update|create|modify)\b",
        task.lower(),
    )
    op_count = len(set(op_keywords))
    if op_count > 2:
        score += (op_count - 2) * 5
        reasons.append(f"{op_count} distinct operations")

    if len(task) > 500:
        score += 10
        reasons.append("long task description")

    if "class " in task and "def " in task:
        score += 20
        reasons.append("nested class/function definitions in task")

    triple_quote_count = task.count(3 * chr(34)) + task.count(3 * chr(39))
    if triple_quote_count > 0:
        score += 15
        reasons.append("triple-quoted strings in task")

    reason = "; ".join(reasons) if reasons else "simple task"
    return score, reason


def is_complex(task: str, file_contexts: dict) -> bool:
    """Return True if task should be decomposed before CodeAct."""
    score, _ = score_task(task, file_contexts)
    return score > COMPLEXITY_THRESHOLD
