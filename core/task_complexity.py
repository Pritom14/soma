# core/task_complexity.py - heuristic task complexity scoring before CodeAct
from __future__ import annotations

COMPLEXITY_THRESHOLD = 30


def score_task(task: str, file_contexts: dict) -> tuple[int, str]:
    # Score a task for complexity. Returns (score, reason). Score > COMPLEXITY_THRESHOLD means decompose.
    score = 0
    reasons = []

    file_count = len(file_contexts)
    if file_count > 1:
        score += (file_count - 1) * 10
        reasons.append(f"{file_count} files touched")

    import re

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
    # Return True if task should be decomposed before CodeAct.
    score, _ = score_task(task, file_contexts)
    return score > COMPLEXITY_THRESHOLD
