"""
SOMA self-improvement task: Structured Failure Classification.

Add failure_class enum to the experiences SQLite schema.
Classify failures at recording time from VerifyResult/EditResult structs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.executor import execute_edit
from core.llm import LLMClient
from config import TIER_3_MODEL

REPO_PATH = Path(__file__).parent.parent

TASK = """
Add structured failure classification to SOMA's experience system.

## Step 1: Modify core/experience.py

1. Add a `FailureClass` string enum after the imports:
```python
class FailureClass:
    LOCALIZATION_MISS = "LOCALIZATION_MISS"   # wrong file identified
    EDIT_SYNTAX_ERROR = "EDIT_SYNTAX_ERROR"   # generated code has syntax errors
    VERIFY_BUILD_FAIL = "VERIFY_BUILD_FAIL"   # build/compile failed after edit
    VERIFY_TEST_FAIL  = "VERIFY_TEST_FAIL"    # tests failed after edit
    CI_FAIL           = "CI_FAIL"             # CI checks failed on PR
    LLM_HALLUCINATION = "LLM_HALLUCINATION"   # model returned non-actionable output
    PUSH_FAIL         = "PUSH_FAIL"           # git push/PR creation failed
    NONE              = ""                    # success or unclassified
```

2. In `ExperienceStore._init_db()`, add a migration that adds `failure_class TEXT DEFAULT ''`
   column to the experiences table if it doesn't already exist:
```python
try:
    self.conn.execute("ALTER TABLE experiences ADD COLUMN failure_class TEXT DEFAULT ''")
    self.conn.commit()
except sqlite3.OperationalError:
    pass  # column already exists
```
   Add this AFTER the existing CREATE TABLE statements.

3. In `ExperienceStore.record()` method, add `failure_class: str = ""` parameter and include it
   in the INSERT statement. Find the record() method and add the parameter + column.

## Step 2: Modify orchestrator.py

Find every call to `self.store.record(...)` where `success=False` is passed and add the
appropriate `failure_class=` keyword argument based on the context:

- If the outcome string contains "Verification failed" and "test" (case-insensitive):
  use `failure_class=FailureClass.VERIFY_TEST_FAIL`
- If the outcome string contains "Verification failed" and "build":
  use `failure_class=FailureClass.VERIFY_BUILD_FAIL`
- If the outcome string contains "Verification failed" (generic):
  use `failure_class=FailureClass.VERIFY_BUILD_FAIL`
- If the context is about locating/identifying files and success=False:
  use `failure_class=FailureClass.LOCALIZATION_MISS`
- If the outcome contains "Push failed" or "PR failed":
  use `failure_class=FailureClass.PUSH_FAIL`
- If the outcome contains "did not provide" or "only provided partial" or "no actionable":
  use `failure_class=FailureClass.LLM_HALLUCINATION`

Also add `from core.experience import FailureClass` to orchestrator.py imports if not already there.

## Step 3: Add failure_stats() to ExperienceStore

Add this method to ExperienceStore:
```python
def failure_stats(self) -> dict:
    \"\"\"Return counts of each failure class across all failed experiences.\"\"\"
    rows = self.conn.execute(
        \"\"\"SELECT failure_class, COUNT(*) as cnt
           FROM experiences
           WHERE success=0 AND failure_class != ''
           GROUP BY failure_class
           ORDER BY cnt DESC\"\"\"
    ).fetchall()
    return {r["failure_class"]: r["cnt"] for r in rows}
```

Constraints:
- Do not change any existing method signatures (record() gets a new optional kwarg only)
- Do not break existing callers — failure_class defaults to ""
- The ALTER TABLE migration must be idempotent (try/except already handles this)
"""

file_contexts = {
    "core/experience.py": (REPO_PATH / "core/experience.py").read_text(),
    "orchestrator.py": (REPO_PATH / "orchestrator.py").read_text(),
}

llm = LLMClient()
print("[Task] Structured Failure Classification")
print(f"[Task] Model: {TIER_3_MODEL}")
print("[Task] Running CodeAct executor...")

result = execute_edit(TASK, file_contexts, REPO_PATH, llm, TIER_3_MODEL)

if result.success:
    print(f"[Task] SUCCESS in {result.iterations} iteration(s)")
    print(f"[Task] Files changed: {result.files_changed}")
else:
    print(f"[Task] FAILED after {result.iterations} iteration(s)")
    print(f"[Task] Error: {result.error}")
    sys.exit(1)
