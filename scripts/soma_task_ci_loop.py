"""
SOMA self-improvement task: CI-Aware Pre-Submit Verification Loop.

After SOMA creates a PR, poll get_pr_checks() for up to 10 minutes.
If CI fails, feed the failure output back into the CodeAct loop and force-push a fix.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.executor import execute_edit
from core.llm import LLMClient
from config import TIER_3_MODEL

REPO_PATH = Path(__file__).parent.parent

TASK = """
Modify orchestrator.py to add a CI-Aware Pre-Submit Verification Loop.

After PR creation succeeds in the `contribute()` method (after the `pr = github.create_pr(...)` call
and the `if pr.success:` block), add a new step that:

1. Polls `github.get_pr_checks(repo, pr_number)` in a loop with 30s sleep between attempts.
   - Maximum wait: 10 minutes (20 attempts)
   - Skip if `pr.success` is False
   - Get `pr_number` from the PR URL using `github.issue_number_from_url(pr.url)`

2. If all checks pass (all conclusions == "success" or "skipped"): log "[SOMA] CI: all checks passed"
   and return the existing success dict with `"ci": "passed"` added.

3. If any check has conclusion == "failure":
   a. Collect the failing check names into a string
   b. Re-run `execute_edit()` with the SAME task/file_contexts/repo_path/llm/model BUT with
      an updated `beliefs_context` that appends:
      "CI failed checks: {failing_names}. Fix the failing checks before re-pushing."
   c. If the re-edit succeeds, force-push the branch:
      Run `git push --force-with-lease origin {branch}` via subprocess in repo_path
   d. Add `"ci": "fixed"` or `"ci": "failed"` to the return dict

4. If checks are still pending after 10 minutes, return with `"ci": "timeout"`.

5. Wrap the entire CI polling block in try/except so a CI check failure never prevents the PR URL
   from being returned — just add `"ci_error": str(e)` to the return dict.

Important constraints:
- Do not change the function signature of `contribute()`
- Do not touch any other methods
- The force-push uses subprocess.run(['git', 'push', '--force-with-lease', 'origin', branch], ...)
  with cwd=repo_path
- Only add this after `if pr.success:` — if PR creation failed, skip CI polling entirely
"""

# Provide just the contribute() tail as context — lines 2183-2220
with open(REPO_PATH / "orchestrator.py") as f:
    orch_lines = f.readlines()
# Find the contribute method start
start = next(i for i, l in enumerate(orch_lines) if "def contribute(" in l)
file_contexts = {
    "orchestrator.py": "".join(orch_lines[start:]),
    "core/github.py": (REPO_PATH / "core/github.py").read_text(),
}

llm = LLMClient()
print("[Task] CI-Aware Pre-Submit Verification Loop")
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
