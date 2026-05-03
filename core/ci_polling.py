"""
core/ci_polling.py - CI-aware pre-submit verification loop.

After PR creation, polls GitHub checks until all conclude.
On CI failure: extracts stderr/stdout, creates new CodeAct iteration with CI failure.
On success or max retries: exits loop and returns result.
"""
from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import Optional

from core import github


@dataclass
class CICheckResult:
    """Result of a single CI check."""
    name: str
    state: str
    conclusion: Optional[str]  # "success", "failure", "neutral", "skipped", or None (pending)

    @property
    def is_pending(self) -> bool:
        return self.conclusion is None

    @property
    def is_failed(self) -> bool:
        return self.conclusion == "failure"

    @property
    def is_success(self) -> bool:
        return self.conclusion == "success"


@dataclass
class CIPollingResult:
    """Result of CI polling loop."""
    success: bool
    all_checks_passed: bool
    final_checks: list[CICheckResult]
    failure_summary: str = ""
    retries: int = 0


def _extract_pr_number(pr_url: str) -> Optional[int]:
    """Extract PR number from GitHub PR URL.

    Args:
        pr_url: PR URL like https://github.com/owner/repo/pull/123

    Returns:
        PR number or None if extraction fails.
    """
    match = re.search(r'/pull/(\d+)(?:/|$)', pr_url)
    return int(match.group(1)) if match else None


def _parse_checks(checks_json: dict) -> list[CICheckResult]:
    """Parse gh pr checks JSON output into CICheckResult objects.

    Args:
        checks_json: Dict with "checks" key containing list of check objects

    Returns:
        List of CICheckResult objects.
    """
    results = []
    for check in checks_json.get("checks", []):
        results.append(
            CICheckResult(
                name=check.get("name", "unknown"),
                state=check.get("state", "unknown"),
                conclusion=check.get("conclusion"),  # None if pending
            )
        )
    return results


def _has_pending_checks(checks: list[CICheckResult]) -> bool:
    """Check if any checks are still pending.

    Args:
        checks: List of CICheckResult objects.

    Returns:
        True if any check is pending.
    """
    return any(c.is_pending for c in checks)


def _get_failed_checks(checks: list[CICheckResult]) -> list[CICheckResult]:
    """Get list of failed checks.

    Args:
        checks: List of CICheckResult objects.

    Returns:
        List of failed checks.
    """
    return [c for c in checks if c.is_failed]


def _build_failure_context(failed_checks: list[CICheckResult]) -> str:
    """Build a summary of failed checks for LLM context.

    Args:
        failed_checks: List of failed CICheckResult objects.

    Returns:
        Formatted failure summary.
    """
    if not failed_checks:
        return ""

    lines = ["CI check failures detected:"]
    for check in failed_checks:
        lines.append(f"- {check.name}: {check.conclusion}")
    return "\n".join(lines)


def poll_ci_checks(
    repo: str,
    pr_url: str,
    max_retries: int = 5,
    poll_interval: int = 10,
) -> CIPollingResult:
    """Poll GitHub checks until all conclude or max retries exceeded.

    After PR creation, this function:
    1. Extracts PR number from pr_url
    2. Polls get_pr_checks() every poll_interval seconds
    3. Exits when no pending checks remain or max_retries exceeded
    4. Returns summary of results

    Args:
        repo: Repository in owner/repo format (e.g., "facebook/react")
        pr_url: PR URL from github.create_pr() (e.g., https://github.com/owner/repo/pull/123)
        max_retries: Maximum polling attempts (default 5, ~50s at 10s interval)
        poll_interval: Seconds between polls (default 10)

    Returns:
        CIPollingResult with success status, final check states, and failure summary.
    """
    pr_number = _extract_pr_number(pr_url)
    if pr_number is None:
        return CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[],
            failure_summary=f"Could not extract PR number from URL: {pr_url}",
            retries=0,
        )

    print(f"[SOMA] CI Polling: PR #{pr_number}, max_retries={max_retries}")

    final_checks = []
    failed_checks = []

    for attempt in range(1, max_retries + 1):
        checks_json = github.get_pr_checks(repo, pr_number)
        checks = _parse_checks(checks_json)
        final_checks = checks

        pending = _has_pending_checks(checks)
        failed = _get_failed_checks(checks)

        status_msg = f"[SOMA] CI Poll {attempt}/{max_retries}: "
        status_msg += f"{len(checks)} check(s)"
        if pending:
            status_msg += f", {sum(1 for c in checks if c.is_pending)} pending"
        if failed:
            status_msg += f", {len(failed)} failed"
        print(status_msg)

        # Exit loop early if all checks have concluded
        if not pending:
            failed_checks = failed
            print("[SOMA] CI Checks concluded (no pending)")
            break

        # If this is not the last attempt, sleep before polling again
        if attempt < max_retries:
            time.sleep(poll_interval)

    # Determine overall success
    failed_checks = _get_failed_checks(final_checks)
    all_passed = len(final_checks) > 0 and all(c.is_success for c in final_checks)

    failure_summary = ""
    if failed_checks:
        failure_summary = _build_failure_context(failed_checks)

    return CIPollingResult(
        success=all_passed or (len(final_checks) == 0),  # No checks = success
        all_checks_passed=all_passed,
        final_checks=final_checks,
        failure_summary=failure_summary,
        retries=max_retries,
    )


def extract_ci_failure_for_retry(
    repo: str,
    pr_number: int,
    failed_checks: list[CICheckResult],
) -> str:
    """Extract CI failure details for retry context.

    In a full implementation, this would fetch detailed logs from CI.
    For now, it returns a summary suitable for LLM re-analysis.

    Args:
        repo: Repository in owner/repo format
        pr_number: PR number
        failed_checks: List of failed checks

    Returns:
        Formatted failure details for LLM context.
    """
    lines = [
        f"CI checks failed on PR #{pr_number} in {repo}",
        "Failed checks:",
    ]

    for check in failed_checks:
        lines.append(f"- {check.name} ({check.state}): {check.conclusion}")

    lines.append(
        "\nTo fix this:\n"
        "1. Analyze the failure reason\n"
        "2. Identify which code change caused the failure\n"
        "3. Generate a corrective edit\n"
        "4. Force-push the updated code to the branch"
    )

    return "\n".join(lines)
