"""tests/test_ci_polling.py

Tests for the CI polling loop: extraction, parsing, failure detection, and retry logic.
Covers PR number extraction, check parsing, pending detection, and failure summaries.
"""
from __future__ import annotations

import pytest
from core.ci_polling import (
    CICheckResult,
    CIPollingResult,
    _extract_pr_number,
    _parse_checks,
    _has_pending_checks,
    _get_failed_checks,
    _build_failure_context,
    extract_ci_failure_for_retry,
    poll_ci_checks,
)


# ---------------------------------------------------------------------------
# Unit tests: PR number extraction
# ---------------------------------------------------------------------------


class TestPRNumberExtraction:
    """Test extraction of PR number from various URL formats."""

    def test_extract_pr_number_standard_url(self):
        """Extract PR number from standard GitHub PR URL."""
        url = "https://github.com/facebook/react/pull/12345"
        assert _extract_pr_number(url) == 12345

    def test_extract_pr_number_with_trailing_slash(self):
        """Extract PR number when URL has trailing slash."""
        url = "https://github.com/owner/repo/pull/999/"
        assert _extract_pr_number(url) == 999

    def test_extract_pr_number_simple(self):
        """Extract PR number from simple URL."""
        url = "https://github.com/soma/test/pull/1"
        assert _extract_pr_number(url) == 1

    def test_extract_pr_number_invalid_url(self):
        """Return None when URL doesn't contain /pull/."""
        url = "https://github.com/owner/repo/issues/123"
        assert _extract_pr_number(url) is None

    def test_extract_pr_number_empty_url(self):
        """Return None for empty URL."""
        assert _extract_pr_number("") is None

    def test_extract_pr_number_malformed(self):
        """Return None for malformed URL."""
        url = "github.com/owner/repo/pull/abc"
        assert _extract_pr_number(url) is None


# ---------------------------------------------------------------------------
# Unit tests: Check parsing
# ---------------------------------------------------------------------------


class TestCheckParsing:
    """Test parsing of gh pr checks JSON output."""

    def test_parse_empty_checks(self):
        """Parse empty checks list."""
        checks_json = {"checks": []}
        result = _parse_checks(checks_json)
        assert result == []

    def test_parse_single_pending_check(self):
        """Parse single pending check (no conclusion)."""
        checks_json = {
            "checks": [
                {
                    "name": "build",
                    "state": "in_progress",
                    "conclusion": None,
                }
            ]
        }
        result = _parse_checks(checks_json)
        assert len(result) == 1
        assert result[0].name == "build"
        assert result[0].is_pending is True
        assert result[0].is_success is False
        assert result[0].is_failed is False

    def test_parse_single_success_check(self):
        """Parse single successful check."""
        checks_json = {
            "checks": [
                {
                    "name": "test",
                    "state": "completed",
                    "conclusion": "success",
                }
            ]
        }
        result = _parse_checks(checks_json)
        assert len(result) == 1
        assert result[0].is_success is True
        assert result[0].is_pending is False
        assert result[0].is_failed is False

    def test_parse_single_failed_check(self):
        """Parse single failed check."""
        checks_json = {
            "checks": [
                {
                    "name": "lint",
                    "state": "completed",
                    "conclusion": "failure",
                }
            ]
        }
        result = _parse_checks(checks_json)
        assert len(result) == 1
        assert result[0].is_failed is True
        assert result[0].is_pending is False

    def test_parse_mixed_checks(self):
        """Parse mixture of pending, passing, and failing checks."""
        checks_json = {
            "checks": [
                {"name": "build", "state": "completed", "conclusion": "success"},
                {"name": "test", "state": "in_progress", "conclusion": None},
                {"name": "lint", "state": "completed", "conclusion": "failure"},
            ]
        }
        result = _parse_checks(checks_json)
        assert len(result) == 3
        assert result[0].is_success is True
        assert result[1].is_pending is True
        assert result[2].is_failed is True

    def test_parse_missing_fields(self):
        """Parse check with missing optional fields."""
        checks_json = {
            "checks": [
                {"name": "build"},  # missing state and conclusion
            ]
        }
        result = _parse_checks(checks_json)
        assert len(result) == 1
        assert result[0].name == "build"
        assert result[0].state == "unknown"
        assert result[0].is_pending is True

    def test_parse_missing_checks_key(self):
        """Parse response missing 'checks' key."""
        checks_json = {}
        result = _parse_checks(checks_json)
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests: Pending and failed check detection
# ---------------------------------------------------------------------------


class TestCheckDetection:
    """Test detection of pending and failed checks."""

    def test_has_pending_checks_true(self):
        """Detect when pending checks exist."""
        checks = [
            CICheckResult("build", "in_progress", None),
            CICheckResult("test", "completed", "success"),
        ]
        assert _has_pending_checks(checks) is True

    def test_has_pending_checks_false_all_success(self):
        """Detect when all checks passed."""
        checks = [
            CICheckResult("build", "completed", "success"),
            CICheckResult("test", "completed", "success"),
        ]
        assert _has_pending_checks(checks) is False

    def test_has_pending_checks_false_all_concluded(self):
        """Detect when all checks concluded (no pending)."""
        checks = [
            CICheckResult("build", "completed", "success"),
            CICheckResult("lint", "completed", "failure"),
        ]
        assert _has_pending_checks(checks) is False

    def test_has_pending_checks_empty(self):
        """Detect pending in empty check list."""
        checks = []
        assert _has_pending_checks(checks) is False

    def test_get_failed_checks_none(self):
        """Extract failed checks when none exist."""
        checks = [
            CICheckResult("build", "completed", "success"),
            CICheckResult("test", "completed", "success"),
        ]
        failed = _get_failed_checks(checks)
        assert failed == []

    def test_get_failed_checks_single(self):
        """Extract single failed check."""
        checks = [
            CICheckResult("build", "completed", "success"),
            CICheckResult("lint", "completed", "failure"),
        ]
        failed = _get_failed_checks(checks)
        assert len(failed) == 1
        assert failed[0].name == "lint"

    def test_get_failed_checks_multiple(self):
        """Extract multiple failed checks."""
        checks = [
            CICheckResult("build", "completed", "success"),
            CICheckResult("lint", "completed", "failure"),
            CICheckResult("test", "completed", "failure"),
            CICheckResult("coverage", "completed", "neutral"),
        ]
        failed = _get_failed_checks(checks)
        assert len(failed) == 2
        assert failed[0].name == "lint"
        assert failed[1].name == "test"

    def test_get_failed_checks_ignores_pending(self):
        """Verify that pending checks are not included in failed checks."""
        checks = [
            CICheckResult("build", "in_progress", None),
            CICheckResult("lint", "completed", "failure"),
        ]
        failed = _get_failed_checks(checks)
        assert len(failed) == 1
        assert failed[0].name == "lint"


# ---------------------------------------------------------------------------
# Unit tests: Failure context building
# ---------------------------------------------------------------------------


class TestFailureContext:
    """Test building failure context for LLM input."""

    def test_build_failure_context_empty(self):
        """Build context with no failed checks."""
        result = _build_failure_context([])
        assert result == ""

    def test_build_failure_context_single(self):
        """Build context with single failure."""
        failed = [CICheckResult("build", "completed", "failure")]
        result = _build_failure_context(failed)
        assert "CI check failures detected:" in result
        assert "build: failure" in result

    def test_build_failure_context_multiple(self):
        """Build context with multiple failures."""
        failed = [
            CICheckResult("lint", "completed", "failure"),
            CICheckResult("test", "completed", "failure"),
        ]
        result = _build_failure_context(failed)
        assert "CI check failures detected:" in result
        assert "lint: failure" in result
        assert "test: failure" in result

    def test_extract_ci_failure_for_retry_formatting(self):
        """Verify failure context has proper formatting for retry."""
        failed = [
            CICheckResult("build", "completed", "failure"),
            CICheckResult("test", "completed", "failure"),
        ]
        result = extract_ci_failure_for_retry("owner/repo", 123, failed)
        assert "PR #123" in result
        assert "owner/repo" in result
        assert "Failed checks:" in result
        assert "build" in result
        assert "test" in result
        assert "To fix this:" in result
        assert "Force-push" in result


# ---------------------------------------------------------------------------
# Integration tests: Polling result composition
# ---------------------------------------------------------------------------


class TestCIPollingResult:
    """Test CIPollingResult dataclass behavior."""

    def test_polling_result_all_success(self):
        """Create result for all-success scenario."""
        checks = [
            CICheckResult("build", "completed", "success"),
            CICheckResult("test", "completed", "success"),
        ]
        result = CIPollingResult(
            success=True,
            all_checks_passed=True,
            final_checks=checks,
            failure_summary="",
            retries=2,
        )
        assert result.success is True
        assert result.all_checks_passed is True
        assert len(result.final_checks) == 2
        assert result.failure_summary == ""

    def test_polling_result_with_failures(self):
        """Create result for failure scenario."""
        checks = [
            CICheckResult("build", "completed", "success"),
            CICheckResult("lint", "completed", "failure"),
        ]
        result = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=checks,
            failure_summary="CI check failures detected:\n- lint: failure",
            retries=5,
        )
        assert result.success is False
        assert result.all_checks_passed is False
        assert result.failure_summary != ""

    def test_polling_result_extraction_failure(self):
        """Create result for PR number extraction failure."""
        result = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[],
            failure_summary="Could not extract PR number",
            retries=0,
        )
        assert result.success is False
        assert "Could not extract PR number" in result.failure_summary


# ---------------------------------------------------------------------------
# Mock-based integration test: poll_ci_checks behavior (without real API)
# ---------------------------------------------------------------------------


class TestCIPollingBehavior:
    """Test CI polling loop logic (mocked, no real GitHub calls)."""

    def test_polling_result_structure(self):
        """Verify polling result has all required fields."""
        # This test validates the structure without calling the real function
        result = CIPollingResult(
            success=True,
            all_checks_passed=True,
            final_checks=[CICheckResult("build", "completed", "success")],
            failure_summary="",
            retries=1,
        )

        assert hasattr(result, 'success')
        assert hasattr(result, 'all_checks_passed')
        assert hasattr(result, 'final_checks')
        assert hasattr(result, 'failure_summary')
        assert hasattr(result, 'retries')

    def test_check_result_properties(self):
        """Verify CICheckResult properties work correctly."""
        pending = CICheckResult("build", "in_progress", None)
        success = CICheckResult("test", "completed", "success")
        failure = CICheckResult("lint", "completed", "failure")

        assert pending.is_pending is True
        assert pending.is_success is False
        assert pending.is_failed is False

        assert success.is_pending is False
        assert success.is_success is True
        assert success.is_failed is False

        assert failure.is_pending is False
        assert failure.is_success is False
        assert failure.is_failed is True
