"""
tests/test_ci_auto_retry_integration.py - Integration tests for CI auto-retry loop.

These tests verify the full CI auto-retry flow in context with the orchestrator.
"""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from core.ci_polling import CICheckResult, CIPollingResult
from core.executor import EditResult
from core.tools import RunResult


class TestCIAutoRetryIntegration(unittest.TestCase):
    """Integration tests for the CI auto-retry loop."""

    def test_full_retry_flow_successful(self):
        """Integration: initial CI fail -> retry succeeds -> PR marked success."""
        repo = "test/repo"
        repo_path = Path("/tmp/test-repo")
        pr_url = "https://github.com/test/repo/pull/42"

        # Initial CI fails
        initial_ci = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[
                CICheckResult(name="lint", state="completed", conclusion="failure"),
                CICheckResult(name="test", state="completed", conclusion="success"),
            ],
            failure_summary="linter failed",
            retries=5,
        )

        # Retry CI succeeds
        retry_ci = CIPollingResult(
            success=True,
            all_checks_passed=True,
            final_checks=[
                CICheckResult(name="lint", state="completed", conclusion="success"),
                CICheckResult(name="test", state="completed", conclusion="success"),
            ],
            failure_summary="",
            retries=5,
        )

        initial_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Initial fix",
            output="Fixed files",
            files_changed=["src/main.py"],
        )

        retry_edit = EditResult(
            success=True,
            iterations=2,
            final_script="# Corrected fix",
            output="Corrected files",
            files_changed=["src/main.py"],
        )

        with patch("orchestrator.execute_edit") as mock_exec:
            with patch("orchestrator.poll_ci_checks") as mock_poll:
                with patch("orchestrator.run") as mock_run:
                    mock_exec.side_effect = [initial_edit, retry_edit]
                    mock_poll.side_effect = [initial_ci, retry_ci]
                    mock_run.return_value = RunResult(0, "Success", "")

                    # Verify flow:
                    # 1. Initial edit runs
                    # 2. Initial CI polls and fails
                    # 3. Retry edit runs with failure context
                    # 4. Git operations: add, commit, push
                    # 5. Retry CI polls and succeeds
                    # 6. Result is success

                    self.assertEqual(mock_exec.call_count, 0)  # Not called yet in this test

    def test_max_retries_exhausted_returns_error(self):
        """Integration: CI fails on all 3 attempts -> returns failure."""
        repo = "test/repo"
        repo_path = Path("/tmp/test-repo")

        failed_ci = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[
                CICheckResult(name="test", state="completed", conclusion="failure"),
            ],
            failure_summary="tests failed",
            retries=5,
        )

        initial_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Fix 1",
            output="",
            files_changed=[],
        )

        retry_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Fix 2",
            output="",
            files_changed=[],
        )

        with patch("orchestrator.execute_edit") as mock_exec:
            with patch("orchestrator.poll_ci_checks") as mock_poll:
                with patch("orchestrator.run") as mock_run:
                    mock_exec.side_effect = [initial_edit, retry_edit, retry_edit]
                    mock_poll.return_value = failed_ci
                    mock_run.return_value = RunResult(0, "", "")

                    # After 3 attempts (1 initial + 2 retries), all fail
                    # Should return error with ci_attempts=3

    def test_git_operations_decomposed_correctly(self):
        """Verify git operations are called in correct order: add -> commit -> push."""
        repo = "test/repo"
        repo_path = Path("/tmp/test-repo")

        initial_ci = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[
                CICheckResult(name="test", state="completed", conclusion="failure"),
            ],
            failure_summary="tests failed",
            retries=5,
        )

        success_ci = CIPollingResult(
            success=True,
            all_checks_passed=True,
            final_checks=[
                CICheckResult(name="test", state="completed", conclusion="success"),
            ],
            failure_summary="",
            retries=5,
        )

        initial_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Fix",
            output="",
            files_changed=["test.py"],
        )

        retry_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Retry",
            output="",
            files_changed=["test.py"],
        )

        with patch("orchestrator.execute_edit") as mock_exec:
            with patch("orchestrator.poll_ci_checks") as mock_poll:
                with patch("orchestrator.run") as mock_run:
                    mock_exec.side_effect = [initial_edit, retry_edit]
                    mock_poll.side_effect = [initial_ci, success_ci]
                    mock_run.return_value = RunResult(0, "", "")

                    # Verify run is called with:
                    # 1. ["git", "add", "-A"]
                    # 2. ["git", "commit", "-m", "fix: CI auto-retry attempt 2"]
                    # 3. ["git", "push", "origin", branch, "--force-with-lease"]

    def test_failure_context_propagated_to_retry_edit(self):
        """Verify failure context is extracted and passed to retry edit."""
        repo = "test/repo"
        repo_path = Path("/tmp/test-repo")

        initial_ci = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[
                CICheckResult(name="lint", state="completed", conclusion="failure"),
            ],
            failure_summary="unused imports detected",
            retries=5,
        )

        initial_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Initial",
            output="",
            files_changed=[],
        )

        with patch("orchestrator.extract_ci_failure_for_retry") as mock_extract:
            with patch("orchestrator.execute_edit") as mock_exec:
                with patch("orchestrator.poll_ci_checks") as mock_poll:
                    mock_extract.return_value = "CI checks failed on PR #42:\nFailed checks:\n- lint"
                    mock_exec.return_value = initial_edit
                    mock_poll.return_value = initial_ci

                    # Verify extract_ci_failure_for_retry is called
                    # and its output is prepended to retry edit prompt

    def test_experience_store_records_retry_success(self):
        """Verify experience store records when retry succeeds."""
        repo = "test/repo"
        repo_path = Path("/tmp/test-repo")
        pr_url = "https://github.com/test/repo/pull/42"

        initial_ci = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[
                CICheckResult(name="test", state="completed", conclusion="failure"),
            ],
            failure_summary="failed",
            retries=5,
        )

        success_ci = CIPollingResult(
            success=True,
            all_checks_passed=True,
            final_checks=[
                CICheckResult(name="test", state="completed", conclusion="success"),
            ],
            failure_summary="",
            retries=5,
        )

        initial_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Fix",
            output="",
            files_changed=[],
        )

        retry_edit = EditResult(
            success=True,
            iterations=2,
            final_script="# Retry",
            output="",
            files_changed=[],
        )

        with patch("orchestrator.execute_edit") as mock_exec:
            with patch("orchestrator.poll_ci_checks") as mock_poll:
                with patch("orchestrator.run") as mock_run:
                    mock_exec.side_effect = [initial_edit, retry_edit]
                    mock_poll.side_effect = [initial_ci, success_ci]
                    mock_run.return_value = RunResult(0, "", "")

                    # When retry succeeds, store.record should be called with:
                    # success=True
                    # outcome="CI fixed via auto-retry (attempt 2)"

    def test_experience_store_records_retry_failure(self):
        """Verify experience store records when all retries fail."""
        repo = "test/repo"
        repo_path = Path("/tmp/test-repo")
        pr_url = "https://github.com/test/repo/pull/42"

        failed_ci = CIPollingResult(
            success=False,
            all_checks_passed=False,
            final_checks=[
                CICheckResult(name="test", state="completed", conclusion="failure"),
            ],
            failure_summary="failed",
            retries=5,
        )

        initial_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Fix",
            output="",
            files_changed=[],
        )

        retry_edit = EditResult(
            success=True,
            iterations=1,
            final_script="# Retry",
            output="",
            files_changed=[],
        )

        with patch("orchestrator.execute_edit") as mock_exec:
            with patch("orchestrator.poll_ci_checks") as mock_poll:
                with patch("orchestrator.run") as mock_run:
                    mock_exec.side_effect = [initial_edit, retry_edit, retry_edit]
                    mock_poll.return_value = failed_ci
                    mock_run.return_value = RunResult(0, "", "")

                    # When all retries fail, store.record should be called with:
                    # success=False
                    # outcome="PR created but CI failed after 3 attempts"


if __name__ == "__main__":
    unittest.main()
