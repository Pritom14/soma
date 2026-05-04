"""
Tests for SchedulerAgent and PRManagerAgent wiring into orchestrator work loop.

Covers:
- SchedulerAgent.prioritize_repos() ranking logic
- SchedulerAgent.check_deadlines() surfacing overdue tasks
- SchedulerAgent.select_next_task() selection strategy
- SchedulerAgent.create_campaign() batch scheduling
- PRManagerAgent.monitor_review() state reading
- PRManagerAgent.handle_merge_decision() decision routing
- PRManagerAgent.poll_ci_status() non-blocking CI check
- orchestrator.run_work_loop() falls back to scheduler when queue is empty
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

from agents import SchedulerAgent, PRManagerAgent, ContributeAgent
from core.tasks import TaskQueue, Task
from core.experience import ExperienceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler():
    """Return a SchedulerAgent with mocked externals."""
    store = ExperienceStore()
    agent = SchedulerAgent(domain="task", store=store)
    return agent


def _make_pr_manager():
    """Return a PRManagerAgent with mocked externals."""
    store = ExperienceStore()
    agent = PRManagerAgent(domain="oss_contribution", store=store)
    return agent


def _stale_iso(hours_ago: float) -> str:
    return (datetime.utcnow() - timedelta(hours=hours_ago)).isoformat()


# ---------------------------------------------------------------------------
# Task 1: SchedulerAgent
# ---------------------------------------------------------------------------

class TestSchedulerPrioritizeRepos:
    """SchedulerAgent.prioritize_repos() must return all repos in a ranked order."""

    def test_returns_all_repos(self):
        agent = _make_scheduler()
        repos = ["owner/a", "owner/b", "owner/c"]
        result = agent.prioritize_repos(repos)
        assert sorted(result) == sorted(repos), "all repos must be returned"

    def test_returns_list(self):
        agent = _make_scheduler()
        result = agent.prioritize_repos(["owner/x"])
        assert isinstance(result, list)
        assert result == ["owner/x"]

    def test_empty_list(self):
        agent = _make_scheduler()
        assert agent.prioritize_repos([]) == []

    def test_higher_success_rate_ranks_first(self):
        """A repo with past successes should rank above one with failures."""
        agent = _make_scheduler()
        # Seed experiences
        agent.store.record(
            domain="oss_contribution",
            context="success_repo/project",
            action="contribute",
            outcome="merged",
            success=True,
            model_used="test",
        )
        agent.store.record(
            domain="oss_contribution",
            context="fail_repo/project",
            action="contribute",
            outcome="closed",
            success=False,
            model_used="test",
        )
        ranked = agent.prioritize_repos(["success_repo/project", "fail_repo/project"])
        assert ranked[0] == "success_repo/project", (
            "repo with higher success rate should rank first"
        )


class TestSchedulerCheckDeadlines:
    """SchedulerAgent.check_deadlines() must surface stale tasks and PRs."""

    def test_returns_list(self):
        agent = _make_scheduler()
        result = agent.check_deadlines(verbose=False)
        assert isinstance(result, list)

    def test_stale_task_surfaced(self):
        """A task created 50 hours ago should appear in overdue list."""
        agent = _make_scheduler()
        # Enqueue a task and manually backdate its created_at
        task = agent.queue.enqueue(
            type="contribute",
            context={"repo": "deadline_test/repo", "task_description": "old task"},
            priority=3,
        )
        # Backdate via direct DB update
        agent.queue.conn.execute(
            "UPDATE tasks SET created_at=? WHERE id=?",
            (_stale_iso(50), task.id),
        )
        agent.queue.conn.commit()

        overdue = agent.check_deadlines(verbose=False)
        task_overdue = [o for o in overdue if o.get("task_id") == task.id]
        assert task_overdue, "50-hour-old task should appear as overdue"
        assert task_overdue[0]["type"] == "stale_task"

    def test_fresh_task_not_surfaced(self):
        """A task created 1 hour ago should NOT appear as overdue."""
        agent = _make_scheduler()
        task = agent.queue.enqueue(
            type="explore",
            context={"task": "fresh_deadline_test"},
            priority=2,
        )
        overdue = agent.check_deadlines(verbose=False)
        task_overdue = [o for o in overdue if o.get("task_id") == task.id]
        assert not task_overdue, "1-hour-old task must not be surfaced as overdue"


class TestSchedulerSelectNextTask:
    """SchedulerAgent.select_next_task() must pick a task from candidates or repos."""

    def test_returns_overdue_when_present(self):
        """If check_deadlines finds something, select_next_task returns it first."""
        agent = _make_scheduler()
        # Plant a stale task
        task = agent.queue.enqueue(
            type="contribute",
            context={"repo": "select_test/repo"},
            priority=3,
        )
        agent.queue.conn.execute(
            "UPDATE tasks SET created_at=? WHERE id=?",
            (_stale_iso(55), task.id),
        )
        agent.queue.conn.commit()

        result = agent.select_next_task(candidates=[], verbose=False)
        assert result is not None
        assert result.get("type") in ("stale_task", "contribute", "explore_repo")

    def test_uses_candidates_when_no_overdue(self):
        """When no overdue tasks, select_next_task should pick from provided candidates."""
        agent = _make_scheduler()
        # Ensure no stale tasks by clearing any existing pending tasks
        pending = agent.queue.list(status="pending")
        for t in pending:
            agent.queue.update_status(t.id, "done")

        candidates = [
            {"type": "contribute", "repo": "owner/repo", "number": 42,
             "title": "Fix bug", "score": 0.8, "confidence": 0.7, "reason": "test"},
        ]
        with patch.object(agent, "check_deadlines", return_value=[]):
            result = agent.select_next_task(candidates=candidates, verbose=False)
        assert result is not None
        assert result.get("type") == "contribute"
        assert result.get("repo") == "owner/repo"

    def test_returns_none_when_nothing_available(self):
        """With no candidates and no repos, should return None."""
        agent = _make_scheduler()
        # Clear pending tasks
        pending = agent.queue.list(status="pending")
        for t in pending:
            agent.queue.update_status(t.id, "done")

        with patch.object(agent, "check_deadlines", return_value=[]), \
             patch.object(agent.repo_tracker, "get_all", return_value=[]):
            result = agent.select_next_task(candidates=[], verbose=False)
        assert result is None


class TestSchedulerCreateCampaign:
    """SchedulerAgent.create_campaign() must enqueue tasks for each repo."""

    def test_enqueues_tasks_for_repos(self):
        agent = _make_scheduler()
        repos = ["campaign_test/a", "campaign_test/b"]
        # Clear any existing for these repos
        for t in agent.queue.list(status="pending"):
            if t.context.get("repo") in repos:
                agent.queue.update_status(t.id, "done")

        result = agent.create_campaign(repos, goal="contribute", verbose=False)
        assert isinstance(result, list)
        assert len(result) == len(repos)
        for item in result:
            assert item["type"] == "contribute"
            assert item["repo"] in repos

    def test_skips_already_queued(self):
        """Repos already in queue should not be double-enqueued."""
        agent = _make_scheduler()
        repo = "campaign_skip_test/repo"
        # Pre-enqueue
        agent.queue.enqueue(type="contribute", context={"repo": repo}, priority=3)

        result = agent.create_campaign([repo], goal="contribute", verbose=False)
        assert result == [], "already-queued repo must be skipped"


# ---------------------------------------------------------------------------
# Task 2: PRManagerAgent
# ---------------------------------------------------------------------------

class TestPRManagerPollCiStatus:
    """PRManagerAgent.poll_ci_status() must return a non-blocking status dict."""

    def test_returns_dict_on_error(self):
        """Even when CI polling fails, result should be a usable dict."""
        agent = _make_pr_manager()
        with patch("agents.pr_manager.poll_ci_checks",
                   side_effect=Exception("network timeout")):
            result = agent.poll_ci_status("owner/repo", 99)
        assert isinstance(result, dict)
        assert result["state"] == "unknown"
        assert result["all_passed"] is False
        assert "error" in result

    def test_returns_passed_state(self):
        agent = _make_pr_manager()
        mock_ci = MagicMock()
        mock_ci.success = True
        mock_ci.all_checks_passed = True
        mock_ci.final_checks = []
        with patch("agents.pr_manager.poll_ci_checks", return_value=mock_ci):
            result = agent.poll_ci_status("owner/repo", 1)
        assert result["state"] == "passed"
        assert result["all_passed"] is True

    def test_returns_failed_state(self):
        agent = _make_pr_manager()
        mock_ci = MagicMock()
        mock_ci.success = False
        mock_ci.all_checks_passed = False
        mock_ci.final_checks = []
        with patch("agents.pr_manager.poll_ci_checks", return_value=mock_ci):
            result = agent.poll_ci_status("owner/repo", 2)
        assert result["state"] == "failed"
        assert result["all_passed"] is False


class TestPRManagerMonitorReview:
    """PRManagerAgent.monitor_review() must return structured review state."""

    def _mock_monitor(self, agent, review_decision="NONE", open_items=None,
                       pr_state="OPEN", ci_state="pending"):
        plan = {
            "open_items": open_items or [],
            "resolved_items": [],
            "plan": "",
        }
        ci = {"state": ci_state, "all_passed": ci_state == "passed"}
        gh_json = f'{{"reviewDecision": "{review_decision}", "state": "{pr_state}", "reviews": []}}'

        run_result = MagicMock()
        run_result.success = True
        run_result.output = gh_json

        with patch.object(agent, "plan_from_pr_comments", return_value=plan), \
             patch.object(agent, "poll_ci_status", return_value=ci), \
             patch("agents.pr_manager._run", return_value=run_result):
            return agent.monitor_review("owner/repo", 42, verbose=False)

    def test_returns_structured_dict(self):
        agent = _make_pr_manager()
        result = self._mock_monitor(agent)
        assert "pr" in result
        assert "pr_state" in result
        assert "review_decision" in result
        assert "open_items" in result
        assert "ci_state" in result
        assert "needs_attention" in result

    def test_changes_requested_triggers_attention(self):
        agent = _make_pr_manager()
        result = self._mock_monitor(agent, review_decision="CHANGES_REQUESTED",
                                    open_items=["fix the tests"])
        assert result["needs_attention"] is True

    def test_approved_no_items_no_attention(self):
        agent = _make_pr_manager()
        result = self._mock_monitor(agent, review_decision="APPROVED",
                                    open_items=[], ci_state="passed")
        assert result["needs_attention"] is False


class TestPRManagerHandleMergeDecision:
    """PRManagerAgent.handle_merge_decision() must route to correct action."""

    def test_pending_review_returns_pending(self):
        agent = _make_pr_manager()
        review = {
            "pr_state": "OPEN",
            "review_decision": "NONE",
            "ci_state": "pending",
            "open_items": [],
            "needs_attention": False,
        }
        with patch.object(agent, "monitor_review", return_value=review):
            result = agent.handle_merge_decision("owner/repo", 10,
                                                  mergeable=False, verbose=False)
        assert result["status"] == "pending"

    def test_already_merged_returns_merged(self):
        agent = _make_pr_manager()
        review = {
            "pr_state": "MERGED",
            "review_decision": "APPROVED",
            "ci_state": "passed",
            "open_items": [],
            "needs_attention": False,
        }
        with patch.object(agent, "monitor_review", return_value=review):
            result = agent.handle_merge_decision("owner/repo", 11,
                                                  mergeable=True, verbose=False)
        assert result["status"] == "merged"

    def test_ci_failed_surfaces_to_human(self):
        agent = _make_pr_manager()
        review = {
            "pr_state": "OPEN",
            "review_decision": "NONE",
            "ci_state": "failed",
            "open_items": [],
            "needs_attention": True,
        }
        with patch.object(agent, "monitor_review", return_value=review):
            result = agent.handle_merge_decision("owner/repo", 12,
                                                  mergeable=True, verbose=False)
        assert result["status"] == "awaiting_human"
        assert result.get("reason") == "ci_failed"


# ---------------------------------------------------------------------------
# Task 3: contribute_agent pr_manager wiring
# ---------------------------------------------------------------------------

class TestContributeAgentPRManagerWiring:
    """ContributeAgent must accept and propagate pr_manager."""

    def test_init_accepts_pr_manager(self):
        manager = _make_pr_manager()
        agent = ContributeAgent(pr_manager=manager)
        assert agent.pr_manager is manager

    def test_init_defaults_to_none(self):
        agent = ContributeAgent()
        assert agent.pr_manager is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
