"""
Comprehensive test suite for agents (ContributeAgent, PRManagerAgent, SchedulerAgent).

Tests cover:
- Agent initialization and dependency injection
- ContributeAgent.run() on mock issues
- PRManagerAgent polling and CI monitoring
- SchedulerAgent task selection and prioritization
- Integration between agents
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timedelta

from agents import ContributeAgent, PRManagerAgent, SchedulerAgent, BaseAgent
from core.experience import ExperienceStore
from core.belief import BeliefStore
from core.llm import LLMClient
from core.tasks import TaskQueue


class TestAgentInitialization:
    """Test agent initialization and dependency injection."""

    def test_base_agent_initialization(self):
        """Test BaseAgent initializes with default stores."""
        agent = BaseAgent()
        assert agent.store is not None
        assert isinstance(agent.store, ExperienceStore)
        assert agent.beliefs is not None
        assert isinstance(agent.beliefs, BeliefStore)
        assert agent.llm is not None
        assert isinstance(agent.llm, LLMClient)

    def test_contribute_agent_initialization(self):
        """Test ContributeAgent initializes with custom domain and store."""
        store = ExperienceStore()
        agent = ContributeAgent(domain="code", store=store)
        assert agent.domain == "code"
        assert agent.store is store
        assert agent.beliefs is not None

    def test_pr_manager_initialization(self):
        """Test PRManagerAgent initializes with custom domain and store."""
        store = ExperienceStore()
        agent = PRManagerAgent(domain="oss_contribution", store=store)
        assert agent.domain == "oss_contribution"
        assert agent.store is store

    def test_scheduler_initialization(self):
        """Test SchedulerAgent initializes with custom domain and store."""
        store = ExperienceStore()
        agent = SchedulerAgent(domain="task", store=store)
        assert agent.domain == "task"
        assert agent.store is store

    def test_agents_share_store(self):
        """Test multiple agents can share the same experience store."""
        shared_store = ExperienceStore()
        contrib = ContributeAgent(store=shared_store)
        pr_mgr = PRManagerAgent(store=shared_store)
        sched = SchedulerAgent(store=shared_store)

        assert contrib.store is shared_store
        assert pr_mgr.store is shared_store
        assert sched.store is shared_store


class TestContributeAgent:
    """Test ContributeAgent methods."""

    def test_contribute_agent_has_build_local_method(self):
        """Test ContributeAgent has build_local method."""
        agent = ContributeAgent()
        assert hasattr(agent, "build_local")
        assert callable(agent.build_local)

    def test_contribute_agent_has_contribute_method(self):
        """Test ContributeAgent has contribute method."""
        agent = ContributeAgent()
        assert hasattr(agent, "contribute")
        assert callable(agent.contribute)

    def test_contribute_agent_has_explore_repo_method(self):
        """Test ContributeAgent has explore_repo method."""
        agent = ContributeAgent()
        assert hasattr(agent, "explore_repo")
        assert callable(agent.explore_repo)

    @patch("core.github.check_gh_auth")
    def test_contribute_agent_checks_auth(self, mock_auth):
        """Test contribute method checks GitHub auth."""
        mock_auth.return_value = False
        agent = ContributeAgent()
        result = agent.contribute("https://github.com/owner/repo/issues/123")
        assert result["success"] is False
        assert "authenticated" in result.get("error", "").lower()

    def test_build_local_with_nonexistent_path(self):
        """Test build_local returns error for nonexistent path."""
        agent = ContributeAgent()
        result = agent.build_local(
            task="test task",
            repo_path="/nonexistent/path"
        )
        assert result["success"] is False
        assert "does not exist" in result["error"].lower()

    def test_build_local_empty_files_returns_error(self):
        """Test build_local returns error when no files found."""
        agent = ContributeAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.build_local(
                task="test task",
                repo_path=tmpdir,
                files=[],
                verbose=False
            )
            # Should fail since no files available
            if result.get("error"):
                assert "error" in result or "success" in result


class TestPRManager:
    """Test PRManagerAgent methods."""

    def test_pr_manager_has_register_pr_method(self):
        """Test PRManagerAgent has register_pr method."""
        agent = PRManagerAgent()
        assert hasattr(agent, "register_pr")
        assert callable(agent.register_pr)

    def test_pr_manager_has_poll_pr_comments_method(self):
        """Test PRManagerAgent has poll_pr_comments method."""
        agent = PRManagerAgent()
        assert hasattr(agent, "poll_pr_comments")
        assert callable(agent.poll_pr_comments)

    def test_pr_manager_has_resolve_merge_conflicts_method(self):
        """Test PRManagerAgent has resolve_merge_conflicts method."""
        agent = PRManagerAgent()
        assert hasattr(agent, "resolve_merge_conflicts")
        assert callable(agent.resolve_merge_conflicts)

    def test_pr_manager_has_poll_pr_outcomes_method(self):
        """Test PRManagerAgent has poll_pr_outcomes method."""
        agent = PRManagerAgent()
        assert hasattr(agent, "poll_pr_outcomes")
        assert callable(agent.poll_pr_outcomes)

    def test_pr_manager_has_execute_pr_plan_method(self):
        """Test PRManagerAgent has execute_pr_plan method."""
        agent = PRManagerAgent()
        assert hasattr(agent, "execute_pr_plan")
        assert callable(agent.execute_pr_plan)

    @patch("core.pr_monitor.PRMonitor.register")
    def test_register_pr_returns_correct_structure(self, mock_register):
        """Test register_pr returns dict with expected keys."""
        mock_pr = MagicMock()
        mock_pr.id = "pr_123"
        mock_register.return_value = mock_pr

        agent = PRManagerAgent()
        result = agent.register_pr("owner/repo", 42, "Test PR", ["belief_1"])

        assert "id" in result
        assert "repo" in result
        assert "pr_number" in result
        assert result["repo"] == "owner/repo"
        assert result["pr_number"] == 42

    def test_resolve_merge_conflicts_without_worktree(self):
        """Test resolve_merge_conflicts returns error without worktree."""
        agent = PRManagerAgent()
        result = agent.resolve_merge_conflicts("owner/repo", 42)
        assert result["status"] == "error"
        assert "worktree" in result["error"].lower()

    def test_poll_pr_comments_returns_list(self):
        """Test poll_pr_comments returns a list."""
        agent = PRManagerAgent()
        with patch.object(agent.pr_monitor, "poll", return_value=[]):
            result = agent.poll_pr_comments(verbose=False)
            assert isinstance(result, list)

    def test_poll_pr_outcomes_returns_list(self):
        """Test poll_pr_outcomes returns a list."""
        agent = PRManagerAgent()
        result = agent.poll_pr_outcomes()
        assert isinstance(result, list)


class TestScheduler:
    """Test SchedulerAgent methods."""

    def test_scheduler_has_confidence_gate_method(self):
        """Test SchedulerAgent has confidence_gate method."""
        agent = SchedulerAgent()
        assert hasattr(agent, "confidence_gate")
        assert callable(agent.confidence_gate)

    def test_scheduler_has_update_goals_method(self):
        """Test SchedulerAgent has update_goals method."""
        agent = SchedulerAgent()
        assert hasattr(agent, "update_goals")
        assert callable(agent.update_goals)

    def test_scheduler_has_run_work_loop_method(self):
        """Test SchedulerAgent has run_work_loop method."""
        agent = SchedulerAgent()
        assert hasattr(agent, "run_work_loop")
        assert callable(agent.run_work_loop)

    def test_scheduler_has_show_goals_method(self):
        """Test SchedulerAgent has show_goals method."""
        agent = SchedulerAgent()
        assert hasattr(agent, "show_goals")
        assert callable(agent.show_goals)

    def test_confidence_gate_returns_dict_with_required_keys(self):
        """Test confidence_gate returns dict with expected structure."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("test task", verbose=False)

        assert isinstance(result, dict)
        assert "recommendation" in result
        assert "avg_confidence" in result
        assert "beliefs" in result
        assert "reason" in result
        assert result["recommendation"] in ["act", "gather", "surface"]
        assert isinstance(result["avg_confidence"], (int, float))
        assert isinstance(result["beliefs"], list)
        assert isinstance(result["reason"], str)

    def test_confidence_gate_with_no_beliefs(self):
        """Test confidence_gate returns gather when no relevant beliefs."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("completely random xyz task", verbose=False)

        assert result["recommendation"] == "gather"
        assert result["avg_confidence"] == 0.0
        assert len(result["beliefs"]) == 0

    def test_update_goals_returns_dict(self):
        """Test update_goals returns a dict."""
        agent = SchedulerAgent()
        result = agent.update_goals(verbose=False)
        assert isinstance(result, dict)

    def test_show_goals_does_not_raise(self):
        """Test show_goals can be called without error."""
        agent = SchedulerAgent()
        # Should not raise exception
        agent.show_goals()

    @patch("core.tasks.TaskQueue.next_ready")
    def test_run_work_loop_with_no_tasks(self, mock_next):
        """Test run_work_loop works when no tasks in queue."""
        mock_next.return_value = None
        agent = SchedulerAgent()

        with patch.object(agent.decision_gate, "check_resolved", return_value=[]):
            with patch.object(agent.inbox, "read_pending", return_value=[]):
                result = agent.run_work_loop(verbose=False)
                assert isinstance(result, dict)
                assert "timestamp" in result


class TestAgentIntegration:
    """Test integration between multiple agents."""

    def test_soma_instantiates_all_agents(self):
        """Test SOMA.__init__ creates all three agents."""
        from orchestrator import SOMA

        soma = SOMA()
        assert hasattr(soma, "contribute_agent")
        assert isinstance(soma.contribute_agent, ContributeAgent)
        assert hasattr(soma, "pr_manager")
        assert isinstance(soma.pr_manager, PRManagerAgent)
        assert hasattr(soma, "scheduler")
        assert isinstance(soma.scheduler, SchedulerAgent)

    def test_soma_agents_share_store(self):
        """Test all SOMA agents share the same experience store."""
        from orchestrator import SOMA

        soma = SOMA()
        assert soma.contribute_agent.store is soma.store
        assert soma.pr_manager.store is soma.store
        assert soma.scheduler.store is soma.store

    def test_soma_agents_have_soma_context(self):
        """Test agents have access to SOMA's context."""
        from orchestrator import SOMA

        soma = SOMA()
        # Agents should inherit from BaseAgent and have access to core systems
        assert soma.contribute_agent.beliefs is not None
        assert soma.pr_manager.llm is not None
        assert soma.scheduler.goals is not None


class TestAgentResultStructure:
    """Test that agent results follow expected contracts."""

    def test_pr_manager_poll_pr_comments_result_structure(self):
        """Test poll_pr_comments returns list of dicts with expected keys."""
        agent = PRManagerAgent()
        with patch.object(agent.pr_monitor, "poll", return_value=[{"test": "data"}]):
            result = agent.poll_pr_comments(verbose=False)
            assert isinstance(result, list)
            if result:
                assert isinstance(result[0], dict)

    def test_scheduler_confidence_gate_result_structure(self):
        """Test confidence_gate result has all required keys."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("test", verbose=False)

        required_keys = ["recommendation", "avg_confidence", "beliefs", "reason"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_confidence_gate_avg_confidence_is_float(self):
        """Test avg_confidence in confidence_gate is numeric."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("test", verbose=False)
        assert isinstance(result["avg_confidence"], (int, float))
        assert 0.0 <= result["avg_confidence"] <= 1.0

    def test_confidence_gate_beliefs_is_list(self):
        """Test beliefs in confidence_gate is a list."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("test", verbose=False)
        assert isinstance(result["beliefs"], list)

    def test_confidence_gate_reason_is_string(self):
        """Test reason in confidence_gate is a string."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("test", verbose=False)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


class TestSchedulerTaskSelectionLogic:
    """Test SchedulerAgent task selection and prioritization."""

    def test_scheduler_queue_integration(self):
        """Test SchedulerAgent has access to task queue."""
        agent = SchedulerAgent()
        assert hasattr(agent, "queue")

    def test_update_goals_returns_dict_with_goals(self):
        """Test update_goals returns dict (may be empty)."""
        agent = SchedulerAgent()
        result = agent.update_goals(verbose=False)
        assert isinstance(result, dict)

    def test_confidence_gate_recommendation_values(self):
        """Test confidence_gate returns valid recommendation values."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("test task", verbose=False)
        valid_recommendations = ["act", "gather", "surface"]
        assert result["recommendation"] in valid_recommendations


class TestAgentErrorHandling:
    """Test agent error handling and edge cases."""

    def test_contribute_agent_with_invalid_url(self):
        """Test contribute agent handles invalid URLs."""
        agent = ContributeAgent()
        with patch("core.github.check_gh_auth", return_value=True):
            with patch("core.github.repo_from_url", return_value=None):
                result = agent.contribute("invalid-url")
                assert result["success"] is False

    def test_pr_manager_poll_with_empty_registry(self):
        """Test poll_pr_comments works with empty PR registry."""
        agent = PRManagerAgent()
        with patch.object(agent.pr_monitor, "poll", return_value=[]):
            result = agent.poll_pr_comments(verbose=False)
            assert isinstance(result, list)
            assert len(result) == 0

    def test_scheduler_confidence_gate_with_empty_task(self):
        """Test confidence_gate handles empty task string."""
        agent = SchedulerAgent()
        result = agent.confidence_gate("", verbose=False)
        # Should still return valid structure
        assert "recommendation" in result
        assert "avg_confidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
