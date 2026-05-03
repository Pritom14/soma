"""tests/test_explore_repo.py

Tests for explore_repo() and related functions.
Covers: clone, README parsing, linting config detection, file sampling,
convention extraction, and belief candidate generation.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from orchestrator import (
    explore_repo,
    ExplorationResult,
    BeliefCandidate,
    _clone_repo_temp,
    _extract_readme_insights,
    _parse_linting_config,
    _sample_source_files,
    _extract_conventions,
    _generate_belief_candidates,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_repo() -> Path:
    """Create a temporary test repository with various files."""
    tmp = Path(tempfile.mkdtemp(prefix="test_explore_repo_"))

    # Create basic structure
    (tmp / "README.md").write_text(
        """# Test Project

This is a Python project using TypeScript frontend.

## Contributing

Please follow PEP 8. Submit PRs to develop branch.
We use pytest for testing.
"""
    )

    (tmp / "CONTRIBUTING.md").write_text(
        """# Contributing Guidelines

1. Fork and clone the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit PR with description
"""
    )

    # Create .eslintrc
    (tmp / ".eslintrc.json").write_text(
        json.dumps(
            {
                "extends": ["eslint:recommended"],
                "parser": "@typescript-eslint/parser",
                "rules": {
                    "semi": [2, "always"],
                    "quotes": [2, "double"],
                    "indent": [2, 2],
                },
            }
        )
    )

    # Create .prettierrc
    (tmp / ".prettierrc").write_text(
        json.dumps(
            {
                "semi": False,
                "singleQuote": True,
                "trailingComma": "es5",
                "tabWidth": 2,
            }
        )
    )

    # Create .ruff.toml
    (tmp / ".ruff.toml").write_text(
        """
[tool.ruff]
line-length = 100
select = ["E", "F", "W"]
ignore = ["E501"]
"""
    )

    # Create Python source files
    (tmp / "src").mkdir(parents=True, exist_ok=True)

    (tmp / "src" / "main.py").write_text(
        '''"""Main entry point."""
import sys
from utils import helper_func

def main():
    """Execute the application."""
    try:
        result = helper_func()
        print(result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''
    )

    (tmp / "src" / "utils.py").write_text(
        '''"""Utility functions."""

def helper_func():
    """Helper function."""
    return "success"
'''
    )

    # Create test files
    (tmp / "tests").mkdir(parents=True, exist_ok=True)

    (tmp / "tests" / "test_utils.py").write_text(
        '''"""Tests for utils module."""
import pytest
from src.utils import helper_func

def test_helper_func():
    """Test helper function."""
    assert helper_func() == "success"
'''
    )

    # Create TypeScript files
    (tmp / "src" / "index.ts").write_text(
        """import { helper } from './utils';

function main(): void {
  try {
    const result = helper();
    console.log(result);
  } catch (e) {
    console.error(e);
  }
}

main();
"""
    )

    (tmp / "src" / "utils.ts").write_text(
        """export function helper(): string {
  return 'success';
}
"""
    )

    yield tmp

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# Tests: Clone and Cleanup
# =============================================================================


class TestCloneAndCleanup:
    def test_clone_invalid_repo(self):
        """Clone should fail gracefully for non-existent repo."""
        repo_path, error = _clone_repo_temp("invalid/nonexistent-repo-12345")
        assert repo_path is None
        assert error is not None
        assert "Clone failed" in error or "not found" in error.lower()

    def test_clone_returns_path_structure(self, temp_repo):
        """Temp directory should exist after clone (mocked via direct usage)."""
        # Note: We can't actually clone without gh CLI, so we test the
        # directory structure expectations instead
        assert temp_repo.exists()
        assert (temp_repo / "README.md").exists()


# =============================================================================
# Tests: README Parsing
# =============================================================================


class TestREADMEParsing:
    def test_extract_tech_stack_python(self, temp_repo):
        """Should detect Python from README."""
        readme = (temp_repo / "README.md").read_text()
        findings = _extract_readme_insights(readme)
        assert "python" in findings["tech_stack"]

    def test_extract_tech_stack_typescript(self, temp_repo):
        """Should detect TypeScript from README."""
        readme = (temp_repo / "README.md").read_text()
        findings = _extract_readme_insights(readme)
        assert "typescript" in findings["tech_stack"]

    def test_extract_contribution_mentions(self, temp_repo):
        """Should extract contribution guidelines."""
        readme = (temp_repo / "README.md").read_text()
        findings = _extract_readme_insights(readme)
        assert len(findings["contribution_mentions"]) > 0
        assert any("PEP 8" in m or "pytest" in m for m in findings["contribution_mentions"])

    def test_empty_readme(self):
        """Should handle empty README gracefully."""
        findings = _extract_readme_insights("")
        assert findings["tech_stack"] == []
        assert findings["contribution_mentions"] == []

    def test_readme_with_docker(self):
        """Should detect Docker in README."""
        readme = "# Project\nUse Docker to build: `docker build .`"
        findings = _extract_readme_insights(readme)
        assert "docker" in findings["tech_stack"]


# =============================================================================
# Tests: Linting Config Parsing
# =============================================================================


class TestLintingConfigParsing:
    def test_parse_eslintrc(self, temp_repo):
        """Should parse .eslintrc.json correctly."""
        content = (temp_repo / ".eslintrc.json").read_text()
        config = _parse_linting_config(content, "eslintrc")
        assert "extends" in config
        assert "eslint:recommended" in config["extends"]
        assert "parser" in config
        assert "@typescript-eslint/parser" in config["parser"]

    def test_parse_prettierrc(self, temp_repo):
        """Should parse .prettierrc correctly."""
        content = (temp_repo / ".prettierrc").read_text()
        config = _parse_linting_config(content, "prettierrc")
        assert config["semi"] is False
        assert config["single_quote"] is True
        assert config["tab_width"] == 2

    def test_parse_ruff_config(self, temp_repo):
        """Should parse .ruff.toml correctly."""
        content = (temp_repo / ".ruff.toml").read_text()
        config = _parse_linting_config(content, "ruff")
        assert config.get("line_length") == 100
        assert "E" in config.get("select", [])

    def test_parse_invalid_json_eslintrc(self):
        """Should handle invalid JSON gracefully."""
        content = "{ invalid json }"
        config = _parse_linting_config(content, "eslintrc")
        assert "parse_error" in config


# =============================================================================
# Tests: File Sampling
# =============================================================================


class TestFileSampling:
    def test_sample_python_files(self, temp_repo):
        """Should sample Python source files."""
        sampled = _sample_source_files(temp_repo, "python", limit=5)
        assert len(sampled) > 0
        assert any("main.py" in f or "utils.py" in f for f in sampled.keys())

    def test_sample_typescript_files(self, temp_repo):
        """Should sample TypeScript source files."""
        sampled = _sample_source_files(temp_repo, "typescript", limit=5)
        assert len(sampled) > 0
        assert any("index.ts" in f or "utils.ts" in f for f in sampled.keys())

    def test_sample_respects_limit(self, temp_repo):
        """Should not exceed limit."""
        sampled = _sample_source_files(temp_repo, "python", limit=3)
        assert len(sampled) <= 3

    def test_sample_includes_entry_points(self, temp_repo):
        """Should prioritize entry points like main.py."""
        sampled = _sample_source_files(temp_repo, "python", limit=5)
        # At least one should be main.py or similar
        assert any("main.py" in f for f in sampled.keys())

    def test_sample_includes_tests(self, temp_repo):
        """Should include test files in sampling."""
        sampled = _sample_source_files(temp_repo, "python", limit=5)
        assert any("test_" in f for f in sampled.keys())


# =============================================================================
# Tests: Convention Extraction
# =============================================================================


class TestConventionExtraction:
    def test_extract_import_style(self, temp_repo):
        """Should detect explicit import style."""
        main_py = (temp_repo / "src" / "main.py").read_text()
        sampled = {"src/main.py": main_py}
        conventions = _extract_conventions(sampled)
        assert "from X import Y (explicit imports)" in conventions["import_style"]

    def test_extract_docstring_style(self, temp_repo):
        """Should detect triple-quoted docstring style."""
        main_py = (temp_repo / "src" / "main.py").read_text()
        sampled = {"src/main.py": main_py}
        conventions = _extract_conventions(sampled)
        assert "triple-quoted docstrings" in conventions["docstring_style"]

    def test_extract_naming_conventions(self, temp_repo):
        """Should detect naming convention (snake_case)."""
        main_py = (temp_repo / "src" / "main.py").read_text()
        sampled = {"src/main.py": main_py}
        conventions = _extract_conventions(sampled)
        assert any("snake_case" in c for c in conventions["naming_conventions"])

    def test_extract_error_handling(self, temp_repo):
        """Should detect error handling patterns."""
        main_py = (temp_repo / "src" / "main.py").read_text()
        sampled = {"src/main.py": main_py}
        conventions = _extract_conventions(sampled)
        assert any("try-except" in c for c in conventions["error_handling"])

    def test_extract_comment_style(self, temp_repo):
        """Should detect comment style."""
        # Create a file with explicit single-line comments
        python_with_comments = '''# This is a comment
def foo():
    # Another comment
    pass
'''
        sampled = {"test.py": python_with_comments}
        conventions = _extract_conventions(sampled)
        assert any("single-line comments" in c for c in conventions["comment_style"])


# =============================================================================
# Tests: Belief Candidate Generation
# =============================================================================


class TestBeliefCandidateGeneration:
    def test_generate_from_docstring_style(self):
        """Should generate belief candidate for docstring style."""
        exploration = ExplorationResult(
            repo_url="test/repo",
            success=True,
            conventions={
                "docstring_style": ["triple-quoted docstrings"],
                "import_style": [],
                "naming_conventions": [],
                "error_handling": [],
                "comment_style": [],
            },
        )
        candidates = _generate_belief_candidates(exploration)
        assert any(
            "triple-quoted docstrings" in c.statement for c in candidates
        )

    def test_generate_from_naming_conventions(self):
        """Should generate belief candidate for naming convention."""
        exploration = ExplorationResult(
            repo_url="test/repo",
            success=True,
            conventions={
                "naming_conventions": ["snake_case preferred"],
                "import_style": [],
                "docstring_style": [],
                "error_handling": [],
                "comment_style": [],
            },
        )
        candidates = _generate_belief_candidates(exploration)
        assert any("snake_case" in c.statement for c in candidates)

    def test_generate_from_linting_config(self):
        """Should generate belief candidate from linting config."""
        exploration = ExplorationResult(
            repo_url="test/repo",
            success=True,
            conventions={
                "import_style": [],
                "docstring_style": [],
                "naming_conventions": [],
                "error_handling": [],
                "comment_style": [],
            },
            linting_configs={
                ".eslintrc": {
                    "extends": ["eslint:recommended"],
                    "rules": ["semi", "quotes"],
                }
            },
        )
        candidates = _generate_belief_candidates(exploration)
        assert any("ESLint" in c.statement for c in candidates)

    def test_generate_from_prettier_config(self):
        """Should generate belief candidate from prettier config."""
        exploration = ExplorationResult(
            repo_url="test/repo",
            success=True,
            conventions={
                "import_style": [],
                "docstring_style": [],
                "naming_conventions": [],
                "error_handling": [],
                "comment_style": [],
            },
            linting_configs={
                ".prettierrc": {
                    "semi": False,
                }
            },
        )
        candidates = _generate_belief_candidates(exploration)
        assert any("semicolons=false" in c.statement for c in candidates)

    def test_confidence_level_extraction(self):
        """Generated candidates should have reasonable confidence."""
        exploration = ExplorationResult(
            repo_url="test/repo",
            success=True,
            conventions={
                "naming_conventions": ["snake_case preferred"],
                "import_style": [],
                "docstring_style": [],
                "error_handling": [],
                "comment_style": [],
            },
        )
        candidates = _generate_belief_candidates(exploration)
        assert len(candidates) > 0
        assert all(0.5 < c.confidence <= 0.8 for c in candidates)


# =============================================================================
# Tests: Integration
# =============================================================================


class TestExploreRepoIntegration:
    def test_exploration_result_dataclass(self):
        """ExplorationResult should have proper structure."""
        result = ExplorationResult(
            repo_url="test/repo",
            success=True,
            readme_content="# Test",
            belief_candidates=[],
        )
        assert result.repo_url == "test/repo"
        assert result.success is True
        assert result.readme_content == "# Test"
        assert isinstance(result.belief_candidates, list)

    def test_belief_candidate_dataclass(self):
        """BeliefCandidate should have proper defaults."""
        candidate = BeliefCandidate(
            statement="test statement",
        )
        assert candidate.statement == "test statement"
        assert candidate.domain == "oss_contribution"
        assert candidate.confidence == 0.55

    def test_belief_candidate_custom_values(self):
        """BeliefCandidate should accept custom values."""
        candidate = BeliefCandidate(
            statement="test",
            domain="custom_domain",
            confidence=0.75,
        )
        assert candidate.domain == "custom_domain"
        assert candidate.confidence == 0.75

    def test_full_exploration_workflow(self, temp_repo):
        """Full exploration should complete successfully with sample repo."""
        # Note: This can't test actual repo cloning without gh CLI,
        # but we verify the component functions work together

        readme = (temp_repo / "README.md").read_text()
        readme_insights = _extract_readme_insights(readme)

        sampled = _sample_source_files(temp_repo, "python", limit=5)
        conventions = _extract_conventions(sampled)

        config = (temp_repo / ".eslintrc.json").read_text()
        linting = _parse_linting_config(config, "eslintrc")

        # Build exploration result
        exploration = ExplorationResult(
            repo_url="test/repo",
            success=True,
            readme_content=readme,
            conventions=conventions,
            linting_configs={".eslintrc": linting},
        )

        # Generate beliefs
        candidates = _generate_belief_candidates(exploration)

        # Verify output
        assert len(candidates) > 0
        assert all(isinstance(c, BeliefCandidate) for c in candidates)
        assert all(c.domain == "oss_contribution" for c in candidates)
        assert all(0.5 <= c.confidence <= 0.85 for c in candidates)


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_exploration_with_missing_readme(self):
        """Should handle missing README gracefully."""
        tmp = Path(tempfile.mkdtemp(prefix="test_no_readme_"))
        try:
            sampled = _sample_source_files(tmp, "python", limit=1)
            assert isinstance(sampled, dict)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_parse_config_with_parse_error(self):
        """Should handle config parsing errors gracefully."""
        bad_json = "{ not valid json"
        result = _parse_linting_config(bad_json, "eslintrc")
        assert "parse_error" in result

    def test_sample_empty_directory(self):
        """Should handle empty directory gracefully."""
        tmp = Path(tempfile.mkdtemp(prefix="test_empty_"))
        try:
            sampled = _sample_source_files(tmp, "python", limit=5)
            assert isinstance(sampled, dict)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_belief_candidate_confidence_bounds(self):
        """BeliefCandidate confidence should be reasonable."""
        # Test lower bound
        candidate_low = BeliefCandidate(statement="test", confidence=0.0)
        assert candidate_low.confidence == 0.0

        # Test upper bound
        candidate_high = BeliefCandidate(statement="test", confidence=1.0)
        assert candidate_high.confidence == 1.0
