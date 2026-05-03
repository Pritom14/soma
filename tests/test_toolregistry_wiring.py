"""
Tests for ToolRegistry wiring into executor.py
"""
import pytest
from pathlib import Path
from core.tool_registry import ToolRegistry, Tool


def test_tool_registry_initialization():
    """Verify ToolRegistry can be instantiated and discovers tools."""
    repo_path = Path("/Users/pritommazumdar/Desktop/soma")
    registry = ToolRegistry(repo_path)
    assert registry is not None
    assert registry.repo == repo_path


def test_tool_registry_discover():
    """Verify ToolRegistry discovers tools from repo."""
    repo_path = Path("/Users/pritommazumdar/Desktop/soma")
    registry = ToolRegistry(repo_path)

    # Should discover at least pytest (pyproject.toml exists)
    tools = registry.available()
    assert len(tools) > 0

    # Tools should be Tool instances with required attributes
    for tool in tools:
        assert hasattr(tool, "name")
        assert hasattr(tool, "command")
        assert hasattr(tool, "description")
        assert hasattr(tool, "timeout")
        assert isinstance(tool.name, str)
        assert isinstance(tool.command, list)
        assert isinstance(tool.description, str)


def test_tool_registry_get_tool():
    """Verify getting a specific tool by name."""
    repo_path = Path("/Users/pritommazumdar/Desktop/soma")
    registry = ToolRegistry(repo_path)

    # Try to get a tool that should exist
    test_tool = registry.get("test")
    if test_tool:
        assert test_tool.name == "test"
        assert isinstance(test_tool.command, list)


def test_tool_registry_to_prompt_context():
    """Verify tools are formatted correctly for prompt context."""
    repo_path = Path("/Users/pritommazumdar/Desktop/soma")
    registry = ToolRegistry(repo_path)

    context = registry.to_prompt_context()
    assert isinstance(context, str)

    if registry._tools:
        # Should contain tool names and descriptions
        assert "Available repo tools" in context
        for tool in registry._tools.values():
            assert tool.name in context


def test_build_system_prompt_with_tools():
    """Verify system prompt is built with tool information."""
    from core.executor import _build_system_prompt

    repo_path = Path("/Users/pritommazumdar/Desktop/soma")
    registry = ToolRegistry(repo_path)

    # Build system prompt with tool registry
    system_prompt = _build_system_prompt(registry)
    assert isinstance(system_prompt, str)
    assert "code editing agent" in system_prompt

    # Should include tool names if tools were discovered
    if registry._tools:
        for tool in registry._tools.values():
            assert tool.name in system_prompt or tool.description in system_prompt


def test_build_system_prompt_without_tools():
    """Verify system prompt works without tool registry."""
    from core.executor import _build_system_prompt

    # Build system prompt without tool registry
    system_prompt = _build_system_prompt(None)
    assert isinstance(system_prompt, str)
    assert "code editing agent" in system_prompt
    assert "Python stdlib" in system_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
