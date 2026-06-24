from pathlib import Path

import pytest

from harness.messages import ToolCall, ToolDefinition, ToolResult
from tools.base import ToolContext
from tools.registry import ToolRegistry


class DummyTool:
    name = "dummy"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description="A dummy tool", parameters={})

    def run(self, arguments, context) -> ToolResult:
        return ToolResult(tool_call_id="", name=self.name, content="dummy result", ok=True)


class ExplodingTool:
    name = "explode"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description="Always raises", parameters={})

    def run(self, arguments, context) -> ToolResult:
        raise RuntimeError("intentional failure")


def _ctx():
    return ToolContext(session_id="test", workspace_root=Path("/tmp"))


def test_registered_tool_definition_is_available():
    reg = ToolRegistry()
    reg.register(DummyTool())
    defs = reg.definitions()
    assert len(defs) == 1
    assert defs[0].name == "dummy"


def test_running_registered_tool_calls_correct_tool():
    reg = ToolRegistry()
    reg.register(DummyTool())
    result = reg.run(ToolCall(id="c1", name="dummy", arguments={}), _ctx())
    assert result.ok
    assert result.content == "dummy result"


def test_unknown_tool_returns_failed_result():
    reg = ToolRegistry()
    result = reg.run(ToolCall(id="c1", name="ghost", arguments={}), _ctx())
    assert not result.ok
    assert "ghost" in (result.error or "") or "ghost" in result.content


def test_duplicate_tool_name_raises():
    reg = ToolRegistry()
    reg.register(DummyTool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(DummyTool())


def test_tool_exception_returns_failed_result():
    reg = ToolRegistry()
    reg.register(ExplodingTool())
    result = reg.run(ToolCall(id="c1", name="explode", arguments={}), _ctx())
    assert not result.ok
    assert "intentional failure" in (result.error or "")


def test_tool_call_id_preserved_in_result():
    reg = ToolRegistry()
    reg.register(DummyTool())
    result = reg.run(ToolCall(id="my-call-id", name="dummy", arguments={}), _ctx())
    assert result.tool_call_id == "my-call-id"
