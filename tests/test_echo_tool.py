from pathlib import Path

from tools.base import ToolContext
from tools.echo_tool import EchoTool


def _ctx():
    return ToolContext(session_id="test", workspace_root=Path("/tmp"))


def test_returns_input_text():
    tool = EchoTool()
    result = tool.run({"text": "hello world"}, _ctx())
    assert result.ok
    assert result.content == "hello world"


def test_missing_text_returns_failed_result():
    tool = EchoTool()
    result = tool.run({}, _ctx())
    assert not result.ok
    assert result.error is not None


def test_non_string_text_returns_failed_result():
    tool = EchoTool()
    result = tool.run({"text": 42}, _ctx())
    assert not result.ok
    assert "str" in (result.error or "").lower()


def test_definition_has_name_description_and_schema():
    tool = EchoTool()
    defn = tool.definition()
    assert defn.name == "echo"
    assert defn.description
    assert "properties" in defn.parameters
    assert "text" in defn.parameters["properties"]
    assert "required" in defn.parameters
