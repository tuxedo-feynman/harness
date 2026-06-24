from unittest.mock import MagicMock, patch
import json
import pytest

from harness.config import ProviderConfig
from harness.messages import AssistantTurn, Message, ToolCall, ToolDefinition


def _make_provider():
    from providers.openai_compat_provider import OpenAICompatProvider
    config = ProviderConfig(type="openai_compat", base_url="http://localhost:8080/v1", model="local")
    with patch("providers.openai_compat_provider.OpenAI"):
        provider = OpenAICompatProvider(config)
    return provider


def _mock_response(content=None, tool_calls=None):
    choice = MagicMock()
    choice.content = content
    choice.tool_calls = tool_calls or []
    response = MagicMock()
    response.choices = [MagicMock(message=choice)]
    return response


# --- message conversion ---

def test_converts_system_message():
    from providers.openai_compat_provider import _to_openai_message
    msg = Message(role="system", content="You are helpful.")
    assert _to_openai_message(msg) == {"role": "system", "content": "You are helpful."}


def test_converts_user_message():
    from providers.openai_compat_provider import _to_openai_message
    msg = Message(role="user", content="hello")
    assert _to_openai_message(msg) == {"role": "user", "content": "hello"}


def test_converts_assistant_message_no_tool_calls():
    from providers.openai_compat_provider import _to_openai_message
    msg = Message(role="assistant", content="hi there")
    assert _to_openai_message(msg) == {"role": "assistant", "content": "hi there"}


def test_converts_assistant_message_with_tool_calls():
    from providers.openai_compat_provider import _to_openai_message
    tc = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    msg = Message(role="assistant", content=None, tool_calls=[tc])
    result = _to_openai_message(msg)
    assert result["role"] == "assistant"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "c1"
    assert result["tool_calls"][0]["type"] == "function"
    assert result["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"text": "hi"}


def test_converts_tool_result_message():
    from providers.openai_compat_provider import _to_openai_message
    msg = Message(role="tool", content="hello", tool_call_id="c1")
    result = _to_openai_message(msg)
    assert result == {"role": "tool", "content": "hello", "tool_call_id": "c1"}


# --- tool definition conversion ---

def test_converts_tool_definition():
    from providers.openai_compat_provider import _to_openai_tool
    defn = ToolDefinition(
        name="echo",
        description="Echoes text",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    result = _to_openai_tool(defn)
    assert result["type"] == "function"
    assert result["function"]["name"] == "echo"
    assert result["function"]["description"] == "Echoes text"
    assert result["function"]["parameters"] == defn.parameters


# --- response conversion ---

def test_converts_final_response_to_assistant_turn():
    provider = _make_provider()
    provider._client.chat.completions.create.return_value = _mock_response(content="Hello!")
    turn = provider.complete([Message(role="user", content="hi")], [])
    assert turn.content == "Hello!"
    assert turn.tool_calls == []


def test_converts_tool_call_response_to_assistant_turn():
    provider = _make_provider()
    mock_tc = MagicMock()
    mock_tc.id = "c1"
    mock_tc.function.name = "echo"
    mock_tc.function.arguments = json.dumps({"text": "hello"})
    provider._client.chat.completions.create.return_value = _mock_response(
        content=None, tool_calls=[mock_tc]
    )
    turn = provider.complete([Message(role="user", content="go")], [])
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "c1"
    assert turn.tool_calls[0].name == "echo"
    assert turn.tool_calls[0].arguments == {"text": "hello"}


def test_does_not_make_real_http_calls():
    provider = _make_provider()
    # _client is a MagicMock — real HTTP would raise, mock won't
    provider._client.chat.completions.create.return_value = _mock_response(content="ok")
    turn = provider.complete([], [])
    assert provider._client.chat.completions.create.called
