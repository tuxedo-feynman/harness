import pytest

from harness.messages import AssistantTurn, Message, ToolDefinition
from providers.fake_provider import FakeProvider


def test_returns_programmed_responses_in_order():
    provider = FakeProvider([
        AssistantTurn(content="first", tool_calls=[]),
        AssistantTurn(content="second", tool_calls=[]),
    ])
    assert provider.complete([], []).content == "first"
    assert provider.complete([], []).content == "second"


def test_raises_when_no_responses_remain():
    provider = FakeProvider([])
    with pytest.raises(RuntimeError, match="no more"):
        provider.complete([], [])


def test_records_messages_passed_to_it():
    provider = FakeProvider([AssistantTurn(content="ok", tool_calls=[])])
    msgs = [Message(role="user", content="hello")]
    provider.complete(msgs, [])
    assert provider.calls[0][0] == msgs


def test_records_tool_definitions_passed_to_it():
    provider = FakeProvider([AssistantTurn(content="ok", tool_calls=[])])
    tools = [ToolDefinition(name="echo", description="Echoes text", parameters={})]
    provider.complete([], tools)
    assert provider.calls[0][1] == tools
