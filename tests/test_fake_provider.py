from harness.messages import AssistantTurn, Message, ToolCall, ToolDefinition
from providers.fake_provider import FakeProvider


def test_returns_programmed_responses_in_order():
    provider = FakeProvider([
        AssistantTurn(content="first", tool_calls=[]),
        AssistantTurn(content="second", tool_calls=[]),
    ])
    assert provider.complete([], []).content == "first"
    assert provider.complete([], []).content == "second"


def test_greeting_on_first_call_when_no_tools():
    provider = FakeProvider()
    turn = provider.complete([Message(role="user", content="hi")], [])
    assert turn.content == "Hello! How can I help you?"
    assert turn.tool_calls == []


def test_calls_echo_tool_on_first_call_when_echo_available():
    provider = FakeProvider()
    echo_def = ToolDefinition(name="echo", description="Echoes text", parameters={})
    turn = provider.complete([Message(role="user", content="hello")], [echo_def])
    assert turn.content is None
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "echo"
    assert turn.tool_calls[0].arguments == {"text": "hello"}


def test_greeting_returned_after_tool_result():
    provider = FakeProvider()
    echo_def = ToolDefinition(name="echo", description="Echoes text", parameters={})
    provider.complete([Message(role="user", content="hi")], [echo_def])
    msgs = [
        Message(role="user", content="hi"),
        Message(role="tool", content="hi", tool_call_id="fake-1"),
    ]
    turn = provider.complete(msgs, [echo_def])
    assert turn.content == "Hello! How can I help you?"
    assert turn.tool_calls == []


def test_echoes_user_message_on_subsequent_calls():
    provider = FakeProvider()
    provider.complete([Message(role="user", content="first")], [])
    turn = provider.complete([Message(role="user", content="second")], [])
    assert turn.content == "[fake] second"


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
