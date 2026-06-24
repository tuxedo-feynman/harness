from pathlib import Path

import pytest

from harness.config import Config, ProviderConfig
from harness.context import ContextBuilder
from harness.loop import HarnessLoop
from harness.messages import AssistantTurn, ToolCall, ToolDefinition, ToolResult
from harness.sessions import SessionManager
from providers.fake_provider import FakeProvider
from storage.jsonl_store import JsonlSessionStore
from tools.base import ToolContext
from tools.registry import ToolRegistry


class FakeTool:
    name = "fake_tool"

    def __init__(self, response: str = "fake result"):
        self.response = response
        self.calls: list[dict] = []

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description="Test tool", parameters={})

    def run(self, arguments, context) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(tool_call_id="", name=self.name, content=self.response, ok=True)


def _make_loop(tmp_path, responses, tools=None):
    store = JsonlSessionStore(data_dir=tmp_path)
    session = SessionManager(store)
    config = Config(
        default_session_id="test",
        system_prompt="You are helpful.",
        provider=ProviderConfig(type="fake"),
        max_tool_rounds=5,
        data_dir=str(tmp_path),
    )
    provider = FakeProvider(responses)
    registry = ToolRegistry()
    for tool in (tools or []):
        registry.register(tool)
    loop = HarnessLoop(
        config=config,
        session=session,
        context_builder=ContextBuilder(),
        provider=provider,
        tool_registry=registry,
        workspace_root=tmp_path,
    )
    return loop, session, provider


def test_simple_user_message_produces_response(tmp_path):
    loop, _, _ = _make_loop(tmp_path, [AssistantTurn(content="Hello!", tool_calls=[])])
    assert loop.run_turn("s1", "hi") == "Hello!"


def test_user_message_is_persisted(tmp_path):
    loop, session, _ = _make_loop(tmp_path, [AssistantTurn(content="ok", tool_calls=[])])
    loop.run_turn("s1", "my message")
    msgs = session.get_messages("s1")
    assert any(m.role == "user" and m.content == "my message" for m in msgs)


def test_assistant_response_is_persisted(tmp_path):
    loop, session, _ = _make_loop(tmp_path, [AssistantTurn(content="my response", tool_calls=[])])
    loop.run_turn("s1", "hi")
    msgs = session.get_messages("s1")
    assert any(m.role == "assistant" and m.content == "my response" for m in msgs)


def test_tool_call_is_executed(tmp_path):
    fake_tool = FakeTool()
    responses = [
        AssistantTurn(content=None, tool_calls=[ToolCall(id="c1", name="fake_tool", arguments={"x": 1})]),
        AssistantTurn(content="done", tool_calls=[]),
    ]
    loop, _, _ = _make_loop(tmp_path, responses, tools=[fake_tool])
    loop.run_turn("s1", "go")
    assert len(fake_tool.calls) == 1


def test_tool_result_added_to_model_context(tmp_path):
    fake_tool = FakeTool("tool output")
    responses = [
        AssistantTurn(content=None, tool_calls=[ToolCall(id="c1", name="fake_tool", arguments={})]),
        AssistantTurn(content="done", tool_calls=[]),
    ]
    loop, _, provider = _make_loop(tmp_path, responses, tools=[fake_tool])
    loop.run_turn("s1", "go")
    second_call_msgs = provider.calls[1][0]
    assert any(m.role == "tool" for m in second_call_msgs)


def test_final_response_after_tool_call_is_returned(tmp_path):
    fake_tool = FakeTool()
    responses = [
        AssistantTurn(content=None, tool_calls=[ToolCall(id="c1", name="fake_tool", arguments={})]),
        AssistantTurn(content="final answer", tool_calls=[]),
    ]
    loop, _, _ = _make_loop(tmp_path, responses, tools=[fake_tool])
    assert loop.run_turn("s1", "go") == "final answer"


def test_assistant_turn_and_tool_result_persisted(tmp_path):
    fake_tool = FakeTool("tool content")
    responses = [
        AssistantTurn(content=None, tool_calls=[ToolCall(id="c1", name="fake_tool", arguments={})]),
        AssistantTurn(content="done", tool_calls=[]),
    ]
    loop, session, _ = _make_loop(tmp_path, responses, tools=[fake_tool])
    loop.run_turn("s1", "go")
    msgs = session.get_messages("s1")
    assert any(m.role == "tool" for m in msgs)
    assert any(m.role == "assistant" for m in msgs)


def test_unknown_tool_produces_failed_result_and_loop_continues(tmp_path):
    responses = [
        AssistantTurn(content=None, tool_calls=[ToolCall(id="c1", name="ghost", arguments={})]),
        AssistantTurn(content="recovered", tool_calls=[]),
    ]
    loop, _, _ = _make_loop(tmp_path, responses)
    assert loop.run_turn("s1", "hi") == "recovered"


def test_tool_exception_does_not_crash_loop(tmp_path):
    class ExplodingTool:
        name = "boom"
        def definition(self): return ToolDefinition(name=self.name, description="", parameters={})
        def run(self, arguments, context): raise RuntimeError("kaboom")

    responses = [
        AssistantTurn(content=None, tool_calls=[ToolCall(id="c1", name="boom", arguments={})]),
        AssistantTurn(content="survived", tool_calls=[]),
    ]
    loop, _, _ = _make_loop(tmp_path, responses, tools=[ExplodingTool()])
    assert loop.run_turn("s1", "go") == "survived"


def test_max_tool_rounds_stops_infinite_loop(tmp_path):
    fake_tool = FakeTool()
    # max_tool_rounds=5: need 6 tool-call responses to hit the limit
    responses = [
        AssistantTurn(content=None, tool_calls=[ToolCall(id=f"c{i}", name="fake_tool", arguments={})])
        for i in range(6)
    ]
    loop, _, _ = _make_loop(tmp_path, responses, tools=[fake_tool])
    result = loop.run_turn("s1", "go")
    assert "max" in result.lower() or "exceeded" in result.lower()


def test_context_includes_prior_session_history(tmp_path):
    loop1, _, _ = _make_loop(tmp_path, [AssistantTurn(content="first response", tool_calls=[])])
    loop1.run_turn("s1", "first message")

    loop2, _, provider2 = _make_loop(tmp_path, [AssistantTurn(content="second response", tool_calls=[])])
    loop2.run_turn("s1", "second message")

    sent_messages = provider2.calls[0][0]
    assert any(m.content == "first message" for m in sent_messages)


def test_tool_definitions_passed_to_provider(tmp_path):
    fake_tool = FakeTool()
    loop, _, provider = _make_loop(tmp_path, [AssistantTurn(content="ok", tool_calls=[])], tools=[fake_tool])
    loop.run_turn("s1", "hi")
    tool_defs = provider.calls[0][1]
    assert any(d.name == "fake_tool" for d in tool_defs)


def test_no_real_provider_needed(tmp_path):
    # All loop tests use FakeProvider — this test makes that explicit
    loop, _, provider = _make_loop(tmp_path, [AssistantTurn(content="ok", tool_calls=[])])
    loop.run_turn("s1", "hi")
    assert len(provider.calls) == 1
    assert provider.calls[0][0][-1].role == "user"
