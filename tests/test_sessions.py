from datetime import datetime, timezone

import pytest

from harness.events import SessionEvent
from harness.messages import AssistantTurn, ToolCall, ToolResult
from harness.sessions import SessionManager
from storage.jsonl_store import JsonlSessionStore


def _mgr(tmp_path):
    return SessionManager(JsonlSessionStore(data_dir=tmp_path))


def test_user_events_become_user_messages(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.record_user_message("s1", "hello")
    msgs = mgr.get_messages("s1")
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert msgs[0].content == "hello"


def test_assistant_turn_content_only(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.record_assistant_turn("s1", AssistantTurn(content="hi", tool_calls=[]))
    msgs = mgr.get_messages("s1")
    assert len(msgs) == 1
    assert msgs[0].role == "assistant"
    assert msgs[0].content == "hi"
    assert msgs[0].tool_calls is None


def test_assistant_turn_with_tool_calls(tmp_path):
    mgr = _mgr(tmp_path)
    tc = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    mgr.record_assistant_turn("s1", AssistantTurn(content=None, tool_calls=[tc]))
    msgs = mgr.get_messages("s1")
    assert len(msgs) == 1
    assert msgs[0].role == "assistant"
    assert msgs[0].tool_calls is not None
    assert msgs[0].tool_calls[0].name == "echo"
    assert msgs[0].tool_calls[0].id == "c1"


def test_assistant_turn_content_and_tool_calls_stored_atomically(tmp_path):
    mgr = _mgr(tmp_path)
    tc = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    mgr.record_assistant_turn("s1", AssistantTurn(content="thinking...", tool_calls=[tc]))
    msgs = mgr.get_messages("s1")
    assert len(msgs) == 1
    assert msgs[0].content == "thinking..."
    assert msgs[0].tool_calls is not None
    assert len(msgs[0].tool_calls) == 1


def test_tool_result_events_become_tool_messages(tmp_path):
    mgr = _mgr(tmp_path)
    result = ToolResult(tool_call_id="c1", name="echo", content="hello", ok=True)
    mgr.record_tool_result("s1", result)
    msgs = mgr.get_messages("s1")
    assert len(msgs) == 1
    assert msgs[0].role == "tool"
    assert msgs[0].content == "hello"
    assert msgs[0].tool_call_id == "c1"


def test_messages_returned_in_chronological_order(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.record_user_message("s1", "first")
    mgr.record_assistant_turn("s1", AssistantTurn(content="second", tool_calls=[]))
    msgs = mgr.get_messages("s1")
    assert msgs[0].content == "first"
    assert msgs[1].content == "second"


def test_unknown_event_types_are_skipped(tmp_path):
    store = JsonlSessionStore(data_dir=tmp_path)
    store.append_event(SessionEvent(
        session_id="s1",
        event_type="unknown_future_type",  # type: ignore[arg-type]
        payload={"data": "x"},
        created_at=datetime.now(timezone.utc).isoformat(),
    ))
    mgr = SessionManager(store)
    assert mgr.get_messages("s1") == []
