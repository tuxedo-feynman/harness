from datetime import datetime, timezone

from harness.events import SessionEvent
from harness.messages import AssistantTurn, Message, ToolCall, ToolResult
from storage.jsonl_store import JsonlSessionStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    def __init__(self, store: JsonlSessionStore):
        self.store = store

    def get_messages(self, session_id: str) -> list[Message]:
        events = self.store.load_events(session_id)
        messages: list[Message] = []
        for event in events:
            if event.event_type == "user_message":
                messages.append(Message(role="user", content=event.payload["content"]))
            elif event.event_type == "assistant_turn":
                tool_calls_raw = event.payload.get("tool_calls", [])
                tool_calls = [ToolCall(**tc) for tc in tool_calls_raw]
                messages.append(Message(
                    role="assistant",
                    content=event.payload.get("content"),
                    tool_calls=tool_calls if tool_calls else None,
                ))
            elif event.event_type == "tool_result":
                messages.append(Message(
                    role="tool",
                    content=event.payload["content"],
                    tool_call_id=event.payload["tool_call_id"],
                    name=event.payload.get("name"),
                ))
            # unknown event types are silently skipped
        return messages

    def record_user_message(self, session_id: str, content: str) -> None:
        self.store.append_event(SessionEvent(
            session_id=session_id,
            event_type="user_message",
            payload={"content": content},
            created_at=_now(),
        ))

    def record_assistant_turn(self, session_id: str, turn: AssistantTurn) -> None:
        # content and tool_calls stored together so the turn is atomic
        self.store.append_event(SessionEvent(
            session_id=session_id,
            event_type="assistant_turn",
            payload={
                "content": turn.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in turn.tool_calls
                ],
            },
            created_at=_now(),
        ))

    def record_tool_result(self, session_id: str, result: ToolResult) -> None:
        self.store.append_event(SessionEvent(
            session_id=session_id,
            event_type="tool_result",
            payload={
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "content": result.content,
                "ok": result.ok,
                "error": result.error,
            },
            created_at=_now(),
        ))

    def record_error(self, session_id: str, error: str) -> None:
        self.store.append_event(SessionEvent(
            session_id=session_id,
            event_type="error",
            payload={"error": error},
            created_at=_now(),
        ))
