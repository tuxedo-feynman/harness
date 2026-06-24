from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class SessionEvent:
    session_id: str
    event_type: Literal["user_message", "assistant_turn", "tool_result", "error"]
    payload: dict[str, Any]
    created_at: str
