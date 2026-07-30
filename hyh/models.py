"""Core internal data models. See docs/architecture.txt."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from hyh.action import Action

ActionKind = Literal["thinking", "effect", "null"]


@dataclass
class ActionDescription:
    """Everything needed to call an Action. id is a str so it can carry
    provider-issued call ids (e.g. OpenAI's "call_abc123")."""

    id: str
    action_name: str
    method_name: str
    method_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    contents: str
    # Structured event data from the world (e.g. the telegram chat_id/username
    # a message came from). Requests record intent; results record events.
    metadata: dict[str, Any] = field(default_factory=dict)
    action_description_requests: list[ActionDescription] = field(default_factory=list)
    error: str | None = None


@dataclass
class Operand:
    """One node of the operand DAG: a single request for an event, resolved
    when that event arrives (a thinking request by a completion, a send by
    its delivery, an input request by the channel's next event). parents are
    exactly the operands whose results this request consumes; order is this
    operand's position among its siblings — context rendering is topological
    order with sibling-order tie-breaks, so it is deliberately sibling-local
    (a global sequence would conflate with id and overflow a years-long
    agent life)."""

    id: str
    created_at: datetime
    parents: list[str]
    order: int
    action_request: ActionDescription | None = None
    action_result: ActionResult | None = None

    @property
    def resolved(self) -> bool:
        # The root awaits nothing; every other operand is a pending promise
        # until its event arrives.
        return self.action_request is None or self.action_result is not None


@dataclass
class AvailableAction:
    """Read-only projection of one registered Action: its kind and methods,
    nothing callable. This is what Thinking Actions see."""

    kind: ActionKind
    methods: dict[str, "Action.MethodDescription"]


@dataclass
class Context:
    system_prompt: str
    history: list[Operand]
    available_actions: dict[str, AvailableAction]
    # Live pending listener operands. They are off the ancestor path, so
    # Policy can only see them here.
    listeners: list[Operand] = field(default_factory=list)
