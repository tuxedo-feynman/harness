"""Core internal data models. See docs/architecture.txt."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from harness.action import Action

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
    action_description_requests: list[ActionDescription] = field(default_factory=list)
    error: str | None = None


@dataclass
class Operand:
    """One iteration of the execution loop: a batch of requests and their
    results. action_results[i] is the result of action_requests[i]."""

    id: str
    created_at: datetime
    parent: str | None
    action_requests: list[ActionDescription] = field(default_factory=list)
    action_results: list[ActionResult] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return len(self.action_results) == len(self.action_requests)


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
