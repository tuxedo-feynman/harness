"""Action base class: the single abstraction for tools, AI APIs, and null."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.models import ActionKind, ActionResult, Context

# Canonical method names — the harness's verb vocabulary. The name IS the
# marker; there are no flags. Channel actions expose "listen" (wait for the
# world; parks as a pending listener operand) and "send" (tell the world).
# Thinking actions expose "complete". Each action translates the verb into its
# own underlying call (input(), getUpdates, chat.completions.create, ...).
LISTEN_METHOD = "listen"
SEND_METHOD = "send"
THINKING_METHOD = "complete"


class Action(ABC):
    """Base class for everything the harness can execute: terminal, telegram,
    ChatGPT, null, etc. Subclasses set the class attributes and implement run().
    """

    @dataclass
    class MethodDescription:
        name: str
        description: str
        parameters_schema: dict[str, Any] = field(default_factory=dict)

    name: str
    kind: ActionKind
    description: str
    methods: dict[str, MethodDescription]

    @abstractmethod
    def run(
        self, method_name: str, arguments: dict[str, Any], context: Context
    ) -> ActionResult: ...
