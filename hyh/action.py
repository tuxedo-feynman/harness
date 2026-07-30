"""Action base class: the single abstraction for tools, AI APIs, and null."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hyh.models import ActionKind, ActionResult, Context

# Canonical method names — hyh's verb vocabulary. The name IS the
# marker; there are no flags. Channel actions expose "input" (a request for
# the channel's next event — message, poll vote, ... — parked unresolved
# until the world produces one) and "send" (tell the world). Thinking
# actions expose "complete". Each action translates the verb into its own
# underlying call (input(), getUpdates, chat.completions.create, ...).
INPUT_METHOD = "input"
SEND_METHOD = "send"
THINKING_METHOD = "complete"
TYPING_METHOD = "typing"
REACT_METHOD = "react"
POLL_METHOD = "poll"

# Methods that deliver to the world: a turn whose requests are all
# deliveries is complete, and listening re-arms. typing is deliberately
# not one — it signals work in progress, not a finished turn.
DELIVERY_METHODS = frozenset({SEND_METHOD, REACT_METHOD, POLL_METHOD})


class Action(ABC):
    """Base class for everything hyh can execute: terminal, telegram,
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
