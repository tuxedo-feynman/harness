"""Action base class: the single abstraction for tools, AI APIs, and null."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.models import ActionKind, ActionResult, Context


class Action(ABC):
    """Base class for everything the harness can execute: terminal, telegram,
    ChatGPT, null, etc. Subclasses set the class attributes and implement run().
    """

    @dataclass
    class MethodDescription:
        name: str
        description: str
        parameters_schema: dict[str, Any] = field(default_factory=dict)
        # listen methods park as pending listener operands instead of
        # executing to completion (e.g. terminal.read, telegram.receive)
        listen: bool = False

    name: str
    kind: ActionKind
    description: str
    methods: dict[str, MethodDescription]

    @abstractmethod
    def run(
        self, method_name: str, arguments: dict[str, Any], context: Context
    ) -> ActionResult: ...
