from typing import Any

from harness.action import Action
from harness.models import ActionResult, Context


class TerminalAction(Action):
    name = "terminal"
    kind = "effect"
    description = "The local terminal: prints text to the user, reads user input."
    methods = {
        "print": Action.MethodDescription(
            name="print",
            description="Print text to the user's terminal.",
            parameters_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to print"}},
                "required": ["text"],
            },
        ),
        "read": Action.MethodDescription(
            name="read",
            description="Read one line of input from the user's terminal.",
        ),
    }

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        if method_name == "print":
            text = arguments.get("text")
            if not isinstance(text, str):
                return ActionResult(contents="", error="'text' must be a string")
            print(text)
            return ActionResult(contents=text)
        if method_name == "read":
            return ActionResult(contents=input())
        return ActionResult(contents="", error=f"Unknown method: {method_name!r}")
