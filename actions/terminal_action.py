from typing import Any

from hyh.action import INPUT_METHOD, SEND_METHOD, Action
from hyh.models import ActionResult, Context


class TerminalAction(Action):
    name = "terminal"
    kind = "effect"
    description = "The local terminal: sends text to the user, listens for their input."
    methods = {
        SEND_METHOD: Action.MethodDescription(
            name=SEND_METHOD,
            description="Print text to the user's terminal.",
            parameters_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to print"}},
                "required": ["text"],
            },
        ),
        INPUT_METHOD: Action.MethodDescription(
            name=INPUT_METHOD,
            description="The user's next line of terminal input.",
        ),
    }

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        if method_name == SEND_METHOD:
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("'text' must be a string")
            print(text, flush=True)
            return ActionResult(contents=text)
        if method_name == INPUT_METHOD:
            return ActionResult(contents=input("> "))
        raise ValueError(f"Unknown method: {method_name!r}")
