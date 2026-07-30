from typing import Any

from hyh.action import INPUT_METHOD, THINKING_METHOD, Action
from hyh.models import ActionResult, Context


class FakeThinkingAction(Action):
    """Pre-programmed thinking action for testing. Returns queued results in
    order. Default fallback behaviour (no queued results): first user input in
    history gets a greeting, later inputs get a fake echo. Returns contents
    only — Policy routes delivery to the origin channel.
    """

    name = "fake"
    kind = "thinking"
    description = "Fake AI for tests: canned responses, no network."
    methods = {
        THINKING_METHOD: Action.MethodDescription(
            name=THINKING_METHOD,
            description="Return the next canned decision.",
        )
    }

    def __init__(self, responses: list[ActionResult] | None = None):
        self._responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, Any], Context]] = []

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        self.calls.append((method_name, dict(arguments), context))
        if self._responses:
            return self._responses.pop(0)

        user_inputs = [
            operand.action_result.contents
            for operand in context.history
            if operand.action_request is not None
            and operand.action_result is not None
            and operand.action_request.method_name == INPUT_METHOD
            and operand.action_result.contents.strip()
        ]
        if len(user_inputs) <= 1:
            return ActionResult(contents="Hello! How can I help you?")
        return ActionResult(contents=f"[fake] {user_inputs[-1]}")
