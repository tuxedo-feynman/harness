from typing import Any

from harness.action import Action
from harness.models import ActionDescription, ActionResult, Context


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
        "complete": Action.MethodDescription(
            name="complete",
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
            result.contents
            for operand in context.history
            for request, result in zip(operand.action_requests, operand.action_results)
            if self._is_listen(context, request) and result.contents.strip()
        ]
        if len(user_inputs) <= 1:
            return ActionResult(contents="Hello! How can I help you?")
        return ActionResult(contents=f"[fake] {user_inputs[-1]}")

    @staticmethod
    def _is_listen(context: Context, request: ActionDescription) -> bool:
        available = context.available_actions.get(request.action_name)
        if available is None:
            return False
        method = available.methods.get(request.method_name)
        return bool(method and method.listen)
