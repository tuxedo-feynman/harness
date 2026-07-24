from typing import Any

from harness.action import Action
from harness.logger import new_id
from harness.models import ActionDescription, ActionResult, Context


class FakeThinkingAction(Action):
    """Pre-programmed thinking action for testing. Returns queued results in
    order. Default fallback behaviour (no queued results):
    - First user input in history: proposes printing a greeting.
    - Later user inputs: proposes printing a fake echo of the latest input.
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
            if request.action_name == "terminal" and request.method_name == "read"
        ]
        if len(user_inputs) <= 1:
            text = "Hello! How can I help you?"
        else:
            text = f"[fake] {user_inputs[-1]}"
        return ActionResult(
            contents=text,
            action_description_requests=[
                ActionDescription(
                    id=new_id(),
                    action_name="terminal",
                    method_name="print",
                    method_parameters={"text": text},
                )
            ],
        )
