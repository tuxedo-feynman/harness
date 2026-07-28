from typing import Any

from hyh.action import Action
from hyh.models import ActionResult, Context


class NullAction(Action):
    name = "null"
    kind = "null"
    description = "Terminates the execution loop. Requested when there is nothing left to do."
    methods = {
        "terminate": Action.MethodDescription(
            name="terminate",
            description="End the current execution loop.",
        )
    }

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        return ActionResult(contents="")
