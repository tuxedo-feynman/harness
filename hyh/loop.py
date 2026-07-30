import time

from hyh.action import INPUT_METHOD
from hyh.action_directory import ActionDirectory
from hyh.context import ContextBuilder
from hyh.logger import record
from hyh.models import Operand
from hyh.policy import PolicyController


class ExecutionLoop:
    """Single-threaded execution cycle, anchored on a resolved stimulus:
    Build Context --> Policy decides children --> Execute them in sibling
    order --> the children become the next frontier. A turn ends when an
    input request parks (returned to listening) or a Null Action executes
    (branch terminated).
    """

    def __init__(
        self,
        action_directory: ActionDirectory,
        policy: PolicyController,
        context_builder: ContextBuilder,
    ):
        self.action_directory = action_directory
        self.policy = policy
        self.context_builder = context_builder

    def run(self, anchor: Operand) -> None:
        frontier = [anchor]
        while True:
            context = self.context_builder.build(frontier)
            children = self.policy.decide(frontier, context)
            if not children:
                raise RuntimeError("Policy returned no children — the loop cannot advance")

            for child in children:
                request = child.action_request
                if request is None:
                    raise RuntimeError(f"Policy created a request-less operand: {child.id}")
                action = self.action_directory.get(request.action_name)

                if request.method_name == INPUT_METHOD and INPUT_METHOD in action.methods:
                    self.context_builder.park_listener(request.action_name, child)
                    record(child, f"parked channel={request.action_name}")
                    return

                start = time.monotonic()
                result = action.run(request.method_name, request.method_parameters, context)
                child.action_result = result

                duration = time.monotonic() - start
                record(
                    child,
                    f"executed request={request.id}"
                    f" action={request.action_name} method={request.method_name}"
                    f" proposals={len(result.action_description_requests)}"
                    f" duration={duration:.3f}s",
                )
                if action.kind == "null":
                    return

            frontier = children
