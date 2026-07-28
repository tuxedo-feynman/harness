import logging
import time

from harness.action import LISTEN_METHOD
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.models import Operand
from harness.policy import PolicyController

log = logging.getLogger(__name__)


class ExecutionLoop:
    """Single-threaded execution cycle, anchored on a resolved stimulus:
    Build Context --> Evaluate Policy --> Execute Actions --> repeat.
    A turn ends when a listen request parks (returned to listening) or a
    Null Action executes (branch terminated). All requests in a batch run
    sequentially; action_results[i] is the result of action_requests[i].
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
        current = anchor
        while True:
            working = self.context_builder.add_operand(parent=current)
            context = self.context_builder.build(working)
            working = self.policy.evaluate(working, context)

            i = len(working.action_results)
            while i < len(working.action_requests):
                request = working.action_requests[i]
                action = self.action_directory.get(request.action_name)

                if request.method_name == LISTEN_METHOD and LISTEN_METHOD in action.methods:
                    dropped = len(working.action_requests) - (i + 1)
                    if dropped:
                        log.warning(
                            f"requests_dropped operand={working.id} count={dropped}"
                            f" reason=after_listen_request"
                        )
                        del working.action_requests[i + 1:]
                    self.context_builder.park_listener(request.action_name, working)
                    log.info(
                        f"listener_parked operand={working.id} channel={request.action_name}"
                    )
                    return

                start = time.monotonic()
                result = action.run(request.method_name, request.method_parameters, context)
                working.action_results.append(result)

                duration = time.monotonic() - start
                log.info(
                    f"action_executed operand={working.id} request={request.id}"
                    f" action={request.action_name} method={request.method_name}"
                    f" proposals={len(result.action_description_requests)}"
                    f" duration={duration:.3f}s"
                )
                if action.kind == "null":
                    return
                i += 1

            current = working
