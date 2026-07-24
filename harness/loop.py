import logging
import time

from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.models import ActionResult
from harness.policy import PolicyController

log = logging.getLogger(__name__)


class ExecutionLoop:
    """Single-threaded execution cycle:
    Build Context --> Evaluate Policy --> Execute Actions --> repeat,
    until a Null Action is executed. All requests in an operand's batch are
    executed sequentially; action_results[i] is the result of action_requests[i].
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

    def run(self) -> None:
        leaf = self.context_builder.leaf()
        if leaf is None:
            raise RuntimeError("Seed an operand via ContextBuilder before run()")

        while True:
            if leaf.resolved:
                leaf = self.context_builder.add_operand()
            context = self.context_builder.build()
            leaf = self.policy.evaluate(leaf, context)

            executed_null = False
            for request in leaf.action_requests[len(leaf.action_results):]:
                start = time.monotonic()
                try:
                    action = self.action_directory.get(request.action_name)
                    result = action.run(request.method_name, request.method_parameters, context)
                    kind = action.kind
                except Exception as e:
                    result = ActionResult(contents="", error=str(e))
                    kind = None
                leaf.action_results.append(result)

                duration = time.monotonic() - start
                log.info(
                    f"action_executed operand={leaf.id} request={request.id}"
                    f" action={request.action_name} method={request.method_name}"
                    f" ok={result.error is None} proposals={len(result.action_description_requests)}"
                    f" duration={duration:.3f}s"
                )
                if result.error:
                    log.warning(f"action_error operand={leaf.id} request={request.id} error={result.error!r}")
                if kind == "null":
                    executed_null = True

            if executed_null:
                return
