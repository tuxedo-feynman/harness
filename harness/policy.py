import logging

from harness.action_directory import ActionDirectory
from harness.logger import new_id
from harness.models import ActionDescription, Context, Operand

log = logging.getLogger(__name__)

THINKING_METHOD = "complete"  # convention: every thinking action exposes this


class PolicyController:
    """All routing business logic. Inspects the leaf operand and decides what
    runs next by attaching ActionDescriptions to it. The execution loop never
    decides anything; thinking actions only propose.
    """

    def __init__(self, action_directory: ActionDirectory, thinking_action_name: str):
        self.action_directory = action_directory
        self.thinking_action_name = thinking_action_name

    def evaluate(self, operand: Operand, context: Context) -> Operand:
        if operand.action_requests:
            log.info(f"policy operand={operand.id} decision=untouched reason=requests_present")
            return operand

        prev = self._find(context, operand.parent)
        if prev is None:
            self._attach_thinking(operand, reason="no_parent")
            return operand

        proposals = [
            ad for result in prev.action_results
            for ad in result.action_description_requests
        ]
        if proposals:
            vetted = [ad for ad in proposals if self._vet(ad, operand)]
            if vetted:
                operand.action_requests = vetted
                log.info(
                    f"policy operand={operand.id} decision=attach_proposals"
                    f" count={len(vetted)} vetoed={len(proposals) - len(vetted)}"
                )
                return operand

        if self._prev_was_thinking(prev):
            result = prev.action_results[self._last_thinking_index(prev)]
            text = f"error: {result.error}" if result.error else result.contents
            if text:
                operand.action_requests = [self._print_request(text)]
                log.info(f"policy operand={operand.id} decision=deliver_response")
            else:
                self._attach_null(operand, reason="empty_thinking_result")
            return operand

        if not prev.action_requests or self._all_output_or_null(prev):
            self._attach_null(operand, reason="nothing_left_to_do")
            return operand

        # Effect results (user input, tool output) need interpretation.
        self._attach_thinking(operand, reason="effect_results_pending")
        return operand

    def _vet(self, ad: ActionDescription, operand: Operand) -> bool:
        try:
            action = self.action_directory.get(ad.action_name)
        except KeyError:
            log.warning(f"policy operand={operand.id} veto={ad.action_name!r} reason=unknown_action")
            return False
        if ad.method_name not in action.methods:
            log.warning(
                f"policy operand={operand.id} veto={ad.action_name}.{ad.method_name} reason=unknown_method"
            )
            return False
        return True

    def _prev_was_thinking(self, prev: Operand) -> bool:
        return self._last_thinking_index(prev) is not None

    def _last_thinking_index(self, prev: Operand) -> int | None:
        index = None
        for i, req in enumerate(prev.action_requests[: len(prev.action_results)]):
            try:
                if self.action_directory.get(req.action_name).kind == "thinking":
                    index = i
            except KeyError:
                continue
        return index

    def _all_output_or_null(self, prev: Operand) -> bool:
        for req in prev.action_requests:
            is_print = req.action_name == "terminal" and req.method_name == "print"
            is_null = req.action_name == "null"
            if not (is_print or is_null):
                return False
        return True

    def _attach_thinking(self, operand: Operand, reason: str) -> None:
        operand.action_requests = [
            ActionDescription(
                id=new_id(),
                action_name=self.thinking_action_name,
                method_name=THINKING_METHOD,
            )
        ]
        log.info(f"policy operand={operand.id} decision=attach_thinking reason={reason}")

    def _attach_null(self, operand: Operand, reason: str) -> None:
        operand.action_requests = [
            ActionDescription(id=new_id(), action_name="null", method_name="terminate")
        ]
        log.info(f"policy operand={operand.id} decision=attach_null reason={reason}")

    def _print_request(self, text: str) -> ActionDescription:
        return ActionDescription(
            id=new_id(),
            action_name="terminal",
            method_name="print",
            method_parameters={"text": text},
        )

    @staticmethod
    def _find(context: Context, operand_id: str | None) -> Operand | None:
        if operand_id is None:
            return None
        return next((op for op in context.history if op.id == operand_id), None)
