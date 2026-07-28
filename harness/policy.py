import logging

from harness.action import LISTEN_METHOD, SEND_METHOD, THINKING_METHOD
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.logger import new_id
from harness.models import ActionDescription, ActionResult, Context, Operand

log = logging.getLogger(__name__)

QUIT_WORDS = {"quit", "exit", "q"}
EMPTY_INPUT_REPLY = "I didn't get that."


class PolicyController:
    """All routing business logic. Inspects the working leaf and decides what
    runs next by attaching ActionDescriptions to it. Also owns the end-of-turn
    listening decision: which channels keep a pending listener, and where the
    uncles move (mechanics executed by ContextBuilder).
    """

    def __init__(
        self,
        action_directory: ActionDirectory,
        context_builder: ContextBuilder,
        thinking_action_name: str,
    ):
        self.action_directory = action_directory
        self.context_builder = context_builder
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
            operand.action_requests = proposals
            log.info(
                f"policy operand={operand.id} decision=attach_proposals count={len(proposals)}"
            )
            return operand

        stimulus = self._resolved_listen(prev)
        if stimulus is not None:
            channel, result = stimulus
            text = result.contents.strip()
            if text.lower() in QUIT_WORDS:
                self._attach_null(operand, reason=f"channel_quit channel={channel}")
                self._move_uncles(context, prev)
            elif not text:
                # The channel heard something it couldn't turn into text
                # (voice note, photo, blank line). Say so, honestly, on the
                # same channel; the normal turn flow then re-arms listening.
                self._attach_delivery(operand, channel, EMPTY_INPUT_REPLY, result.metadata)
            else:
                self._attach_thinking(operand, reason="stimulus_received")
            return operand

        thinking_index = self._last_thinking_index(prev)
        if thinking_index is not None:
            text = prev.action_results[thinking_index].contents
            origin = self._origin_stimulus(context)
            if origin is None:
                self._attach_null(operand, reason="no_origin_channel")
            elif text:
                channel, stimulus_result = origin
                self._attach_delivery(operand, channel, text, stimulus_result.metadata)
            else:
                self._attach_listen(operand, origin[0], reason="empty_thinking_result")
                self._move_uncles(context, prev)
            return operand

        if prev.action_requests and self._all_delivery(prev):
            origin = self._origin_stimulus(context)
            if origin is None:
                self._attach_null(operand, reason="no_origin_channel")
            else:
                self._attach_listen(operand, origin[0], reason="turn_complete")
                self._move_uncles(context, prev)
            return operand

        # Effect results (tool output) need interpretation.
        self._attach_thinking(operand, reason="effect_results_pending")
        return operand

    @staticmethod
    def _resolved_listen(prev: Operand) -> tuple[str, ActionResult] | None:
        """(channel, result) of prev's resolved listen request, if any."""
        for request, result in zip(prev.action_requests, prev.action_results):
            if request.method_name == LISTEN_METHOD:
                return request.action_name, result
        return None

    @staticmethod
    def _origin_stimulus(context: Context) -> tuple[str, ActionResult] | None:
        """(channel, result) of the stimulus that started the current branch:
        the most recent resolved listen result on the ancestor path."""
        for operand in reversed(context.history):
            for request, result in zip(operand.action_requests, operand.action_results):
                if request.method_name == LISTEN_METHOD:
                    return request.action_name, result
        return None

    def _last_thinking_index(self, prev: Operand) -> int | None:
        index = None
        for i, req in enumerate(prev.action_requests[: len(prev.action_results)]):
            if self.action_directory.get(req.action_name).kind == "thinking":
                index = i
        return index

    @staticmethod
    def _all_delivery(prev: Operand) -> bool:
        return all(req.method_name == SEND_METHOD for req in prev.action_requests)

    def _move_uncles(self, context: Context, new_parent: Operand) -> None:
        """Bring the other channels' pending listeners to the current tip so
        their next stimulus continues from the full conversation."""
        for uncle in context.listeners:
            self.context_builder.move(uncle, new_parent)
            log.info(f"policy decision=keep_listening operand={uncle.id} moved_to={new_parent.id}")

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

    def _attach_listen(self, operand: Operand, channel: str, reason: str) -> None:
        operand.action_requests = [
            ActionDescription(id=new_id(), action_name=channel, method_name=LISTEN_METHOD)
        ]
        log.info(f"policy operand={operand.id} decision=attach_listen channel={channel} reason={reason}")

    def _attach_delivery(
        self, operand: Operand, channel: str, text: str, metadata: dict | None = None
    ) -> None:
        schema_properties = (
            self.action_directory.get(channel)
            .methods[SEND_METHOD]
            .parameters_schema.get("properties", {})
        )
        # Addressing comes from the origin stimulus: metadata keys the send
        # method declares in its schema (e.g. telegram chat_id) pass through.
        parameters = {k: v for k, v in (metadata or {}).items() if k in schema_properties}
        parameters["text"] = text
        operand.action_requests = [
            ActionDescription(
                id=new_id(),
                action_name=channel,
                method_name=SEND_METHOD,
                method_parameters=parameters,
            )
        ]
        log.info(f"policy operand={operand.id} decision=deliver_response channel={channel}")

    @staticmethod
    def _find(context: Context, operand_id: str | None) -> Operand | None:
        if operand_id is None:
            return None
        return next((op for op in context.history if op.id == operand_id), None)
