
from hyh.action import DELIVERY_METHODS, INPUT_METHOD, SEND_METHOD, THINKING_METHOD, TYPING_METHOD
from hyh.action_directory import ActionDirectory
from hyh.context import ContextBuilder
from hyh.logger import new_id, record
from hyh.models import ActionDescription, ActionResult, Context, Operand

QUIT_WORDS = {"quit", "exit", "q"}
EMPTY_INPUT_REPLY = "I didn't get that."


class PolicyController:
    """All routing business logic. Inspects the resolved frontier — the
    operands whose results feed the next decision — and decides what runs
    next by creating child operands (parents = the frontier, order = position
    in the batch). Also owns the end-of-turn listening decision: which
    channels keep a pending input request, and where the uncles move
    (mechanics executed by ContextBuilder).
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

    def decide(self, frontier: list[Operand], context: Context) -> list[Operand]:
        """The children to execute next, in order."""
        proposals = [
            ad
            for operand in frontier
            if operand.action_result is not None
            for ad in operand.action_result.action_description_requests
        ]
        if proposals:
            children = [
                self.context_builder.add_operand(parents=frontier, order=i, action_request=ad)
                for i, ad in enumerate(proposals)
            ]
            for child in children:
                record(child, f"policy decision=attach_proposals count={len(proposals)}")
            return children

        stimulus = self._resolved_input(frontier)
        if stimulus is not None:
            channel, result = stimulus
            text = result.contents.strip()
            if text.lower() in QUIT_WORDS:
                child = self._null(frontier, reason=f"channel_quit channel={channel}")
                self._move_uncles(context, frontier)
                return [child]
            if not text:
                # The channel heard something it couldn't turn into text
                # (voice note, blank line). Say so, honestly, on the same
                # channel; the normal turn flow then re-arms listening.
                return [self._delivery(frontier, channel, EMPTY_INPUT_REPLY, result.metadata)]
            return [self._thinking(frontier, context, reason="stimulus_received")]

        thinking = self._thinking_result(frontier)
        if thinking is not None:
            origin = self._origin_stimulus(context)
            if origin is None:
                return [self._null(frontier, reason="no_origin_channel")]
            if thinking.contents:
                channel, stimulus_result = origin
                return [
                    self._delivery(frontier, channel, thinking.contents, stimulus_result.metadata)
                ]
            child = self._input(frontier, origin[0], reason="empty_thinking_result")
            self._move_uncles(context, frontier)
            return [child]

        if any(op.action_result is not None and op.action_result.error for op in frontier):
            # A failed request (e.g. an invalid parameter the model can fix)
            # must reach the model — even when every request was a delivery,
            # which would otherwise end the turn silently.
            return [self._thinking(frontier, context, reason="error_results_pending")]

        if frontier and self._all_delivery(frontier):
            origin = self._origin_stimulus(context)
            if origin is None:
                return [self._null(frontier, reason="no_origin_channel")]
            child = self._input(frontier, origin[0], reason="turn_complete")
            self._move_uncles(context, frontier)
            return [child]

        # Effect results (tool output) need interpretation.
        return [self._thinking(frontier, context, reason="effect_results_pending")]

    @staticmethod
    def _resolved_input(frontier: list[Operand]) -> tuple[str, ActionResult] | None:
        """(channel, result) of the frontier's resolved input request, if any."""
        for operand in frontier:
            request, result = operand.action_request, operand.action_result
            if request is not None and result is not None and request.method_name == INPUT_METHOD:
                return request.action_name, result
        return None

    @staticmethod
    def _origin_stimulus(context: Context) -> tuple[str, ActionResult] | None:
        """(channel, result) of the stimulus that started the current branch:
        the most recent resolved input result on the ancestor closure."""
        for operand in reversed(context.history):
            request, result = operand.action_request, operand.action_result
            if request is not None and result is not None and request.method_name == INPUT_METHOD:
                return request.action_name, result
        return None

    def _thinking_result(self, frontier: list[Operand]) -> ActionResult | None:
        result = None
        for operand in frontier:
            request = operand.action_request
            if request is not None and self.action_directory.get(request.action_name).kind == "thinking":
                result = operand.action_result
        return result

    def _all_delivery(self, frontier: list[Operand]) -> bool:
        return all(
            op.action_request is not None and op.action_request.method_name in DELIVERY_METHODS
            for op in frontier
        )

    def _move_uncles(self, context: Context, new_parents: list[Operand]) -> None:
        """Bring the other channels' pending input requests to the current tip
        so their next stimulus continues from the full conversation."""
        for uncle in context.listeners:
            self.context_builder.move(uncle, new_parents)
            record(uncle, "policy decision=keep_listening")

    def _thinking(self, frontier: list[Operand], context: Context, reason: str) -> Operand:
        child = self.context_builder.add_operand(
            parents=frontier,
            order=0,
            action_request=ActionDescription(
                id=new_id(),
                action_name=self.thinking_action_name,
                method_name=THINKING_METHOD,
            ),
        )
        record(child, f"policy decision=attach_thinking reason={reason}")
        self._indicate_typing(frontier, context)
        return child

    def _indicate_typing(self, frontier: list[Operand], context: Context) -> None:
        """Fire the origin channel's typing indicator, recorded as an
        already-resolved sibling of the thinking operand. Off the ancestor
        closure, so the model never sees it — and fired only here, after real
        thinking was attached, so the indicator never lies about canned
        replies. Channels without a typing method opt out by omission."""
        origin = self._origin_stimulus(context)
        if origin is None:
            return
        channel, stimulus = origin
        action = self.action_directory.get(channel)
        method = action.methods.get(TYPING_METHOD)
        if method is None:
            return
        properties = method.parameters_schema.get("properties", {})
        parameters = {k: v for k, v in stimulus.metadata.items() if k in properties}
        request = ActionDescription(
            id=new_id(), action_name=channel, method_name=TYPING_METHOD,
            method_parameters=parameters,
        )
        result = action.run(TYPING_METHOD, parameters, context)
        sibling = self.context_builder.add_operand(
            parents=frontier, order=1, action_request=request, action_result=result
        )
        record(sibling, f"policy decision=typing_indicator channel={channel}")

    def _null(self, frontier: list[Operand], reason: str) -> Operand:
        child = self.context_builder.add_operand(
            parents=frontier,
            order=0,
            action_request=ActionDescription(
                id=new_id(), action_name="null", method_name="terminate"
            ),
        )
        record(child, f"policy decision=attach_null reason={reason}")
        return child

    def _input(self, frontier: list[Operand], channel: str, reason: str) -> Operand:
        child = self.context_builder.add_operand(
            parents=frontier,
            order=0,
            action_request=ActionDescription(
                id=new_id(), action_name=channel, method_name=INPUT_METHOD
            ),
        )
        record(child, f"policy decision=attach_input channel={channel} reason={reason}")
        return child

    def _delivery(
        self, frontier: list[Operand], channel: str, text: str, metadata: dict | None = None
    ) -> Operand:
        schema_properties = (
            self.action_directory.get(channel)
            .methods[SEND_METHOD]
            .parameters_schema.get("properties", {})
        )
        # Addressing comes from the origin stimulus: metadata keys the send
        # method declares in its schema (e.g. telegram chat_id) pass through.
        parameters = {k: v for k, v in (metadata or {}).items() if k in schema_properties}
        parameters["text"] = text
        child = self.context_builder.add_operand(
            parents=frontier,
            order=0,
            action_request=ActionDescription(
                id=new_id(),
                action_name=channel,
                method_name=SEND_METHOD,
                method_parameters=parameters,
            ),
        )
        record(child, f"policy decision=deliver_response channel={channel}")
        return child
