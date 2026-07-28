from typing import Any

from harness.action import LISTEN_METHOD, SEND_METHOD, THINKING_METHOD, Action
from harness.config import ClaudeActionConfig
from harness.models import ActionDescription, ActionResult, Context


class ClaudeThinkingAction(Action):
    """Anthropic Claude via the Messages API. Owns all wire-format translation
    in both directions: Context -> system/messages/tools, response ->
    ActionResult. Credentials come from config.yaml (api_key) or, when unset,
    the environment (ANTHROPIC_API_KEY or an `ant auth login` profile).
    """

    name = "claude"
    kind = "thinking"
    description = "Anthropic Claude Messages API."
    methods = {
        THINKING_METHOD: Action.MethodDescription(
            name=THINKING_METHOD,
            description="Send the current context to Claude, return its decision.",
        )
    }

    def __init__(self, config: ClaudeActionConfig):
        self.config = config

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        if method_name != THINKING_METHOD:
            raise ValueError(f"Unknown method: {method_name!r}")

        tools, lookup = self._build_tools(context)
        messages = self._to_claude_messages(context)

        import anthropic

        # api_key=None falls back to the SDK's environment resolution
        client = anthropic.Anthropic(api_key=self.config.api_key)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
            # Safety classifiers can decline a request; the server-side
            # fallback re-runs it on Anthropic's recommended model instead.
            "betas": ["server-side-fallback-2026-07-01"],
            "extra_body": {"fallbacks": "default"},
        }
        if context.system_prompt:
            kwargs["system"] = context.system_prompt
        if tools:
            kwargs["tools"] = tools
        response = client.beta.messages.create(**kwargs)
        return self._parse_response(response, lookup)

    def _build_tools(self, context: Context) -> tuple[list[dict], dict[str, tuple[str, str]]]:
        """Flat-named tool list for the API plus the reverse lookup table.
        Only effect actions are advertised as callable."""
        tools: list[dict] = []
        lookup: dict[str, tuple[str, str]] = {}
        for action_name, available in context.available_actions.items():
            if available.kind != "effect":
                continue
            for method in available.methods.values():
                flat_name = f"{action_name}_{method.name}"
                lookup[flat_name] = (action_name, method.name)
                tools.append(
                    {
                        "name": flat_name,
                        "description": method.description,
                        "input_schema": method.parameters_schema
                        or {"type": "object", "properties": {}},
                    }
                )
        return tools, lookup

    def _to_claude_messages(self, context: Context) -> list[dict]:
        """The system prompt travels separately (the `system` parameter);
        messages carry only the conversation."""
        messages: list[dict] = []
        proposal_ids: set[str] = set()
        last_assistant_text = ""
        for operand in context.history:
            for request, result in zip(operand.action_requests, operand.action_results):
                kind = self._kind_of(context, request.action_name)
                if request.method_name == LISTEN_METHOD:
                    messages.append(
                        {"role": "user", "content": result.contents or self._render_empty(result)}
                    )
                elif kind == "thinking":
                    # Thinking blocks must be echoed back unchanged on
                    # multi-turn tool use; they ride in result metadata.
                    blocks: list[dict] = list(result.metadata.get("thinking_blocks", []))
                    if result.contents:
                        blocks.append({"type": "text", "text": result.contents})
                    for ad in result.action_description_requests:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": ad.id,
                                "name": f"{ad.action_name}_{ad.method_name}",
                                "input": ad.method_parameters,
                            }
                        )
                        proposal_ids.add(ad.id)
                    if blocks:
                        messages.append({"role": "assistant", "content": blocks})
                        last_assistant_text = result.contents
                elif kind == "effect" and request.id in proposal_ids:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": request.id,
                                    "content": result.contents,
                                }
                            ],
                        }
                    )
                elif request.method_name == SEND_METHOD and result.contents != last_assistant_text:
                    # A policy-authored send (e.g. a canned reply) is something
                    # the user heard from "the assistant" — the model must see
                    # it too. Deliveries relaying the thinking result's own
                    # text are the utterance already emitted, so they're skipped.
                    messages.append({"role": "assistant", "content": result.contents})
                    last_assistant_text = result.contents
                # Anything else (null, delivery relays) isn't model-facing.
        return messages

    def _parse_response(self, response: Any, lookup: dict[str, tuple[str, str]]) -> ActionResult:
        # Check the stop reason before reading content: safety classifiers can
        # decline with a normal HTTP 200 and an empty/partial content array.
        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"Claude declined the request: {getattr(response, 'stop_details', None)}"
            )

        contents = ""
        proposals: list[ActionDescription] = []
        thinking_blocks: list[dict] = []
        for block in response.content:
            if block.type == "text":
                contents += block.text
            elif block.type in ("thinking", "redacted_thinking"):
                thinking_blocks.append(block.model_dump())
            elif block.type == "tool_use":
                pair = lookup.get(block.name)
                if pair is None:
                    raise ValueError(f"Model requested unknown tool: {block.name!r}")
                proposals.append(
                    ActionDescription(
                        id=block.id,
                        action_name=pair[0],
                        method_name=pair[1],
                        method_parameters=dict(block.input),
                    )
                )
        return ActionResult(
            contents=contents,
            metadata={"thinking_blocks": thinking_blocks} if thinking_blocks else {},
            action_description_requests=proposals,
        )

    @staticmethod
    def _render_empty(result: ActionResult) -> str:
        """A contentless stimulus, described from the recorded event so the
        model knows what actually happened (voice note, photo, blank line)."""
        message = result.metadata.get("message")
        if isinstance(message, dict):
            kinds = [
                k
                for k in ("voice", "audio", "photo", "video", "video_note",
                          "document", "sticker", "location", "contact", "poll")
                if k in message
            ]
            if kinds:
                return (
                    f"[the user sent a message with no text — it contains: "
                    f"{', '.join(kinds)} — this content type isn't supported]"
                )
        return "[the user sent an empty message]"

    @staticmethod
    def _kind_of(context: Context, action_name: str) -> str | None:
        available = context.available_actions.get(action_name)
        return available.kind if available else None
