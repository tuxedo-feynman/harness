import json
import os
from typing import Any

from harness.action import LISTEN_METHOD, THINKING_METHOD, Action
from harness.config import ThinkingActionConfig
from harness.models import ActionDescription, ActionResult, Context


class OpenAIThinkingAction(Action):
    """OpenAI-compatible chat completion API. Works with llama-server or the
    real OpenAI API. Owns all wire-format translation in both directions:
    Context -> messages/tools, response -> ActionResult.
    """

    name = "openai"
    kind = "thinking"
    description = "OpenAI-compatible chat completion API."
    methods = {
        THINKING_METHOD: Action.MethodDescription(
            name=THINKING_METHOD,
            description="Send the current context to the model, return its decision.",
        )
    }

    def __init__(self, config: ThinkingActionConfig):
        self.config = config

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        if method_name != THINKING_METHOD:
            raise ValueError(f"Unknown method: {method_name!r}")

        tools, lookup = self._build_tools(context)
        messages = self._to_openai_messages(context)

        from openai import OpenAI

        client = OpenAI(
            base_url=self.config.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        )
        kwargs: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = client.chat.completions.create(**kwargs)
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
                        "type": "function",
                        "function": {
                            "name": flat_name,
                            "description": method.description,
                            "parameters": method.parameters_schema
                            or {"type": "object", "properties": {}},
                        },
                    }
                )
        return tools, lookup

    def _to_openai_messages(self, context: Context) -> list[dict]:
        messages: list[dict] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})

        proposal_ids: set[str] = set()
        for operand in context.history:
            for request, result in zip(operand.action_requests, operand.action_results):
                kind = self._kind_of(context, request.action_name)
                if request.method_name == LISTEN_METHOD:
                    messages.append({"role": "user", "content": result.contents})
                elif kind == "thinking":
                    message: dict[str, Any] = {
                        "role": "assistant",
                        "content": result.contents or None,
                    }
                    if result.action_description_requests:
                        message["tool_calls"] = [
                            {
                                "id": ad.id,
                                "type": "function",
                                "function": {
                                    "name": f"{ad.action_name}_{ad.method_name}",
                                    "arguments": json.dumps(ad.method_parameters),
                                },
                            }
                            for ad in result.action_description_requests
                        ]
                        proposal_ids.update(ad.id for ad in result.action_description_requests)
                    messages.append(message)
                elif kind == "effect" and request.id in proposal_ids:
                    messages.append(
                        {"role": "tool", "tool_call_id": request.id, "content": result.contents}
                    )
                # Anything else (null, policy-synthesized prints) isn't part of
                # the model-facing conversation.
        return messages

    def _parse_response(self, response: Any, lookup: dict[str, tuple[str, str]]) -> ActionResult:
        message = response.choices[0].message
        proposals: list[ActionDescription] = []
        for tool_call in message.tool_calls or []:
            pair = lookup.get(tool_call.function.name)
            if pair is None:
                raise ValueError(f"Model requested unknown tool: {tool_call.function.name!r}")
            try:
                parameters = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad tool arguments for {tool_call.function.name!r}: {e}") from e
            proposals.append(
                ActionDescription(
                    id=tool_call.id,
                    action_name=pair[0],
                    method_name=pair[1],
                    method_parameters=parameters,
                )
            )
        return ActionResult(
            contents=message.content or "",
            action_description_requests=proposals,
        )

    @staticmethod
    def _kind_of(context: Context, action_name: str) -> str | None:
        available = context.available_actions.get(action_name)
        return available.kind if available else None
