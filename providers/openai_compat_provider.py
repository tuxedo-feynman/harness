import json
import os

from harness.config import ProviderConfig
from harness.messages import AssistantTurn, Message, ToolCall, ToolDefinition

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' package is required for the openai_compat provider. "
        "Install it with: pip install 'llm-harness[openai_compat]'"
    ) from e


def _to_openai_message(msg: Message) -> dict:
    if msg.role == "tool":
        return {
            "role": "tool",
            "content": msg.content or "",
            "tool_call_id": msg.tool_call_id,
        }
    if msg.role == "assistant" and msg.tool_calls:
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ],
        }
    return {"role": msg.role, "content": msg.content or ""}


def _to_openai_tool(defn: ToolDefinition) -> dict:
    return {
        "type": "function",
        "function": {
            "name": defn.name,
            "description": defn.description,
            "parameters": defn.parameters,
        },
    }


class OpenAICompatProvider:
    def __init__(self, config: ProviderConfig):
        # llama-server doesn't require a real key; fall back to "local" if unset
        api_key = os.environ.get("OPENAI_API_KEY", "local")
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)
        self._model = config.model

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AssistantTurn:
        kwargs: dict = {
            "model": self._model,
            "messages": [_to_openai_message(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return AssistantTurn(content=choice.content, tool_calls=tool_calls)
