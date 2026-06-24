from typing import Protocol

from harness.messages import AssistantTurn, Message, ToolDefinition


class ModelProvider(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AssistantTurn:
        ...
