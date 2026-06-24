from harness.messages import AssistantTurn, Message, ToolDefinition


class FakeProvider:
    """Pre-programmed responses for testing. Returns turns in order, raises when exhausted."""

    def __init__(self, responses: list[AssistantTurn]):
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], list[ToolDefinition]]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AssistantTurn:
        self.calls.append((list(messages), list(tools)))
        if not self._responses:
            raise RuntimeError("FakeProvider has no more programmed responses")
        return self._responses.pop(0)
