from harness.messages import AssistantTurn, Message, ToolCall, ToolDefinition


class FakeProvider:
    """Pre-programmed responses for testing. Returns turns in order.

    Default fallback behaviour (no pre-programmed responses):
    - First call, echo tool available: calls echo with the user's text to exercise
      the full tool call path without a real model.
    - First call, no echo tool: returns a greeting directly.
    - Subsequent calls: echoes the last user message prefixed with [fake].
    """

    def __init__(self, responses: list[AssistantTurn] | None = None):
        self._responses = list(responses or [])
        self.calls: list[tuple[list[Message], list[ToolDefinition]]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AssistantTurn:
        self.calls.append((list(messages), list(tools)))
        if self._responses:
            return self._responses.pop(0)
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        tool_names = {t.name for t in tools}
        if len(self.calls) == 1 and "echo" in tool_names:
            return AssistantTurn(
                content=None,
                tool_calls=[ToolCall(id="fake-1", name="echo", arguments={"text": last_user})],
            )
        if len(self.calls) == 1 or any(m.role == "tool" for m in messages):
            return AssistantTurn(content="Hello! How can I help you?", tool_calls=[])
        return AssistantTurn(content=f"[fake] {last_user}", tool_calls=[])
