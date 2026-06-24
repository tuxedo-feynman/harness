from harness.messages import ToolCall, ToolDefinition, ToolResult
from tools.base import Tool, ToolContext


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition() for t in self._tools.values()]

    def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name!r}",
                ok=False,
                error=f"Tool {call.name!r} is not registered",
            )
        try:
            result = tool.run(call.arguments, context)
            result.tool_call_id = call.id
            return result
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool error: {e}",
                ok=False,
                error=str(e),
            )
