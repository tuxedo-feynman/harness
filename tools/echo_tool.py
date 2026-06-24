from typing import Any

from harness.messages import ToolDefinition, ToolResult
from tools.base import ToolContext


class EchoTool:
    name = "echo"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Echoes the provided text back unchanged.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to echo."},
                },
                "required": ["text"],
            },
        )

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        text = arguments.get("text")
        if text is None:
            return ToolResult(
                tool_call_id="",
                name=self.name,
                content="Missing required argument: text",
                ok=False,
                error="Missing required argument: text",
            )
        if not isinstance(text, str):
            return ToolResult(
                tool_call_id="",
                name=self.name,
                content=f"Expected text to be a string, got {type(text).__name__}",
                ok=False,
                error=f"Invalid type for 'text': expected str, got {type(text).__name__}",
            )
        return ToolResult(
            tool_call_id="",
            name=self.name,
            content=text,
            ok=True,
        )
