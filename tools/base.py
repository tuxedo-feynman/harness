from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from harness.messages import ToolDefinition, ToolResult


@dataclass
class ToolContext:
    session_id: str
    workspace_root: Path


class Tool(Protocol):
    name: str

    def definition(self) -> ToolDefinition:
        ...

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        ...
