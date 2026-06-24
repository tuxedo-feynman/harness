from pathlib import Path

from harness.config import Config
from harness.context import ContextBuilder
from harness.messages import AssistantTurn, Message, ToolResult
from harness.sessions import SessionManager
from providers.base import ModelProvider
from tools.base import ToolContext
from tools.registry import ToolRegistry


def _assistant_turn_to_message(turn: AssistantTurn) -> Message:
    return Message(
        role="assistant",
        content=turn.content,
        tool_calls=turn.tool_calls if turn.tool_calls else None,
    )


def _tool_result_to_message(result: ToolResult) -> Message:
    return Message(
        role="tool",
        content=result.content,
        tool_call_id=result.tool_call_id,
        name=result.name,
    )


class HarnessLoop:
    def __init__(
        self,
        config: Config,
        session: SessionManager,
        context_builder: ContextBuilder,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        workspace_root: Path,
    ):
        self.config = config
        self.session = session
        self.context_builder = context_builder
        self.provider = provider
        self.tool_registry = tool_registry
        self.workspace_root = workspace_root

    def run_turn(self, session_id: str, user_text: str) -> str:
        history = self.session.get_messages(session_id)
        messages = self.context_builder.build(self.config.system_prompt, history, user_text)
        self.session.record_user_message(session_id, user_text)

        tool_context = ToolContext(session_id=session_id, workspace_root=self.workspace_root)
        tool_rounds = 0

        while True:
            assistant_turn = self.provider.complete(messages, self.tool_registry.definitions())
            self.session.record_assistant_turn(session_id, assistant_turn)
            messages.append(_assistant_turn_to_message(assistant_turn))

            if not assistant_turn.tool_calls:
                return assistant_turn.content or ""

            if tool_rounds >= self.config.max_tool_rounds:
                error_msg = f"Max tool rounds ({self.config.max_tool_rounds}) exceeded"
                self.session.record_error(session_id, error_msg)
                return error_msg

            for call in assistant_turn.tool_calls:
                result = self.tool_registry.run(call, tool_context)
                self.session.record_tool_result(session_id, result)
                messages.append(_tool_result_to_message(result))

            tool_rounds += 1
