import json
import logging
import time
from pathlib import Path

from harness.config import Config
from harness.context import ContextBuilder
from harness.logger import new_id
from harness.messages import AssistantTurn, Message, ToolResult
from harness.sessions import SessionManager
from providers.base import ModelProvider
from tools.base import ToolContext
from tools.registry import ToolRegistry

log = logging.getLogger(__name__)


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
        request_id = new_id()
        history, context_at = self.session.get_context_snapshot(session_id)
        messages = self.context_builder.build(self.config.system_prompt, history, user_text)
        self.session.record_user_message(session_id, user_text)

        log.info(
            f"user_message session={session_id} request={request_id} content={user_text!r}"
        )

        tool_context = ToolContext(session_id=session_id, workspace_root=self.workspace_root)
        tool_rounds = 0
        total_tool_calls = 0
        turn_start = time.monotonic()

        while True:
            turn_id = new_id()
            assistant_turn = self.provider.complete(messages, self.tool_registry.definitions())
            self.session.record_assistant_turn(session_id, assistant_turn)
            messages.append(_assistant_turn_to_message(assistant_turn))

            if not assistant_turn.tool_calls:
                duration = time.monotonic() - turn_start
                log.info(
                    f"assistant_turn session={session_id} request={request_id} turn={turn_id}"
                    f" context_at={context_at} content={assistant_turn.content!r}"
                    f" provider={self.config.provider.type} model={self.config.provider.model}"
                    f" rounds={tool_rounds} tool_calls={total_tool_calls} duration={duration:.3f}s"
                )
                return assistant_turn.content or ""

            if tool_rounds >= self.config.max_tool_rounds:
                error_msg = f"Max tool rounds ({self.config.max_tool_rounds}) exceeded"
                self.session.record_error(session_id, error_msg)
                duration = time.monotonic() - turn_start
                log.warning(
                    f"error session={session_id} request={request_id} turn={turn_id}"
                    f" msg={error_msg!r} rounds={tool_rounds} duration={duration:.3f}s"
                )
                return error_msg

            log.info(
                f"assistant_turn session={session_id} request={request_id} turn={turn_id}"
                f" context_at={context_at} content={assistant_turn.content!r}"
                f" provider={self.config.provider.type} model={self.config.provider.model}"
                f" rounds={tool_rounds} tool_calls_requested={len(assistant_turn.tool_calls)}"
            )

            for call in assistant_turn.tool_calls:
                result = self.tool_registry.run(call, tool_context)
                self.session.record_tool_result(session_id, result)
                messages.append(_tool_result_to_message(result))
                total_tool_calls += 1

                log.info(
                    f"tool_call session={session_id} request={request_id} turn={turn_id}"
                    f" call_id={call.id} tool={call.name} args={json.dumps(call.arguments)}"
                    f" ok={result.ok} result={result.content!r}"
                )

            tool_rounds += 1
