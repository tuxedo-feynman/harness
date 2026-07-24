from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.openai_thinking_action import OpenAIThinkingAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.config import ThinkingActionConfig
from harness.context import ContextBuilder
from harness.models import ActionDescription, ActionResult


def test_converts_user_input_history_to_openai_messages():
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    builder.add_operand(
        action_requests=[
            ActionDescription(id="ad-1", action_name="terminal", method_name="read")
        ],
        action_results=[ActionResult(contents="hello")],
    )

    action = OpenAIThinkingAction(ThinkingActionConfig(type="openai"))
    messages = action._to_openai_messages(builder.build())

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
