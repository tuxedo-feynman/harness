from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.models import ActionDescription, ActionResult
from harness.policy import PolicyController


def test_primary_path_attaches_thinking_after_user_input():
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = PolicyController(directory, thinking_action_name="fake")

    builder.add_operand(
        action_requests=[
            ActionDescription(id="ad-1", action_name="terminal", method_name="read")
        ],
        action_results=[ActionResult(contents="hello")],
    )
    leaf = builder.add_operand()

    evaluated = policy.evaluate(leaf, builder.build())

    assert len(evaluated.action_requests) == 1
    assert evaluated.action_requests[0].action_name == "fake"
    assert evaluated.action_requests[0].method_name == "complete"
