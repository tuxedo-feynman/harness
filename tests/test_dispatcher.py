from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.dispatcher import Dispatcher
from harness.loop import ExecutionLoop
from harness.models import ActionResult
from harness.policy import PolicyController


def test_stimulus_runs_full_turn_and_rearms_the_listener(capsys):
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = PolicyController(directory, builder, thinking_action_name="fake")
    loop = ExecutionLoop(directory, policy, builder)
    dispatcher = Dispatcher(directory, builder, loop)

    dispatcher.arm_initial()
    assert builder.listener_channels() == {"terminal"}

    dispatcher.handle_stimulus("terminal", ActionResult(contents="hi"))

    assert "Hello! How can I help you?" in capsys.readouterr().out
    assert builder.listener_channels() == {"terminal"}  # re-armed by policy
