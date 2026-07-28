from unittest.mock import Mock

from actions.fake_thinking_action import FakeThinkingAction
from harness.action import Action
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.dispatcher import Dispatcher
from harness.loop import ExecutionLoop
from harness.models import ActionResult
from harness.policy import PolicyController


def _utility_get_dispatcher(loop=None) -> tuple[Dispatcher, ContextBuilder]:
    """A dispatcher over the real stack; pass a mock loop to isolate dispatch."""
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    if loop is None:
        policy = PolicyController(directory, builder, thinking_action_name="fake")
        loop = ExecutionLoop(directory, policy, builder)
    return Dispatcher(directory, builder, loop), builder


def test_stimulus_runs_full_turn_and_rearms_the_listener(capsys):
    dispatcher, builder = _utility_get_dispatcher()

    dispatcher.arm_initial(builder.add_root())
    assert builder.listener_channels() == {"terminal"}

    dispatcher.handle_stimulus("terminal", ActionResult(contents="hi"))

    assert "Hello! How can I help you?" in capsys.readouterr().out
    assert builder.listener_channels() == {"terminal"}  # re-armed by policy


def test_handle_stimulus_resolves_the_listener_and_runs_a_mocked_loop():
    loop = Mock(spec=ExecutionLoop)
    dispatcher, builder = _utility_get_dispatcher(loop)
    dispatcher.arm_initial(builder.add_root())
    listener = builder.pop_listener("terminal")
    assert listener is not None
    builder.park_listener("terminal", listener)

    dispatcher.handle_stimulus("terminal", ActionResult(contents="hi"))

    loop.run.assert_called_once_with(listener)
    assert listener.resolved  # the stimulus result was attached before the run


def test_channels_lists_only_listening_actions():
    dispatcher, _ = _utility_get_dispatcher(Mock(spec=ExecutionLoop))
    assert dispatcher._channels() == ["terminal"]  # null and fake do not listen


def test_arm_initial_parents_listeners_to_root():
    dispatcher, builder = _utility_get_dispatcher(Mock(spec=ExecutionLoop))
    root = builder.add_root()

    dispatcher.arm_initial(root)

    listener = builder.pop_listener("terminal")
    assert listener is not None
    assert listener.parent == root.id
    assert listener.action_requests[0].method_name == "listen"
    assert not listener.resolved


def test_handle_stimulus_drops_message_without_a_pending_listener():
    loop = Mock(spec=ExecutionLoop)
    dispatcher, builder = _utility_get_dispatcher(loop)
    # no arm_initial: nothing is listening

    dispatcher.handle_stimulus("terminal", ActionResult(contents="orphaned"))

    loop.run.assert_not_called()


def test_listen_pushes_results_then_a_close_sentinel():
    dispatcher, _ = _utility_get_dispatcher(Mock(spec=ExecutionLoop))
    action = Mock(spec=Action)
    action.name = "terminal"
    result = ActionResult(contents="hi")
    action.run.side_effect = [result, EOFError()]

    dispatcher._listen(action)

    assert dispatcher._queue.get_nowait() == ("terminal", result)
    assert dispatcher._queue.get_nowait() == ("terminal", None)


def test_listen_pushes_thread_exceptions_for_serve_to_reraise():
    dispatcher, _ = _utility_get_dispatcher(Mock(spec=ExecutionLoop))
    action = Mock(spec=Action)
    action.name = "telegram"
    boom = RuntimeError("connection lost")
    action.run.side_effect = boom

    dispatcher._listen(action)

    assert dispatcher._queue.get_nowait() == ("telegram", boom)
