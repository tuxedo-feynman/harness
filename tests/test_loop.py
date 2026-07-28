from unittest.mock import ANY, Mock

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action import Action
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.loop import ExecutionLoop
from harness.models import ActionDescription, ActionResult
from harness.policy import PolicyController


def _utility_get_stimulus(builder: ContextBuilder, action_name: str, contents: str):
    """A resolved listen operand attached to a fresh root — one user message."""
    root = builder.add_root()
    return builder.add_operand(
        parent=root,
        action_requests=[
            ActionDescription(id="ad-1", action_name=action_name, method_name="listen")
        ],
        action_results=[ActionResult(contents=contents)],
    )


def test_full_turn_delivers_response_and_returns_to_listening(capsys):
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = PolicyController(directory, builder, thinking_action_name="fake")
    loop = ExecutionLoop(directory, policy, builder)

    loop.run(_utility_get_stimulus(builder, "terminal", "hi"))

    assert "Hello! How can I help you?" in capsys.readouterr().out
    assert builder.listener_channels() == {"terminal"}  # turn ended by re-listening


def test_loop_executes_whatever_a_mocked_policy_attaches():
    # The action always returns the programmed result; the policy always
    # attaches the programmed request. The loop is tested in isolation.
    action = Mock(spec=Action)
    action.kind = "null"  # terminate after one execution
    action.methods = {}
    action.run.return_value = ActionResult(contents="programmed")

    directory = Mock(spec=ActionDirectory)
    directory.get.return_value = action
    directory.method_index.return_value = {}

    request = ActionDescription(id="r1", action_name="mocked", method_name="do")

    def attach(operand, context):
        operand.action_requests = [request]
        return operand

    policy = Mock(spec=PolicyController)
    policy.evaluate.side_effect = attach

    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    stimulus = _utility_get_stimulus(builder, "mocked", "stimulus")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    action.run.assert_called_once_with("do", {}, ANY)
    policy.evaluate.assert_called_once()


def _utility_get_mock_policy(attachments: list[list[ActionDescription]]) -> Mock:
    """A policy that attaches the programmed request batches, one per cycle."""
    batches = list(attachments)

    def attach(operand, context):
        operand.action_requests = batches.pop(0)
        return operand

    policy = Mock(spec=PolicyController)
    policy.evaluate.side_effect = attach
    return policy


def test_parks_listen_and_truncates_trailing_requests():
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = _utility_get_mock_policy(
        [[
            ActionDescription(id="r1", action_name="terminal", method_name="listen"),
            ActionDescription(id="r2", action_name="terminal", method_name="send"),
        ]]
    )
    stimulus = _utility_get_stimulus(builder, "terminal", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    assert builder.listener_channels() == {"terminal"}
    parked = builder.pop_listener("terminal")
    assert parked is not None
    assert len(parked.action_requests) == 1  # trailing send was dropped
    assert not parked.resolved


def test_null_terminates_the_batch_immediately():
    action = Mock(spec=Action)
    action.kind = "null"
    action.methods = {}
    action.run.return_value = ActionResult(contents="")
    directory = Mock(spec=ActionDirectory)
    directory.get.return_value = action
    directory.method_index.return_value = {}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = _utility_get_mock_policy(
        [[
            ActionDescription(id="r1", action_name="null", method_name="terminate"),
            ActionDescription(id="r2", action_name="null", method_name="terminate"),
        ]]
    )
    stimulus = _utility_get_stimulus(builder, "mocked", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    action.run.assert_called_once()  # the request after null never executes


def test_executes_batches_sequentially_pairing_results():
    effect = Mock(spec=Action)
    effect.kind = "effect"
    effect.methods = {}
    first, second = ActionResult(contents="one"), ActionResult(contents="two")
    effect.run.side_effect = [first, second]
    null = Mock(spec=Action)
    null.kind = "null"
    null.methods = {}
    null.run.return_value = ActionResult(contents="")

    directory = Mock(spec=ActionDirectory)
    directory.get.side_effect = lambda name: null if name == "null" else effect
    directory.method_index.return_value = {}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    batch = [
        ActionDescription(id="ra", action_name="tool", method_name="a"),
        ActionDescription(id="rb", action_name="tool", method_name="b"),
    ]
    policy = _utility_get_mock_policy(
        [batch, [ActionDescription(id="rn", action_name="null", method_name="terminate")]]
    )
    stimulus = _utility_get_stimulus(builder, "mocked", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    assert effect.run.call_count == 2
    effect.run.assert_any_call("a", {}, ANY)
    effect.run.assert_any_call("b", {}, ANY)
    # results pair by index on the operand that carried the batch
    batch_operand = policy.evaluate.call_args_list[0].args[0]
    assert batch_operand.action_results == [first, second]
    assert batch_operand.resolved


def test_listen_name_without_listen_method_executes_instead_of_parking():
    action = Mock(spec=Action)
    action.kind = "null"  # terminate after the single execution
    action.methods = {}  # no listen method: the name alone must not park
    action.run.return_value = ActionResult(contents="")
    directory = Mock(spec=ActionDirectory)
    directory.get.return_value = action
    directory.method_index.return_value = {}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = _utility_get_mock_policy(
        [[ActionDescription(id="r1", action_name="mocked", method_name="listen")]]
    )
    stimulus = _utility_get_stimulus(builder, "mocked", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    action.run.assert_called_once_with("listen", {}, ANY)
    assert builder.listener_channels() == set()
