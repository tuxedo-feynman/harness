from unittest.mock import ANY, Mock

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from hyh.action import Action
from hyh.action_directory import ActionDirectory
from hyh.context import ContextBuilder
from hyh.loop import ExecutionLoop
from hyh.models import ActionDescription, ActionResult
from hyh.policy import PolicyController


def _utility_get_stimulus(builder: ContextBuilder, action_name: str, contents: str):
    """A resolved input operand attached to a fresh root — one user message."""
    root = builder.add_root()
    return builder.add_operand(
        parents=[root],
        order=0,
        action_request=ActionDescription(id="ad-1", action_name=action_name, method_name="input"),
        action_result=ActionResult(contents=contents),
    )


def test_full_turn_delivers_response_and_returns_to_listening(capsys):
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = PolicyController(directory, builder, thinking_action_name="fake")
    loop = ExecutionLoop(directory, policy, builder)

    loop.run(_utility_get_stimulus(builder, "terminal", "hi"))

    assert "Hello! How can I help you?" in capsys.readouterr().out
    assert builder.listener_channels() == {"terminal"}  # turn ended by re-listening


def _utility_get_mock_policy(builder: ContextBuilder, batches: list[list[ActionDescription]]) -> Mock:
    """A policy that creates the programmed child batches, one per cycle."""
    remaining = list(batches)

    def decide(frontier, context):
        return [
            builder.add_operand(parents=frontier, order=i, action_request=request)
            for i, request in enumerate(remaining.pop(0))
        ]

    policy = Mock(spec=PolicyController)
    policy.decide.side_effect = decide
    return policy


def test_loop_executes_whatever_a_mocked_policy_decides():
    action = Mock(spec=Action)
    action.kind = "null"  # terminate after one execution
    action.methods = {}
    action.run.return_value = ActionResult(contents="programmed")

    directory = Mock(spec=ActionDirectory)
    directory.get.return_value = action
    directory.method_index.return_value = {}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = _utility_get_mock_policy(
        builder, [[ActionDescription(id="r1", action_name="mocked", method_name="do")]]
    )
    stimulus = _utility_get_stimulus(builder, "mocked", "stimulus")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    action.run.assert_called_once_with("do", {}, ANY)
    policy.decide.assert_called_once()


def test_parks_input_requests():
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = _utility_get_mock_policy(
        builder, [[ActionDescription(id="r1", action_name="terminal", method_name="input")]]
    )
    stimulus = _utility_get_stimulus(builder, "terminal", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    assert builder.listener_channels() == {"terminal"}
    parked = builder.pop_listener("terminal")
    assert parked is not None
    assert not parked.resolved


def test_null_terminates_without_executing_later_siblings():
    action = Mock(spec=Action)
    action.kind = "null"
    action.methods = {}
    action.run.return_value = ActionResult(contents="")
    directory = Mock(spec=ActionDirectory)
    directory.get.return_value = action
    directory.method_index.return_value = {}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = _utility_get_mock_policy(
        builder,
        [[
            ActionDescription(id="r1", action_name="null", method_name="terminate"),
            ActionDescription(id="r2", action_name="null", method_name="terminate"),
        ]],
    )
    stimulus = _utility_get_stimulus(builder, "mocked", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    action.run.assert_called_once()  # the sibling after null never executes


def test_executes_children_sequentially_resolving_each():
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
    policy = _utility_get_mock_policy(
        builder,
        [
            [
                ActionDescription(id="ra", action_name="tool", method_name="a"),
                ActionDescription(id="rb", action_name="tool", method_name="b"),
            ],
            [ActionDescription(id="rn", action_name="null", method_name="terminate")],
        ],
    )
    stimulus = _utility_get_stimulus(builder, "mocked", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    assert effect.run.call_count == 2
    effect.run.assert_any_call("a", {}, ANY)
    effect.run.assert_any_call("b", {}, ANY)
    # each child operand carries its own result; the second cycle's frontier
    # was the resolved pair
    second_frontier = policy.decide.call_args_list[1].args[0]
    assert [op.action_result for op in second_frontier] == [first, second]
    assert all(op.resolved for op in second_frontier)


def test_input_name_without_input_method_executes_instead_of_parking():
    action = Mock(spec=Action)
    action.kind = "null"  # terminate after the single execution
    action.methods = {}  # no input method: the name alone must not park
    action.run.return_value = ActionResult(contents="")
    directory = Mock(spec=ActionDirectory)
    directory.get.return_value = action
    directory.method_index.return_value = {}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = _utility_get_mock_policy(
        builder, [[ActionDescription(id="r1", action_name="mocked", method_name="input")]]
    )
    stimulus = _utility_get_stimulus(builder, "mocked", "hi")

    ExecutionLoop(directory, policy, builder).run(stimulus)

    action.run.assert_called_once_with("input", {}, ANY)
    assert builder.listener_channels() == set()
