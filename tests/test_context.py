from unittest.mock import Mock

import pytest

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from hyh.action_directory import ActionDirectory
from hyh.context import ContextBuilder
from hyh.models import ActionDescription, ActionResult

def _utility_get_builder() -> ContextBuilder:
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    return ContextBuilder(system_prompt="sys", action_directory=directory)


def _utility_get_request(id: str = "ad-1") -> ActionDescription:
    return ActionDescription(id=id, action_name="terminal", method_name="input")


def test_build_walks_ancestor_closure_excluding_offpath_siblings():
    builder = _utility_get_builder()
    root = builder.add_root()
    child = builder.add_operand(
        parents=[root], order=0, action_request=_utility_get_request(), action_result=ActionResult(contents="x")
    )
    sibling = builder.add_operand(parents=[root], order=1, action_request=_utility_get_request("ad-2"))
    grandchild = builder.add_operand(
        parents=[child], order=0, action_request=_utility_get_request("ad-3"), action_result=ActionResult(contents="y")
    )

    context = builder.build([grandchild])

    assert root.parents == []
    assert [op.id for op in context.history] == [root.id, child.id, grandchild.id]
    assert sibling.id not in [op.id for op in context.history]

    # move() reparents; the closure follows the graph, not creation order
    builder.move(sibling, [grandchild])
    moved = builder.build([sibling])
    assert [op.id for op in moved.history] == [root.id, child.id, grandchild.id, sibling.id]


def test_build_linearizes_a_join_deterministically_by_sibling_order():
    builder = _utility_get_builder()
    root = builder.add_root()
    thinking = builder.add_operand(
        parents=[root], order=0, action_request=_utility_get_request("t"), action_result=ActionResult(contents="")
    )
    # fan-out: two effect children in sibling order, deliberately created
    # out of order to prove order wins over creation sequence
    second = builder.add_operand(
        parents=[thinking], order=1, action_request=_utility_get_request("b"), action_result=ActionResult(contents="b")
    )
    first = builder.add_operand(
        parents=[thinking], order=0, action_request=_utility_get_request("a"), action_result=ActionResult(contents="a")
    )
    join = builder.add_operand(
        parents=[first, second], order=0, action_request=_utility_get_request("j"), action_result=ActionResult(contents="")
    )

    context = builder.build([join])

    assert [op.id for op in context.history] == [
        root.id, thinking.id, first.id, second.id, join.id
    ]


def test_build_uses_the_mocked_directory_method_index():
    directory = Mock(spec=ActionDirectory)
    directory.method_index.return_value = {"canned": "index"}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)

    context = builder.build([builder.add_root()])

    assert context.available_actions == {"canned": "index"}
    directory.method_index.assert_called_once()


def test_add_root_is_singular():
    builder = _utility_get_builder()
    root = builder.add_root()
    assert root.parents == []
    with pytest.raises(RuntimeError, match="Root operand already exists"):
        builder.add_root()


def test_listener_registry():
    builder = _utility_get_builder()
    root = builder.add_root()
    operand = builder.add_operand(parents=[root], order=0, action_request=_utility_get_request())

    builder.park_listener("terminal", operand)
    assert builder.listener_channels() == {"terminal"}
    assert builder.build([root]).listeners == [operand]

    assert builder.pop_listener("terminal") is operand
    assert builder.listener_channels() == set()
    assert builder.pop_listener("terminal") is None  # missing channel is not an error
    assert builder.build([root]).listeners == []
