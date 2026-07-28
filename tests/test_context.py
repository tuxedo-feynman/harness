from unittest.mock import Mock

import pytest

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.models import ActionDescription


def _utility_get_builder() -> ContextBuilder:
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    return ContextBuilder(system_prompt="sys", action_directory=directory)


def test_build_walks_ancestor_path_from_root_to_anchor():
    builder = _utility_get_builder()
    root = builder.add_root()
    child = builder.add_operand(parent=root)
    sibling = builder.add_operand(parent=root)
    grandchild = builder.add_operand(parent=child)

    context = builder.build(grandchild)

    assert root.parent is None
    assert [op.id for op in context.history] == [root.id, child.id, grandchild.id]
    assert sibling.id not in [op.id for op in context.history]

    # move() reparents; the ancestor path follows the tree, not creation order
    builder.move(sibling, grandchild)
    moved = builder.build(sibling)
    assert [op.id for op in moved.history] == [root.id, child.id, grandchild.id, sibling.id]


def test_build_uses_the_mocked_directory_method_index():
    directory = Mock(spec=ActionDirectory)
    directory.method_index.return_value = {"canned": "index"}
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)

    context = builder.build(builder.add_root())

    assert context.available_actions == {"canned": "index"}
    directory.method_index.assert_called_once()


def test_add_root_is_singular():
    builder = _utility_get_builder()
    root = builder.add_root()
    assert root.parent is None
    with pytest.raises(RuntimeError, match="Root operand already exists"):
        builder.add_root()


def test_add_operand_copies_request_and_result_lists():
    builder = _utility_get_builder()
    root = builder.add_root()
    requests = [ActionDescription(id="ad-1", action_name="terminal", method_name="listen")]

    operand = builder.add_operand(parent=root, action_requests=requests)
    requests.append(ActionDescription(id="ad-2", action_name="terminal", method_name="send"))

    assert operand.parent == root.id
    assert len(operand.action_requests) == 1  # caller's list mutation is isolated


def test_listener_registry():
    builder = _utility_get_builder()
    root = builder.add_root()
    operand = builder.add_operand(
        parent=root,
        action_requests=[ActionDescription(id="ad-1", action_name="terminal", method_name="listen")],
    )

    builder.park_listener("terminal", operand)
    assert builder.listener_channels() == {"terminal"}
    assert builder.build(root).listeners == [operand]

    assert builder.pop_listener("terminal") is operand
    assert builder.listener_channels() == set()
    assert builder.pop_listener("terminal") is None  # missing channel is not an error
    assert builder.build(root).listeners == []
