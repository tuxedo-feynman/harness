from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder


def _builder() -> ContextBuilder:
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    return ContextBuilder(system_prompt="sys", action_directory=directory)


def test_build_walks_ancestor_path_from_root_to_anchor():
    builder = _builder()
    root = builder.add_root()
    child = builder.add_operand(parent=root)
    sibling = builder.add_operand(parent=root)
    grandchild = builder.add_operand(parent=child)

    context = builder.build(grandchild)

    assert root.parent is None
    assert [op.id for op in context.history] == [root.id, child.id, grandchild.id]
    assert sibling.id not in [op.id for op in context.history]
