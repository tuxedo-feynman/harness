from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder


def _builder() -> ContextBuilder:
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    return ContextBuilder(system_prompt="sys", action_directory=directory)


def test_first_operand_is_root_and_later_operands_chain_onto_the_leaf():
    builder = _builder()
    root = builder.add_operand()
    child = builder.add_operand()
    assert root.parent is None
    assert child.parent == root.id
    assert builder.leaf() is child
