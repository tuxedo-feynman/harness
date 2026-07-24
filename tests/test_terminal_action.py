from actions.terminal_action import TerminalAction
from harness.models import Context


def test_print_writes_to_stdout_and_returns_text(capsys):
    context = Context(system_prompt="", history=[], available_actions={})
    result = TerminalAction().run("print", {"text": "hello"}, context)
    assert result.contents == "hello"
    assert result.error is None
    assert capsys.readouterr().out == "hello\n"
