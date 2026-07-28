from unittest.mock import patch

import pytest

from actions.terminal_action import TerminalAction
from hyh.models import Context


def test_send_writes_to_stdout_and_returns_text(capsys):
    context = Context(system_prompt="", history=[], available_actions={})
    result = TerminalAction().run("send", {"text": "hello"}, context)
    assert result.contents == "hello"
    assert result.error is None
    assert capsys.readouterr().out == "hello\n"


def test_listen_returns_programmed_input():
    context = Context(system_prompt="", history=[], available_actions={})
    with patch("builtins.input", return_value="canned line") as fake_input:
        result = TerminalAction().run("listen", {}, context)
    assert result.contents == "canned line"
    fake_input.assert_called_once()


def test_run_rejects_bad_arguments_and_unknown_methods():
    context = Context(system_prompt="", history=[], available_actions={})
    action = TerminalAction()
    with pytest.raises(ValueError, match="'text' must be a string"):
        action.run("send", {}, context)
    with pytest.raises(ValueError, match="'text' must be a string"):
        action.run("send", {"text": 42}, context)
    with pytest.raises(ValueError, match="Unknown method: 'beep'"):
        action.run("beep", {}, context)
