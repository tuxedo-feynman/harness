from unittest.mock import Mock

import pytest

from actions.null_action import NullAction
from harness.action import Action
from harness.action_directory import ActionDirectory


def _utility_get_mock_action(name, kind, methods) -> Action:
    action = Mock(spec=Action)
    action.name = name
    action.kind = kind
    action.methods = methods
    return action


def test_duplicate_action_names_raise():
    with pytest.raises(ValueError, match="Duplicate action name"):
        ActionDirectory([NullAction(), NullAction()])


def test_directory_get():
    fake = _utility_get_mock_action("mocked", "effect", {})
    fake2 = _utility_get_mock_action("mocked2", "thinking", {})
    directory = ActionDirectory([fake, fake2])
    assert directory.get("mocked") is fake
    assert directory.get("mocked2") is fake2
    with pytest.raises(KeyError, match="Unknown action: 'unknown'"):
        directory.get("unknown")


def test_directory_method_index():
    fake = _utility_get_mock_action("mocked", "effect", {})
    directory = ActionDirectory([fake])
    assert directory.method_index()["mocked"].kind == "effect"
    assert len(directory.method_index().keys()) == 1
    assert directory.method_index()["mocked"].methods == {}  # copied, never None
