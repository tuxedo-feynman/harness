import pytest

from actions.null_action import NullAction
from harness.action_directory import ActionDirectory


def test_duplicate_action_names_raise():
    with pytest.raises(ValueError, match="Duplicate action name"):
        ActionDirectory([NullAction(), NullAction()])
