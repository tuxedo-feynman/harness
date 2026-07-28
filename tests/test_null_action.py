from unittest.mock import Mock

from actions.null_action import NullAction
from harness.models import Context


def test_run():
    context = Mock(spec=Context)
    result = NullAction().run("terminate", {}, context)
    assert result.contents == ""
    assert result.error is None
    assert result.action_description_requests == []
    assert context.mock_calls == []  # never touches the context
