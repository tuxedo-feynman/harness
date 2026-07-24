from actions.null_action import NullAction
from harness.models import Context


def test_terminate_returns_empty_result_without_error():
    result = NullAction().run(
        "terminate", {}, Context(system_prompt="", history=[], available_actions={})
    )
    assert result.contents == ""
    assert result.error is None
    assert result.action_description_requests == []
