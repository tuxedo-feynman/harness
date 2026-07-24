from actions.fake_thinking_action import FakeThinkingAction
from harness.models import ActionResult, Context


def test_returns_programmed_responses_in_order():
    action = FakeThinkingAction(
        [ActionResult(contents="first"), ActionResult(contents="second")]
    )
    context = Context(system_prompt="", history=[], available_actions={})
    assert action.run("complete", {}, context).contents == "first"
    assert action.run("complete", {}, context).contents == "second"
    assert len(action.calls) == 2
