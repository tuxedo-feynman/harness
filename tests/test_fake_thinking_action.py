from datetime import datetime, timezone
from unittest.mock import Mock

from actions.fake_thinking_action import FakeThinkingAction
from harness.models import ActionDescription, ActionResult, Context, Operand


def test_returns_programmed_responses_in_order():
    action = FakeThinkingAction(
        [ActionResult(contents="first"), ActionResult(contents="second")]
    )
    context = Context(system_prompt="", history=[], available_actions={})
    assert action.run("complete", {}, context).contents == "first"
    assert action.run("complete", {}, context).contents == "second"
    assert len(action.calls) == 2


def test_fallback_reads_history_from_a_mocked_context():
    context = Mock(spec=Context)
    context.history = []
    context.available_actions = {}
    result = FakeThinkingAction().run("complete", {}, context)
    assert result.contents == "Hello! How can I help you?"


def _utility_get_listen_operand(contents: str) -> Operand:
    return Operand(
        id=f"op-{contents or 'blank'}",
        created_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
        parent=None,
        action_requests=[
            ActionDescription(id="r1", action_name="terminal", method_name="listen")
        ],
        action_results=[ActionResult(contents=contents)],
    )


def test_fallback_echoes_the_latest_input_ignoring_blanks():
    context = Mock(spec=Context)
    context.history = [
        _utility_get_listen_operand("first"),
        _utility_get_listen_operand("   "),  # blank inputs never count
        _utility_get_listen_operand("second"),
    ]
    context.available_actions = {}
    result = FakeThinkingAction().run("complete", {}, context)
    assert result.contents == "[fake] second"
