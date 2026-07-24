from datetime import datetime, timezone

from harness.models import ActionDescription, ActionResult, Operand


def test_operand_resolved_when_every_request_has_a_result():
    operand = Operand(
        id="op-1",
        created_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
        parent=None,
        action_requests=[
            ActionDescription(id="ad-1", action_name="terminal", method_name="read")
        ],
    )
    assert not operand.resolved
    operand.action_results.append(ActionResult(contents="hello"))
    assert operand.resolved
