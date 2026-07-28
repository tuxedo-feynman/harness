from datetime import datetime, timezone

from harness.models import ActionDescription, ActionResult, Operand


def _utility_get_operand(requests=None, results=None) -> Operand:
    return Operand(
        id="op-1",
        created_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
        parent=None,
        action_requests=list(requests or []),
        action_results=list(results or []),
    )


def test_operand_resolved():
    operand = _utility_get_operand(
        requests=[ActionDescription(id="ad-1", action_name="terminal", method_name="listen")]
    )
    assert not operand.resolved
    operand.action_results.append(ActionResult(contents="hello"))
    assert operand.resolved
    # known edge: an operand with no requests is vacuously resolved
    assert _utility_get_operand().resolved


def test_model_defaults():
    description = ActionDescription(id="ad-1", action_name="terminal", method_name="listen")
    assert description.method_parameters == {}

    result = ActionResult(contents="hi")
    assert result.metadata == {}
    assert result.action_description_requests == []
    assert result.error is None

    operand = _utility_get_operand()
    assert operand.action_requests == []
    assert operand.action_results == []
