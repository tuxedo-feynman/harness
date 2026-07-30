from datetime import datetime, timezone

from hyh.models import ActionDescription, ActionResult, Operand


def _utility_get_operand(request=None, result=None, parents=None) -> Operand:
    return Operand(
        id="op-1",
        created_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
        parents=list(parents or []),
        order=0,
        action_request=request,
        action_result=result,
    )


def test_operand_resolved_is_promise_semantics():
    pending = _utility_get_operand(
        request=ActionDescription(id="ad-1", action_name="terminal", method_name="input")
    )
    assert not pending.resolved  # a request is a promise until its event arrives
    pending.action_result = ActionResult(contents="hello")
    assert pending.resolved
    # the root awaits nothing
    assert _utility_get_operand().resolved


def test_model_defaults():
    description = ActionDescription(id="ad-1", action_name="terminal", method_name="input")
    assert description.method_parameters == {}

    result = ActionResult(contents="hi")
    assert result.metadata == {}
    assert result.action_description_requests == []
    assert result.error is None

    operand = _utility_get_operand()
    assert operand.action_request is None
    assert operand.action_result is None
    assert operand.parents == []
    assert operand.logs == []
