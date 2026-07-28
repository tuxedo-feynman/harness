from datetime import datetime, timezone
from unittest.mock import Mock

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.telegram_action import TelegramAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.config import TelegramActionConfig
from harness.context import ContextBuilder
from harness.models import ActionDescription, ActionResult, Context, Operand
from harness.policy import PolicyController


def _utility_get_directory() -> ActionDirectory:
    return ActionDirectory(
        [
            NullAction(),
            TerminalAction(),
            FakeThinkingAction(),
            TelegramAction(TelegramActionConfig(token="test-token")),
        ]
    )


def _utility_get_policy(builder=None) -> tuple[PolicyController, ActionDirectory]:
    directory = _utility_get_directory()
    if builder is None:
        builder = Mock(spec=ContextBuilder)
    policy = PolicyController(directory, builder, thinking_action_name="fake")
    return policy, directory


def _utility_get_context(directory, history, listeners=None) -> Context:
    return Context(
        system_prompt="sys",
        history=history,
        available_actions=directory.method_index(),
        listeners=list(listeners or []),
    )


def test_primary_path_attaches_thinking_after_listener_resolves():
    directory = _utility_get_directory()
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = PolicyController(directory, builder, thinking_action_name="fake")

    root = builder.add_root()
    listener = builder.add_operand(
        parent=root,
        action_requests=[
            ActionDescription(id="ad-1", action_name="terminal", method_name="listen")
        ],
        action_results=[ActionResult(contents="hello")],
    )
    working = builder.add_operand(parent=listener)

    evaluated = policy.evaluate(working, builder.build(working))

    assert len(evaluated.action_requests) == 1
    assert evaluated.action_requests[0].action_name == "fake"
    assert evaluated.action_requests[0].method_name == "complete"


def _utility_get_operand(id: str, parent: str | None, requests=None, results=None) -> Operand:
    return Operand(
        id=id,
        created_at=datetime(2026, 7, 27, tzinfo=timezone.utc),
        parent=parent,
        action_requests=list(requests or []),
        action_results=list(results or []),
    )


def test_turn_complete_moves_uncles_via_mocked_context_builder():
    directory = _utility_get_directory()
    builder = Mock(spec=ContextBuilder)
    policy = PolicyController(directory, builder, thinking_action_name="fake")

    listener = _utility_get_operand(
        "listener", None,
        requests=[ActionDescription(id="r1", action_name="terminal", method_name="listen")],
        results=[ActionResult(contents="hello")],
    )
    delivery = _utility_get_operand(
        "delivery", "listener",
        requests=[ActionDescription(id="r2", action_name="terminal", method_name="send",
                                    method_parameters={"text": "hi"})],
        results=[ActionResult(contents="hi")],
    )
    working = _utility_get_operand("working", "delivery")
    uncle = _utility_get_operand(
        "uncle", None,
        requests=[ActionDescription(id="r3", action_name="terminal", method_name="listen")],
    )
    context = Context(
        system_prompt="sys",
        history=[listener, delivery, working],
        available_actions=directory.method_index(),
        listeners=[uncle],
    )

    evaluated = policy.evaluate(working, context)

    builder.move.assert_called_once_with(uncle, delivery)
    assert evaluated.action_requests[0].method_name == "listen"  # re-armed the origin channel


def test_untouched_when_requests_already_present():
    policy, directory = _utility_get_policy()
    request = ActionDescription(id="r1", action_name="terminal", method_name="send")
    working = _utility_get_operand("working", None, requests=[request])

    evaluated = policy.evaluate(working, _utility_get_context(directory, [working]))

    assert evaluated.action_requests == [request]


def test_attaches_thinking_when_no_parent():
    policy, directory = _utility_get_policy()
    working = _utility_get_operand("working", None)

    evaluated = policy.evaluate(working, _utility_get_context(directory, [working]))

    assert evaluated.action_requests[0].action_name == "fake"


def test_proposals_pass_through():
    policy, directory = _utility_get_policy()
    proposal = ActionDescription(
        id="call_1", action_name="terminal", method_name="send", method_parameters={"text": "hi"}
    )
    prev = _utility_get_operand(
        "prev", None,
        requests=[ActionDescription(id="r1", action_name="fake", method_name="complete")],
        results=[ActionResult(contents="", action_description_requests=[proposal])],
    )
    working = _utility_get_operand("working", "prev")

    evaluated = policy.evaluate(working, _utility_get_context(directory, [prev, working]))

    assert evaluated.action_requests == [proposal]


def test_quit_attaches_null_and_moves_uncles():
    builder = Mock(spec=ContextBuilder)
    policy, directory = _utility_get_policy(builder)
    listener = _utility_get_operand(
        "listener", None,
        requests=[ActionDescription(id="r1", action_name="terminal", method_name="listen")],
        results=[ActionResult(contents="  QUIT ")],  # whitespace and case are forgiven
    )
    working = _utility_get_operand("working", "listener")
    uncle = _utility_get_operand(
        "uncle", None,
        requests=[ActionDescription(id="r2", action_name="telegram", method_name="listen")],
    )

    evaluated = policy.evaluate(
        working, _utility_get_context(directory, [listener, working], listeners=[uncle])
    )

    assert evaluated.action_requests[0].action_name == "null"
    builder.move.assert_called_once_with(uncle, listener)


def test_empty_input_rearms_listener_without_thinking():
    builder = Mock(spec=ContextBuilder)
    policy, directory = _utility_get_policy(builder)
    listener = _utility_get_operand(
        "listener", None,
        requests=[ActionDescription(id="r1", action_name="terminal", method_name="listen")],
        results=[ActionResult(contents="   ")],
    )
    working = _utility_get_operand("working", "listener")

    evaluated = policy.evaluate(working, _utility_get_context(directory, [listener, working]))

    assert evaluated.action_requests[0].action_name == "terminal"
    assert evaluated.action_requests[0].method_name == "listen"


def test_delivery_routes_to_origin_channel_with_metadata():
    policy, directory = _utility_get_policy()
    listener = _utility_get_operand(
        "listener", None,
        requests=[ActionDescription(id="r1", action_name="telegram", method_name="listen")],
        results=[
            ActionResult(contents="question", metadata={"chat_id": "42", "username": "u"})
        ],
    )
    thinking = _utility_get_operand(
        "thinking", "listener",
        requests=[ActionDescription(id="r2", action_name="fake", method_name="complete")],
        results=[ActionResult(contents="answer")],
    )
    working = _utility_get_operand("working", "thinking")

    evaluated = policy.evaluate(
        working, _utility_get_context(directory, [listener, thinking, working])
    )

    request = evaluated.action_requests[0]
    assert request.action_name == "telegram"
    assert request.method_name == "send"
    # metadata keys in the send schema pass through; others are filtered
    assert request.method_parameters == {"text": "answer", "chat_id": "42"}


def test_empty_thinking_result_rearms_listener():
    builder = Mock(spec=ContextBuilder)
    policy, directory = _utility_get_policy(builder)
    listener = _utility_get_operand(
        "listener", None,
        requests=[ActionDescription(id="r1", action_name="terminal", method_name="listen")],
        results=[ActionResult(contents="hello")],
    )
    thinking = _utility_get_operand(
        "thinking", "listener",
        requests=[ActionDescription(id="r2", action_name="fake", method_name="complete")],
        results=[ActionResult(contents="")],
    )
    working = _utility_get_operand("working", "thinking")

    evaluated = policy.evaluate(
        working, _utility_get_context(directory, [listener, thinking, working])
    )

    assert evaluated.action_requests[0].method_name == "listen"
    assert evaluated.action_requests[0].action_name == "terminal"


def test_no_origin_channel_attaches_null():
    policy, directory = _utility_get_policy()
    thinking = _utility_get_operand(
        "thinking", None,
        requests=[ActionDescription(id="r1", action_name="fake", method_name="complete")],
        results=[ActionResult(contents="answer")],
    )
    working = _utility_get_operand("working", "thinking")

    evaluated = policy.evaluate(working, _utility_get_context(directory, [thinking, working]))

    assert evaluated.action_requests[0].action_name == "null"


def test_effect_results_route_to_thinking():
    policy, directory = _utility_get_policy()
    listener = _utility_get_operand(
        "listener", None,
        requests=[ActionDescription(id="r1", action_name="terminal", method_name="listen")],
        results=[ActionResult(contents="hello")],
    )
    # a resolved effect request that is neither listen nor send (a future tool)
    effect = _utility_get_operand(
        "effect", "listener",
        requests=[ActionDescription(id="r2", action_name="terminal", method_name="beep")],
        results=[ActionResult(contents="beeped")],
    )
    working = _utility_get_operand("working", "effect")

    evaluated = policy.evaluate(
        working, _utility_get_context(directory, [listener, effect, working])
    )

    assert evaluated.action_requests[0].action_name == "fake"
