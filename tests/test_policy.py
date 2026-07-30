from unittest.mock import patch

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.telegram_action import TelegramAction
from actions.terminal_action import TerminalAction
from hyh.action_directory import ActionDirectory
from hyh.config import TelegramActionConfig
from hyh.context import ContextBuilder
from hyh.models import ActionDescription, ActionResult
from hyh.policy import PolicyController


def _utility_get_directory() -> ActionDirectory:
    return ActionDirectory(
        [
            NullAction(),
            TerminalAction(),
            FakeThinkingAction(),
            TelegramAction(TelegramActionConfig(token="test-token")),
        ]
    )


def _utility_get_stack() -> tuple[PolicyController, ContextBuilder]:
    directory = _utility_get_directory()
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    policy = PolicyController(directory, builder, thinking_action_name="fake")
    return policy, builder


def _utility_get_stimulus(builder, channel="terminal", contents="hello", metadata=None):
    root = builder.add_root()
    return builder.add_operand(
        parents=[root],
        order=0,
        action_request=ActionDescription(id="ad-1", action_name=channel, method_name="input"),
        action_result=ActionResult(contents=contents, metadata=dict(metadata or {})),
    )


def test_stimulus_attaches_thinking():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(builder)

    children = policy.decide([stimulus], builder.build([stimulus]))

    assert len(children) == 1
    request = children[0].action_request
    assert request is not None
    assert request.action_name == "fake"
    assert request.method_name == "complete"
    assert children[0].parents == [stimulus.id]


def test_telegram_stimulus_fires_typing_as_offpath_sibling():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(builder, channel="telegram", metadata={"chat_id": "42"})

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        children = policy.decide([stimulus], builder.build([stimulus]))

    api.assert_called_once_with("sendChatAction", {"chat_id": "42", "action": "typing"})
    thinking = children[0]
    assert thinking.action_request is not None
    assert thinking.action_request.method_name == "complete"
    # the typing operand is a resolved sibling: same parents, next order,
    # excluded from the closure the model will see
    sibling = builder._operands[-1]
    assert sibling.parents == [stimulus.id]
    assert sibling.order == 1
    assert sibling.action_request is not None
    assert sibling.action_request.method_name == "typing"
    assert sibling.resolved
    assert sibling not in builder.build([thinking]).history


def test_canned_reply_for_empty_input_fires_no_typing():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(
        builder, channel="telegram", contents="", metadata={"chat_id": "42", "username": "u"}
    )

    with patch.object(TelegramAction, "_api") as api:
        children = policy.decide([stimulus], builder.build([stimulus]))

    api.assert_not_called()
    request = children[0].action_request
    assert request is not None
    assert request.action_name == "telegram"
    assert request.method_name == "send"
    # metadata keys in the send schema pass through; others are filtered
    assert request.method_parameters == {"text": "I didn't get that.", "chat_id": "42"}


def test_proposals_become_ordered_children():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(builder)
    proposals = [
        ActionDescription(id="call_1", action_name="terminal", method_name="send",
                          method_parameters={"text": "a"}),
        ActionDescription(id="call_2", action_name="terminal", method_name="send",
                          method_parameters={"text": "b"}),
    ]
    thinking = builder.add_operand(
        parents=[stimulus],
        order=0,
        action_request=ActionDescription(id="r2", action_name="fake", method_name="complete"),
        action_result=ActionResult(contents="", action_description_requests=proposals),
    )

    children = policy.decide([thinking], builder.build([thinking]))

    assert [c.action_request for c in children] == proposals
    assert [c.order for c in children] == [0, 1]
    assert all(c.parents == [thinking.id] for c in children)


def test_quit_attaches_null_and_moves_uncles():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(builder, contents="  QUIT ")  # whitespace and case forgiven
    root_id = stimulus.parents[0]
    uncle = builder.add_operand(
        parents=[builder._by_id[root_id]],
        order=1,
        action_request=ActionDescription(id="r2", action_name="telegram", method_name="input"),
    )
    builder.park_listener("telegram", uncle)

    children = policy.decide([stimulus], builder.build([stimulus]))

    assert children[0].action_request is not None
    assert children[0].action_request.action_name == "null"
    assert uncle.parents == [stimulus.id]  # moved to the tip


def test_thinking_text_delivers_to_origin_with_metadata():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(
        builder, channel="telegram", contents="question",
        metadata={"chat_id": "42", "username": "u"},
    )
    thinking = builder.add_operand(
        parents=[stimulus],
        order=0,
        action_request=ActionDescription(id="r2", action_name="fake", method_name="complete"),
        action_result=ActionResult(contents="answer"),
    )

    children = policy.decide([thinking], builder.build([thinking]))

    request = children[0].action_request
    assert request is not None
    assert request.action_name == "telegram"
    assert request.method_name == "send"
    assert request.method_parameters == {"text": "answer", "chat_id": "42"}


def test_empty_thinking_result_rearms_input():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(builder)
    thinking = builder.add_operand(
        parents=[stimulus],
        order=0,
        action_request=ActionDescription(id="r2", action_name="fake", method_name="complete"),
        action_result=ActionResult(contents=""),
    )

    children = policy.decide([thinking], builder.build([thinking]))

    request = children[0].action_request
    assert request is not None
    assert request.method_name == "input"
    assert request.action_name == "terminal"


def test_no_origin_channel_attaches_null():
    policy, builder = _utility_get_stack()
    root = builder.add_root()
    thinking = builder.add_operand(
        parents=[root],
        order=0,
        action_request=ActionDescription(id="r1", action_name="fake", method_name="complete"),
        action_result=ActionResult(contents="answer"),
    )

    children = policy.decide([thinking], builder.build([thinking]))

    assert children[0].action_request is not None
    assert children[0].action_request.action_name == "null"


def test_error_results_reprompt_thinking_instead_of_completing_turn():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(
        builder, channel="telegram", contents="thanks!", metadata={"chat_id": "42"}
    )
    failed_react = builder.add_operand(
        parents=[stimulus],
        order=0,
        action_request=ActionDescription(id="call_1", action_name="telegram", method_name="react",
                                         method_parameters={"chat_id": "42", "message_id": "7", "emoji": "🦖"}),
        action_result=ActionResult(contents="", error="not an allowed reaction"),
    )

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        children = policy.decide([failed_react], builder.build([failed_react]))

    assert children[0].action_request is not None
    assert children[0].action_request.method_name == "complete"  # re-prompt, not input
    api.assert_called_once_with("sendChatAction", {"chat_id": "42", "action": "typing"})


def test_delivery_only_frontier_completes_turn_and_rearms_input():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(
        builder, channel="telegram", contents="thanks!", metadata={"chat_id": "42"}
    )
    react = builder.add_operand(
        parents=[stimulus],
        order=0,
        action_request=ActionDescription(id="call_1", action_name="telegram", method_name="react",
                                         method_parameters={"chat_id": "42", "message_id": "7", "emoji": "👍"}),
        action_result=ActionResult(contents="👍"),
    )

    children = policy.decide([react], builder.build([react]))

    request = children[0].action_request
    assert request is not None
    assert request.method_name == "input"
    assert request.action_name == "telegram"


def test_effect_frontier_joins_into_one_thinking_child():
    policy, builder = _utility_get_stack()
    stimulus = _utility_get_stimulus(builder)
    # two resolved effect children of a proposing thinking operand — the
    # next thinking joins them: parents are exactly the results it consumes
    thinking = builder.add_operand(
        parents=[stimulus],
        order=0,
        action_request=ActionDescription(id="r2", action_name="fake", method_name="complete"),
        action_result=ActionResult(contents=""),
    )
    effect_a = builder.add_operand(
        parents=[thinking], order=0,
        action_request=ActionDescription(id="ca", action_name="terminal", method_name="beep"),
        action_result=ActionResult(contents="a"),
    )
    effect_b = builder.add_operand(
        parents=[thinking], order=1,
        action_request=ActionDescription(id="cb", action_name="terminal", method_name="beep"),
        action_result=ActionResult(contents="b"),
    )

    children = policy.decide([effect_a, effect_b], builder.build([effect_a, effect_b]))

    assert len(children) == 1
    join = children[0]
    assert join.action_request is not None
    assert join.action_request.method_name == "complete"
    assert join.parents == [effect_a.id, effect_b.id]
