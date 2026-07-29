from unittest.mock import Mock

import pytest

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.openai_thinking_action import OpenAIThinkingAction
from actions.terminal_action import TerminalAction
from hyh.action_directory import ActionDirectory
from hyh.config import ThinkingActionConfig
from hyh.context import ContextBuilder
from hyh.models import ActionDescription, ActionResult, Context


def test_converts_user_input_history_to_openai_messages():
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    root = builder.add_root()
    listener = builder.add_operand(
        parent=root,
        action_requests=[
            ActionDescription(id="ad-1", action_name="terminal", method_name="listen")
        ],
        action_results=[ActionResult(contents="hello")],
    )

    action = OpenAIThinkingAction(ThinkingActionConfig())
    messages = action._to_openai_messages(builder.build(listener))

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]


def test_parse_response_converts_a_mocked_api_response():
    function = Mock(arguments='{"text": "hi"}')
    function.name = "terminal_send"  # 'name' is reserved in Mock(); set it after
    response = Mock()
    response.choices = [Mock(message=Mock(content="answer", tool_calls=[Mock(id="call_1", function=function)]))]

    action = OpenAIThinkingAction(ThinkingActionConfig())
    result = action._parse_response(response, {"terminal_send": ("terminal", "send")})

    assert result.contents == "answer"
    proposal = result.action_description_requests[0]
    assert proposal.id == "call_1"
    assert proposal.action_name == "terminal"
    assert proposal.method_name == "send"
    assert proposal.method_parameters == {"text": "hi"}


def _utility_get_response(content, tool_calls):
    return Mock(choices=[Mock(message=Mock(content=content, tool_calls=tool_calls))])


def _utility_get_tool_call(id: str, flat_name: str, arguments: str):
    function = Mock(arguments=arguments)
    function.name = flat_name  # 'name' is reserved in Mock(); set it after
    return Mock(id=id, function=function)


def test_parse_response_raises_on_unknown_tools_and_bad_arguments():
    action = OpenAIThinkingAction(ThinkingActionConfig())
    lookup = {"terminal_send": ("terminal", "send")}

    with pytest.raises(ValueError, match="unknown tool"):
        action._parse_response(
            _utility_get_response(None, [_utility_get_tool_call("c1", "hallucinated", "{}")]),
            lookup,
        )
    with pytest.raises(ValueError, match="Bad tool arguments"):
        action._parse_response(
            _utility_get_response(None, [_utility_get_tool_call("c2", "terminal_send", "not json")]),
            lookup,
        )


def test_render_empty_describes_the_recorded_event():
    voice = ActionResult(contents="", metadata={"message": {"voice": {"duration": 3}}})
    rendered = OpenAIThinkingAction._render_empty(voice)
    assert "voice" in rendered
    assert "isn't supported" in rendered

    blank = ActionResult(contents="")
    assert OpenAIThinkingAction._render_empty(blank) == "[the user sent an empty message]"


def test_run_rejects_unknown_methods():
    action = OpenAIThinkingAction(ThinkingActionConfig())
    context = Context(system_prompt="", history=[], available_actions={})
    with pytest.raises(ValueError, match="Unknown method: 'beep'"):
        action.run("beep", {}, context)


def test_build_tools_advertises_effect_methods_only():
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    action = OpenAIThinkingAction(ThinkingActionConfig())

    tools, lookup = action._build_tools(builder.build(builder.add_root()))

    names = {tool["function"]["name"] for tool in tools}
    # fake and null excluded; listen and typing are the harness's verbs,
    # never proposable by the model
    assert names == {"terminal_send"}
    assert lookup == {"terminal_send": ("terminal", "send")}


def test_to_openai_messages_reconstructs_a_full_turn():
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    builder = ContextBuilder(system_prompt="sys", action_directory=directory)
    root = builder.add_root()
    listener = builder.add_operand(
        parent=root,
        action_requests=[ActionDescription(id="r1", action_name="terminal", method_name="listen")],
        action_results=[ActionResult(contents="question")],
    )
    proposal = ActionDescription(
        id="call_9", action_name="terminal", method_name="send", method_parameters={"text": "hi"}
    )
    thinking = builder.add_operand(
        parent=listener,
        action_requests=[ActionDescription(id="r2", action_name="fake", method_name="complete")],
        action_results=[
            ActionResult(contents="thinking aloud", action_description_requests=[proposal])
        ],
    )
    effect = builder.add_operand(
        parent=thinking,
        action_requests=[proposal],  # same ActionDescription: id pairs the tool result
        action_results=[ActionResult(contents="hi")],
    )
    # a delivery relaying the thinking result's own text is already emitted — skipped
    relay = builder.add_operand(
        parent=effect,
        action_requests=[
            ActionDescription(id="relay", action_name="terminal", method_name="send",
                              method_parameters={"text": "thinking aloud"})
        ],
        action_results=[ActionResult(contents="thinking aloud")],
    )
    # a policy-AUTHORED send (e.g. a canned reply) is a new assistant utterance
    synthesized = builder.add_operand(
        parent=relay,
        action_requests=[
            ActionDescription(id="synth", action_name="terminal", method_name="send",
                              method_parameters={"text": "bye"})
        ],
        action_results=[ActionResult(contents="bye")],
    )
    null = builder.add_operand(
        parent=synthesized,
        action_requests=[ActionDescription(id="r5", action_name="null", method_name="terminate")],
        action_results=[ActionResult(contents="")],
    )

    messages = OpenAIThinkingAction(ThinkingActionConfig())._to_openai_messages(
        builder.build(null)
    )

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "thinking aloud",
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "terminal_send", "arguments": '{"text": "hi"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_9", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ]
