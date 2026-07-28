from unittest.mock import Mock

import pytest

from actions.claude_thinking_action import ClaudeThinkingAction
from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action_directory import ActionDirectory
from harness.config import ClaudeActionConfig
from harness.context import ContextBuilder
from harness.models import ActionDescription, ActionResult, Context


def _utility_get_builder() -> ContextBuilder:
    directory = ActionDirectory([NullAction(), TerminalAction(), FakeThinkingAction()])
    return ContextBuilder(system_prompt="sys", action_directory=directory)


def _utility_get_text_block(text: str) -> Mock:
    return Mock(type="text", text=text)


def _utility_get_tool_use_block(id: str, flat_name: str, input: dict) -> Mock:
    block = Mock(type="tool_use", id=id, input=input)
    block.name = flat_name  # 'name' is reserved in Mock(); set it after
    return block


def test_build_tools_advertises_effect_methods_only():
    builder = _utility_get_builder()
    action = ClaudeThinkingAction(ClaudeActionConfig())

    tools, lookup = action._build_tools(builder.build(builder.add_root()))

    names = {tool["name"] for tool in tools}
    assert names == {"terminal_send", "terminal_listen"}  # fake and null excluded
    assert lookup == {
        "terminal_send": ("terminal", "send"),
        "terminal_listen": ("terminal", "listen"),
    }
    assert all("input_schema" in tool for tool in tools)  # anthropic schema shape


def test_to_claude_messages_reconstructs_a_full_turn():
    builder = _utility_get_builder()
    root = builder.add_root()
    listener = builder.add_operand(
        parent=root,
        action_requests=[ActionDescription(id="r1", action_name="terminal", method_name="listen")],
        action_results=[ActionResult(contents="question")],
    )
    proposal = ActionDescription(
        id="toolu_9", action_name="terminal", method_name="send", method_parameters={"text": "hi"}
    )
    thinking = builder.add_operand(
        parent=listener,
        action_requests=[ActionDescription(id="r2", action_name="fake", method_name="complete")],
        action_results=[
            ActionResult(
                contents="thinking aloud",
                metadata={"thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "sig"}]},
                action_description_requests=[proposal],
            )
        ],
    )
    effect = builder.add_operand(
        parent=thinking,
        action_requests=[proposal],  # same ActionDescription: id pairs the tool result
        action_results=[ActionResult(contents="hi")],
    )
    # a policy-synthesized delivery (never proposed by the model) is not model-facing
    synthesized = builder.add_operand(
        parent=effect,
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

    messages = ClaudeThinkingAction(ClaudeActionConfig())._to_claude_messages(builder.build(null))

    assert messages == [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},  # echoed unchanged
                {"type": "text", "text": "thinking aloud"},
                {"type": "tool_use", "id": "toolu_9", "name": "terminal_send", "input": {"text": "hi"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "hi"}],
        },
    ]


def test_parse_response_converts_a_mocked_api_response():
    thinking_block = Mock(type="thinking")
    thinking_block.model_dump.return_value = {"type": "thinking", "thinking": "", "signature": "sig"}
    response = Mock(
        stop_reason="tool_use",
        content=[
            thinking_block,
            _utility_get_text_block("answer"),
            _utility_get_tool_use_block("toolu_1", "terminal_send", {"text": "hi"}),
        ],
    )

    action = ClaudeThinkingAction(ClaudeActionConfig())
    result = action._parse_response(response, {"terminal_send": ("terminal", "send")})

    assert result.contents == "answer"
    assert result.metadata == {
        "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "sig"}]
    }
    proposal = result.action_description_requests[0]
    assert proposal.id == "toolu_1"
    assert proposal.action_name == "terminal"
    assert proposal.method_name == "send"
    assert proposal.method_parameters == {"text": "hi"}


def test_parse_response_raises_on_refusal_and_unknown_tools():
    action = ClaudeThinkingAction(ClaudeActionConfig())

    refusal = Mock(stop_reason="refusal", stop_details=Mock(category="cyber"), content=[])
    with pytest.raises(RuntimeError, match="Claude declined"):
        action._parse_response(refusal, {})

    hallucinated = Mock(
        stop_reason="tool_use",
        content=[_utility_get_tool_use_block("toolu_2", "made_up_tool", {})],
    )
    with pytest.raises(ValueError, match="unknown tool"):
        action._parse_response(hallucinated, {})


def test_run_rejects_unknown_methods():
    action = ClaudeThinkingAction(ClaudeActionConfig())
    context = Context(system_prompt="", history=[], available_actions={})
    with pytest.raises(ValueError, match="Unknown method: 'beep'"):
        action.run("beep", {}, context)
