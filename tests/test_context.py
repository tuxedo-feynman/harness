from harness.context import ContextBuilder
from harness.messages import Message


def test_first_message_is_system_prompt():
    cb = ContextBuilder()
    msgs = cb.build("You are helpful.", [], "hello")
    assert msgs[0].role == "system"
    assert msgs[0].content == "You are helpful."


def test_latest_user_message_appended_last():
    cb = ContextBuilder()
    history = [Message(role="assistant", content="previous")]
    msgs = cb.build("sys", history, "new message")
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "new message"


def test_history_order_preserved():
    cb = ContextBuilder()
    history = [
        Message(role="user", content="first"),
        Message(role="assistant", content="second"),
    ]
    msgs = cb.build("sys", history, "third")
    assert msgs[1].content == "first"
    assert msgs[2].content == "second"
    assert msgs[3].content == "third"


def test_empty_history_works():
    cb = ContextBuilder()
    msgs = cb.build("sys", [], "hello")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"


def test_empty_system_prompt_is_omitted():
    # behavior: empty string → no system message added
    cb = ContextBuilder()
    msgs = cb.build("", [], "hello")
    assert len(msgs) == 1
    assert msgs[0].role == "user"


def test_does_not_mutate_input_history():
    cb = ContextBuilder()
    history = [Message(role="user", content="prior")]
    cb.build("sys", history, "new")
    assert len(history) == 1
