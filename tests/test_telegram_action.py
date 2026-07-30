import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from actions.telegram_action import TelegramAction
from hyh.config import TelegramActionConfig
from hyh.models import Context


def _utility_get_update(text: str, chat_id: int, username: str, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {"text": text, "chat": {"id": chat_id, "username": username}},
    }


def test_accept_filters_by_username_or_numeric_chat_id():
    action = TelegramAction(
        TelegramActionConfig(token="test-token", chat_ids=["@MrFantasticZero"])
    )
    allowed = _utility_get_update("hello", 42, "mrfantasticzero", update_id=7)
    intruder = _utility_get_update("intruder", 99, "eve", update_id=8)

    result = action._accept(allowed)
    assert result is not None
    assert result.contents == "hello"
    assert result.metadata["chat_id"] == "42"
    assert result.metadata["username"] == "mrfantasticzero"
    assert result.metadata["message"]["text"] == "hello"  # the full event is recorded
    assert action._accept(intruder) is None


def test_token_resolves_from_the_environment_or_fails_fast():
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "env-token"}):
        action = TelegramAction(TelegramActionConfig(token=None))
    assert action._token == "env-token"

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="telegram token missing"):
            TelegramAction(TelegramActionConfig(token=None))


def test_listen_returns_message_from_a_mocked_api():
    action = TelegramAction(TelegramActionConfig(token="test-token"))
    update = _utility_get_update("yo", 5, "u")
    context = Context(system_prompt="", history=[], available_actions={})

    with patch.object(TelegramAction, "_api", return_value={"result": [update]}) as api:
        result = action.run("input", {}, context)

    assert result.contents == "yo"
    assert api.call_count == 2  # the poll, then the immediate ack
    assert action._offset == 2  # next poll starts after the consumed update


def test_accept_edge_cases():
    unfiltered = TelegramAction(TelegramActionConfig(token="t"))
    accepted = unfiltered._accept(_utility_get_update("hi", 7, "anyone"))
    assert accepted is not None  # empty allow-list accepts everyone
    assert accepted.metadata["chat_id"] == "7"

    by_id = TelegramAction(TelegramActionConfig(token="t", chat_ids=[7]))
    assert by_id._accept(_utility_get_update("hi", 7, "anyone")) is not None

    assert unfiltered._accept({"update_id": 1}) is None  # no chat: nowhere to reply

    # unsupported content from an allowed chat becomes an empty-content
    # stimulus with the full message recorded, so Policy can reply honestly
    # and the thinking adapters can describe the event
    voice = unfiltered._accept(
        {"update_id": 2, "message": {"chat": {"id": 7, "username": "anyone"}, "voice": {"duration": 3}}}
    )
    assert voice is not None
    assert voice.contents == ""
    assert voice.metadata["message"]["voice"] == {"duration": 3}

    # a caption on media is real user text — promoted to contents
    captioned = unfiltered._accept(
        {"update_id": 3, "message": {"chat": {"id": 7}, "photo": [{}], "caption": "look at this"}}
    )
    assert captioned is not None
    assert captioned.contents == "look at this"

    # but unsupported content from a non-allowed chat is still dropped
    assert by_id._accept({"update_id": 4, "message": {"chat": {"id": 99}, "voice": {}}}) is None


def test_listen_skips_filtered_updates_within_one_poll():
    action = TelegramAction(TelegramActionConfig(token="t", chat_ids=[42]))
    context = Context(system_prompt="", history=[], available_actions={})
    updates = [
        _utility_get_update("intruder", 99, "eve", update_id=10),
        _utility_get_update("hello", 42, "adam", update_id=11),
    ]

    with patch.object(TelegramAction, "_api", return_value={"result": updates}):
        result = action.run("input", {}, context)

    assert result.contents == "hello"
    assert action._offset == 12  # past both updates


def test_listen_reconnects_after_transient_connection_errors():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})
    update = _utility_get_update("back online", 5, "u")

    with (
        patch.object(
            TelegramAction,
            "_api",
            side_effect=[
                urllib.error.URLError("connection reset by peer"),  # poll dies
                {"result": [update]},  # reconnected poll delivers
                {"ok": True},  # the immediate ack
            ],
        ) as api,
        patch("actions.telegram_action.time.sleep") as sleep,
    ):
        result = action.run("input", {}, context)

    assert result.contents == "back online"
    sleep.assert_called_once_with(5)
    assert api.call_count == 3


def test_send_retries_transient_errors_then_crashes():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})

    # one reset, then success — the send survives
    with (
        patch.object(TelegramAction, "_api",
                     side_effect=[urllib.error.URLError("reset"), {"ok": True}]) as api,
        patch("actions.telegram_action.time.sleep") as sleep,
    ):
        result = action.run("send", {"text": "hi", "chat_id": "42"}, context)
    assert result.contents == "hi"
    assert api.call_count == 2
    sleep.assert_called_once_with(5)

    # persistent failure — crashes after the attempt budget
    with (
        patch.object(TelegramAction, "_api", side_effect=urllib.error.URLError("down")) as api,
        patch("actions.telegram_action.time.sleep"),
    ):
        with pytest.raises(urllib.error.URLError):
            action.run("send", {"text": "hi", "chat_id": "42"}, context)
    assert api.call_count == 3


def test_send_posts_the_message():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        result = action.run("send", {"text": "hi", "chat_id": "42"}, context)

    api.assert_called_once_with("sendMessage", {"chat_id": "42", "text": "hi"})
    assert result.contents == "hi"


def test_send_splits_long_text_into_consecutive_messages():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})
    text = "a" * 4000 + "\n\n" + "b" * 4000

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        result = action.run("send", {"text": text, "chat_id": "42"}, context)

    assert result.contents == text
    sent = [call.args[1]["text"] for call in api.call_args_list]
    assert sent == ["a" * 4000 + "\n\n", "b" * 4000]


def test_split_message_prefers_readable_boundaries():
    from actions.telegram_action import _split_message

    # sentence fallback when there are no newlines
    sentences = ("word " * 99 + "end. ") * 12  # 6000 chars of 500-char sentences
    chunks = _split_message(sentences, 4096)
    assert all(len(c) <= 4096 for c in chunks)
    assert chunks[0].endswith("end. ")

    # rightmost match within the sentence tier wins, across scripts
    text = "x" * 2000 + ". " + "y" * 1000 + "！" + "z" * 3000
    chunks = _split_message(text, 4096)
    assert chunks[0].endswith("！")

    # spaceless CJK-style text splits after 。 rather than mid-word
    cjk = ("汉" * 500 + "。") * 12  # 6012 chars, no spaces or newlines
    chunks = _split_message(cjk, 4096)
    assert all(len(c) <= 4096 for c in chunks)
    assert chunks[0].endswith("。")

    # one unbroken token still hard-cuts at the limit
    assert _split_message("a" * 5000, 4096) == ["a" * 4096, "a" * 904]

    # short text passes through untouched
    assert _split_message("hi", 4096) == ["hi"]

    # the limit is a parameter, not telegram's constant
    assert _split_message("one two three", 5) == ["one ", "two ", "three"]


def test_send_threads_reply_on_first_chunk_only():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})
    text = "a" * 4000 + "\n\n" + "b" * 4000

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        action.run("send", {"text": text, "chat_id": "42", "reply_to_message_id": "7"}, context)

    first, second = [call.args[1] for call in api.call_args_list]
    assert first["reply_parameters"] == {"message_id": 7}
    assert "reply_parameters" not in second


def test_typing_sends_chat_action():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        result = action.run("typing", {"chat_id": "42"}, context)

    api.assert_called_once_with("sendChatAction", {"chat_id": "42", "action": "typing"})
    assert result.contents == ""


def test_react_posts_reaction_and_rejects_unknown_emoji():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        result = action.run("react", {"chat_id": "42", "message_id": "7", "emoji": "👍"}, context)

    api.assert_called_once_with(
        "setMessageReaction",
        {"chat_id": "42", "message_id": 7, "reaction": [{"type": "emoji", "emoji": "👍"}]},
    )
    assert result.contents == "👍"

    # a hallucinated emoji is LLM-fixable: an error result, not a crash
    with patch.object(TelegramAction, "_api") as api:
        result = action.run("react", {"chat_id": "42", "message_id": "7", "emoji": "🦖"}, context)
    api.assert_not_called()
    assert result.error is not None and "🦖" in result.error


def test_poll_posts_question_and_wrapped_options():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})

    with patch.object(TelegramAction, "_api", return_value={"ok": True}) as api:
        result = action.run(
            "poll", {"chat_id": "42", "question": "Lunch?", "options": ["yes", "no"]}, context
        )

    api.assert_called_once_with(
        "sendPoll",
        {
            "chat_id": "42",
            "question": "Lunch?",
            "options": [{"text": "yes"}, {"text": "no"}],
            "is_anonymous": False,
        },
    )
    assert result.contents == "Lunch?"

    with patch.object(TelegramAction, "_api") as api:
        result = action.run("poll", {"chat_id": "42", "question": "?", "options": ["only"]}, context)
    api.assert_not_called()
    assert result.error is not None and "2 to 10" in result.error


def test_accept_poll_answers_with_allow_list():
    action = TelegramAction(TelegramActionConfig(token="t", chat_ids=["@MrFantasticZero"]))
    vote = {
        "update_id": 9,
        "poll_answer": {
            "poll_id": "p1",
            "user": {"id": 42, "username": "mrfantasticzero"},
            "option_ids": [1],
        },
    }

    result = action._accept(vote)
    assert result is not None
    assert result.contents == "[poll vote: option_ids=[1]]"
    assert result.metadata["chat_id"] == "42"
    assert result.metadata["poll_answer"]["poll_id"] == "p1"  # the full event is recorded

    intruder = {
        "update_id": 10,
        "poll_answer": {"poll_id": "p1", "user": {"id": 99, "username": "eve"}, "option_ids": [0]},
    }
    assert action._accept(intruder) is None


def test_api_crashes_on_client_errors_and_retries_server_errors():
    import email.message
    import io

    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})

    def _utility_get_http_error(code: int, description: str) -> urllib.error.HTTPError:
        body = ('{"ok": false, "description": "%s"}' % description).encode()
        return urllib.error.HTTPError("url", code, "boom", email.message.Message(), io.BytesIO(body))

    # 4xx is permanent: no retries, and the response body names the cause
    with patch("urllib.request.urlopen", side_effect=_utility_get_http_error(400, "message is too long")):
        with pytest.raises(RuntimeError, match="HTTP 400.*message is too long"):
            action.run("send", {"text": "hi", "chat_id": "42"}, context)

    # 5xx stays in the transient class: retried, then succeeds
    ok = MagicMock()
    ok.__enter__.return_value.read.return_value = b'{"ok": true}'
    with (
        patch(
            "urllib.request.urlopen",
            side_effect=[_utility_get_http_error(502, "bad gateway"), ok],
        ),
        patch("actions.telegram_action.time.sleep"),
    ):
        result = action.run("send", {"text": "hi", "chat_id": "42"}, context)
    assert result.contents == "hi"
    # bad arguments are LLM-fixable: error results, not crashes
    assert action.run("send", {"chat_id": "42"}, context).error == "'text' must be a string"
    assert action.run("send", {"text": "hi"}, context).error == "'chat_id' is required"
    # an unknown method is a vocabulary bug, not an LLM slip — still raises
    with pytest.raises(ValueError, match="Unknown method: 'beep'"):
        action.run("beep", {}, context)


def test_api_raises_when_telegram_says_not_ok():
    action = TelegramAction(TelegramActionConfig(token="t"))
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"ok": false, "description": "bad token"}'

    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="Telegram API getMe failed"):
            action._api("getMe", {})
