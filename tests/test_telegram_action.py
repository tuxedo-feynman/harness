import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from actions.telegram_action import TelegramAction
from harness.config import TelegramActionConfig
from harness.models import Context


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


def test_listen_returns_message_from_a_mocked_api():
    action = TelegramAction(TelegramActionConfig(token="test-token"))
    update = _utility_get_update("yo", 5, "u")
    context = Context(system_prompt="", history=[], available_actions={})

    with patch.object(TelegramAction, "_api", return_value={"result": [update]}) as api:
        result = action.run("listen", {}, context)

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
        result = action.run("listen", {}, context)

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
        result = action.run("listen", {}, context)

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


def test_run_rejects_bad_arguments_and_unknown_methods():
    action = TelegramAction(TelegramActionConfig(token="t"))
    context = Context(system_prompt="", history=[], available_actions={})
    with pytest.raises(ValueError, match="'text' must be a string"):
        action.run("send", {"chat_id": "42"}, context)
    with pytest.raises(ValueError, match="'chat_id' is required"):
        action.run("send", {"text": "hi"}, context)
    with pytest.raises(ValueError, match="Unknown method: 'beep'"):
        action.run("beep", {}, context)


def test_api_raises_when_telegram_says_not_ok():
    action = TelegramAction(TelegramActionConfig(token="t"))
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"ok": false, "description": "bad token"}'

    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="Telegram API getMe failed"):
            action._api("getMe", {})
