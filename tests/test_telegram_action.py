from actions.telegram_action import TelegramAction
from harness.config import TelegramActionConfig


def test_accept_filters_by_username_or_numeric_chat_id():
    action = TelegramAction(
        TelegramActionConfig(token="test-token", chat_ids=["@MrFantasticZero"])
    )
    allowed = {
        "update_id": 7,
        "message": {"text": "hello", "chat": {"id": 42, "username": "mrfantasticzero"}},
    }
    intruder = {
        "update_id": 8,
        "message": {"text": "intruder", "chat": {"id": 99, "username": "eve"}},
    }

    result = action._accept(allowed)
    assert result is not None
    assert result.contents == "hello"
    assert result.metadata == {"chat_id": "42", "username": "mrfantasticzero"}
    assert action._accept(intruder) is None
