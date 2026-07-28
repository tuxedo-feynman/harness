import pytest

from actions.telegram_action import TelegramAction
from hyh.config import Config, build_actions, load_config


def test_load_config():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("no/such/config.yaml")


def test_load_config_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")

    config = load_config(str(path))

    assert config.actions == {"terminal": {}, "fake": {}}
    assert config.default_thinking_action == "fake"
    assert "assistant" in config.system_prompt
    assert config.logging.env == "dev"
    assert config.logging.log_file is None


def test_load_config_parses_all_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "actions:\n"
        "  terminal:\n"
        "  telegram:\n"
        "    token: tok\n"
        '    chat_ids: ["@u", 1]\n'
        "default_thinking_action: openai\n"
        "system_prompt: custom prompt\n"
        "logging:\n"
        "  env: production\n"
        "  log_file: /x.log\n"
    )

    config = load_config(str(path))

    assert config.actions == {"terminal": {}, "telegram": {"token": "tok", "chat_ids": ["@u", 1]}}
    assert config.default_thinking_action == "openai"
    assert config.system_prompt == "custom prompt"
    assert config.logging.env == "production"
    assert config.logging.log_file == "/x.log"


def test_build_actions():
    actions = build_actions(Config(actions={"terminal": {}, "fake": {}}))
    assert [a.name for a in actions] == ["null", "terminal", "fake"]  # null always present

    with pytest.raises(ValueError, match="Unknown action in config: 'bogus'"):
        build_actions(Config(actions={"bogus": {}}))


def test_build_actions_hands_each_action_its_config():
    actions = build_actions(
        Config(actions={"telegram": {"token": "tok", "chat_ids": [1]}})
    )
    telegram = next(a for a in actions if a.name == "telegram")
    assert isinstance(telegram, TelegramAction)
    assert telegram.config.token == "tok"
    assert telegram.config.chat_ids == [1]
