from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from harness.action import Action

DEFAULT_SYSTEM_PROMPT = (
    "You are a local development assistant.\n"
    "Be concise.\n"
    "Use actions only when necessary."
)

DEFAULT_ACTIONS: dict[str, dict[str, Any]] = {"terminal": {}, "fake": {}}


@dataclass
class ThinkingActionConfig:
    model: str = "local"
    base_url: str = "http://localhost:8080/v1"


@dataclass
class TelegramActionConfig:
    token: str
    # Allow-list of chats: numeric chat ids or "@username" entries. Both match
    # incoming messages; only numeric ids can be send targets. Empty = accept all.
    chat_ids: list[str | int] = field(default_factory=list)


@dataclass
class LoggingConfig:
    env: str = "dev"
    log_file: str | None = None


@dataclass
class Config:
    actions: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict(DEFAULT_ACTIONS)
    )
    default_thinking_action: str = "fake"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def build_actions(config: Config) -> list["Action"]:
    """Build the configured actions, handing each its own config. NullAction
    is the only special case: it is always present, because the loop cannot
    terminate without it."""
    # Imported here: the action modules import this module for their configs.
    from actions.fake_thinking_action import FakeThinkingAction
    from actions.null_action import NullAction
    from actions.terminal_action import TerminalAction

    def _openai(cfg: dict[str, Any]) -> "Action":
        from actions.openai_thinking_action import OpenAIThinkingAction

        return OpenAIThinkingAction(ThinkingActionConfig(**cfg))

    def _telegram(cfg: dict[str, Any]) -> "Action":
        from actions.telegram_action import TelegramAction

        return TelegramAction(TelegramActionConfig(**cfg))

    builders = {
        "terminal": lambda cfg: TerminalAction(),
        "fake": lambda cfg: FakeThinkingAction(),
        "openai": _openai,
        "telegram": _telegram,
    }

    actions: list["Action"] = [NullAction()]
    for name, action_cfg in config.actions.items():
        if name not in builders:
            raise ValueError(
                f"Unknown action in config: {name!r} (expected one of {sorted(builders)})"
            )
        actions.append(builders[name](action_cfg or {}))
    return actions


def load_config(path: str = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    logging_raw = raw.get("logging", {})
    logging_config = LoggingConfig(
        env=logging_raw.get("env", "dev"),
        log_file=logging_raw.get("log_file"),
    )

    actions_raw = raw.get("actions") or dict(DEFAULT_ACTIONS)
    actions = {name: (cfg or {}) for name, cfg in actions_raw.items()}
    default_thinking_action = raw.get("default_thinking_action", "fake")
    return Config(
        actions=actions,
        default_thinking_action=default_thinking_action,
        system_prompt=raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        logging=logging_config,
    )
