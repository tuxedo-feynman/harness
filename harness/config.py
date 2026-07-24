from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "You are a local development assistant.\n"
    "Be concise.\n"
    "Use actions only when necessary."
)


@dataclass
class ThinkingActionConfig:
    type: str  # "fake" | "openai"
    model: str = "local"
    base_url: str = "http://localhost:8080/v1"


@dataclass
class LoggingConfig:
    env: str = "dev"
    log_file: str | None = None


@dataclass
class Config:
    thinking_action: ThinkingActionConfig = field(
        default_factory=lambda: ThinkingActionConfig(type="fake")
    )
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    thinking_raw = raw.get("thinking_action", {})
    thinking = ThinkingActionConfig(
        type=thinking_raw.get("type", "fake"),
        model=thinking_raw.get("model", "local"),
        base_url=thinking_raw.get("base_url", "http://localhost:8080/v1"),
    )

    logging_raw = raw.get("logging", {})
    logging_config = LoggingConfig(
        env=logging_raw.get("env", "dev"),
        log_file=logging_raw.get("log_file"),
    )

    return Config(
        thinking_action=thinking,
        system_prompt=raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        logging=logging_config,
    )
