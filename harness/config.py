from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "You are a local development assistant.\n"
    "Be concise.\n"
    "Use tools only when necessary."
)


@dataclass
class ProviderConfig:
    type: str
    model: str = "local"
    base_url: str = "http://localhost:8080/v1"


@dataclass
class LoggingConfig:
    env: str = "dev"
    log_file: str | None = None


@dataclass
class Config:
    default_session_id: str = "default"
    provider: ProviderConfig = field(default_factory=lambda: ProviderConfig(type="fake"))
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: dict = field(default_factory=dict)
    max_tool_rounds: int = 5
    data_dir: str = "data"
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    provider_raw = raw.get("provider", {})
    provider = ProviderConfig(
        type=provider_raw.get("type", "fake"),
        model=provider_raw.get("model", "local"),
        base_url=provider_raw.get("base_url", "http://localhost:8080/v1"),
    )

    logging_raw = raw.get("logging", {})
    logging_config = LoggingConfig(
        env=logging_raw.get("env", "dev"),
        log_file=logging_raw.get("log_file"),
    )

    return Config(
        default_session_id=raw.get("default_session_id", "default"),
        provider=provider,
        system_prompt=raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        tools=raw.get("tools", {}),
        max_tool_rounds=raw.get("max_tool_rounds", 5),
        data_dir=raw.get("data_dir", "data"),
        logging=logging_config,
    )
