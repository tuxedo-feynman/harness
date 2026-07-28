import logging
import uuid
from pathlib import Path

from hyh.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    if config.env == "production":
        if not config.log_file:
            raise ValueError("log_file must be set in config when env=production")
        log_file = config.log_file
    else:
        log_file = config.log_file or "/tmp/hyh.log"

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    formatter = logging.Formatter(fmt)

    logger = logging.getLogger("hyh")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def new_id() -> str:
    return uuid.uuid4().hex[:8]
