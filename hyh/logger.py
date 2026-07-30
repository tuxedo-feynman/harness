import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from hyh.config import LoggingConfig
from hyh.models import LogEntry, Operand

log = logging.getLogger(__name__)


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


def record(operand: Operand, message: str) -> None:
    """Append a timestamped entry to the operand's lifetime log and echo it
    to the process log. The operand log is the source for operand-scoped
    events; the process log is a superset (transport lifecycle, warnings,
    and everything else with no operand to belong to)."""
    operand.logs.append(LogEntry(at=datetime.now(timezone.utc), message=message))
    log.info(f"operand={operand.id} {message}")
