from datetime import datetime, timezone

import pytest

from hyh.config import LoggingConfig
from hyh.logger import new_id, record, setup_logging
from hyh.models import Operand


def test_new_id():
    first, second = new_id(), new_id()
    assert len(first) == 8
    assert int(first, 16) >= 0  # hex
    assert first != second


def test_setup_logging(tmp_path):
    with pytest.raises(ValueError, match="log_file must be set"):
        setup_logging(LoggingConfig(env="production"))

    log_file = tmp_path / "nested" / "hyh.log"
    setup_logging(LoggingConfig(env="production", log_file=str(log_file)))
    assert log_file.parent.exists()  # parent directories are created


def test_record_appends_to_the_operand_log_and_echoes_to_the_process_log(caplog):
    operand = Operand(
        id="op-1",
        created_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        parents=[],
        order=0,
    )

    with caplog.at_level("INFO", logger="hyh.logger"):
        record(operand, "policy decision=attach_thinking reason=test")
        record(operand, "executed duration=0.001s")

    # append-only, in order, timestamped
    assert [entry.message for entry in operand.logs] == [
        "policy decision=attach_thinking reason=test",
        "executed duration=0.001s",
    ]
    assert all(entry.at.tzinfo is not None for entry in operand.logs)
    # the process log is a superset: each entry echoed with the operand id
    assert "operand=op-1 policy decision=attach_thinking reason=test" in caplog.text
