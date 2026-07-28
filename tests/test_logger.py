import pytest

from harness.config import LoggingConfig
from harness.logger import new_id, setup_logging


def test_new_id():
    first, second = new_id(), new_id()
    assert len(first) == 8
    assert int(first, 16) >= 0  # hex
    assert first != second


def test_setup_logging(tmp_path):
    with pytest.raises(ValueError, match="log_file must be set"):
        setup_logging(LoggingConfig(env="production"))

    log_file = tmp_path / "nested" / "harness.log"
    setup_logging(LoggingConfig(env="production", log_file=str(log_file)))
    assert log_file.parent.exists()  # parent directories are created
