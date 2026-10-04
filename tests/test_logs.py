"""``log_to_stderr`` gives actant's loggers one stderr handler and leaves the root alone."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from actant.logs import log_to_stderr


@pytest.fixture
def package() -> Iterator[logging.Logger]:
    logger = logging.getLogger("actant")
    saved = logger.handlers[:], logger.level, logger.propagate
    logger.handlers.clear()
    yield logger
    logger.handlers[:], logger.level, logger.propagate = saved


def test_actant_records_reach_stderr_once_and_the_root_is_untouched(
    package: logging.Logger, capfd: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    before = root.handlers[:], root.level
    log_to_stderr()
    log_to_stderr()
    logging.getLogger("actant.sandbox.processes").info("worker 1 started for p")
    logging.getLogger("elsewhere").info("not ours")
    assert capfd.readouterr().err == "worker 1 started for p\n"
    assert len(package.handlers) == 1
    assert (root.handlers, root.level) == before
