"""Logging for the processes actant itself starts (a sandbox's host and its workers)."""

from __future__ import annotations

import logging


def log_to_stderr(level: int = logging.INFO) -> None:
    """Send actant's own records, ``level`` and up, to stderr, message only. For a process
    actant starts that configures no logging (Python drops INFO by default). The root logger
    is left alone; a second call, or a process that already gave ``actant`` a handler, keeps
    what it has."""
    package = logging.getLogger("actant")
    if package.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    package.addHandler(handler)
    package.setLevel(level)
    package.propagate = False
