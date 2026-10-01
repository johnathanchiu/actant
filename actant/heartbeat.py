"""Heartbeats for a Temporal activity while slow work runs inside it.

A Temporal activity with a heartbeat timeout is retried as lost when it stops beating.
Work that waits minutes on something else (a sandbox opening, a pull, a push, a call
into a service host) beats on a timer while it runs, so the caller's activity is not
mistaken for a dead worker. Outside an activity it does nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from temporalio import activity

#: How often a beat is sent: well inside any heartbeat timeout worth setting.
HEARTBEAT_EVERY_S = 30.0


@contextlib.asynccontextmanager
async def heartbeating(every_s: float | None = None) -> AsyncIterator[None]:
    """Heartbeat the current activity every ``every_s`` (``HEARTBEAT_EVERY_S``) while the
    body runs."""
    if not activity.in_activity():
        yield
        return

    async def beat() -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_EVERY_S if every_s is None else every_s)
            activity.heartbeat()

    ticker = asyncio.create_task(beat())
    try:
        yield
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
