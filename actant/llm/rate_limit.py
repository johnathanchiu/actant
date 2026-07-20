"""Per-(provider, model) rate limiter with sliding-window TPM/RPM accounting.

Used by LLM providers to wait BEFORE sending a request if it would
exceed the per-minute budget, so we don't burn credits on retries
after a 429.

Estimates aren't always right (especially for reasoning models that
spend tokens internally). On a miss the provider catches
``RateLimitError`` once, sleeps the retry-after, and retries. The
limiter is provider-agnostic — wire it in from the application
layer and pass a single shared instance per ``(provider, model)``
pair to every caller for that model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_WINDOW_SECS = 60.0


@dataclass
class RateLimitConfig:
    tokens_per_minute: int
    requests_per_minute: int


class RateLimiter:
    """Sliding-window rate limiter for tokens/min + requests/min."""

    def __init__(self, config: RateLimitConfig, *, name: str = "") -> None:
        self._config = config
        self._name = name
        self._token_window: deque[tuple[float, int]] = deque()
        self._request_window: deque[float] = deque()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def reserve(self, estimated_tokens: int) -> AsyncIterator["_Reservation"]:
        """Block until the request fits in the budget, then yield a
        reservation that the caller updates with actual usage on exit.
        """
        await self._acquire(estimated_tokens)
        reservation = _Reservation(self, estimated_tokens)
        try:
            yield reservation
        finally:
            reservation._commit()

    async def _acquire(self, estimated: int) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._evict(now)
                tokens_in_window = sum(t for _, t in self._token_window)
                requests_in_window = len(self._request_window)
                fits_tokens = (
                    tokens_in_window + estimated <= self._config.tokens_per_minute
                )
                fits_requests = requests_in_window < self._config.requests_per_minute
                if fits_tokens and fits_requests:
                    self._token_window.append((now, estimated))
                    self._request_window.append(now)
                    return
                wait_secs = self._compute_wait(
                    now, estimated, tokens_in_window, requests_in_window
                )
                logger.info(
                    "actant.rate_limit.waiting name=%s wait_secs=%.2f "
                    "tokens_in_window=%d requests_in_window=%d "
                    "estimated=%d limit_tpm=%d limit_rpm=%d",
                    self._name,
                    wait_secs,
                    tokens_in_window,
                    requests_in_window,
                    estimated,
                    self._config.tokens_per_minute,
                    self._config.requests_per_minute,
                )
                await asyncio.sleep(wait_secs)

    def _compute_wait(
        self,
        now: float,
        estimated: int,
        tokens_in_window: int,
        requests_in_window: int,
    ) -> float:
        # Find the earliest entry whose expiry would free enough budget.
        target = now
        if (
            tokens_in_window + estimated > self._config.tokens_per_minute
            and self._token_window
        ):
            need = (tokens_in_window + estimated) - self._config.tokens_per_minute
            cumulative = 0
            for ts, t in self._token_window:
                cumulative += t
                if cumulative >= need:
                    target = max(target, ts)
                    break
        if (
            requests_in_window >= self._config.requests_per_minute
            and self._request_window
        ):
            target = max(target, self._request_window[0])
        # Wait until the target entry is older than the window, plus
        # a small slack so we re-check on the right side of the edge.
        return max(0.05, _WINDOW_SECS - (now - target) + 0.1)

    def _evict(self, now: float) -> None:
        cutoff = now - _WINDOW_SECS
        while self._token_window and self._token_window[0][0] < cutoff:
            self._token_window.popleft()
        while self._request_window and self._request_window[0] < cutoff:
            self._request_window.popleft()

    def _record_actual(self, estimated: int, actual: int) -> None:
        # Replace the most recent entry's estimate with the actual
        # token count. Cheap fix-up; lock not required because we
        # only mutate the tail.
        del estimated
        if self._token_window:
            ts, _ = self._token_window[-1]
            self._token_window[-1] = (ts, actual)


@dataclass
class _Reservation:
    _limiter: RateLimiter
    _estimated: int
    _actual: int | None = None

    def record_actual(self, actual: int) -> None:
        self._actual = actual

    def _commit(self) -> None:
        if self._actual is not None and self._actual != self._estimated:
            self._limiter._record_actual(self._estimated, self._actual)


__all__ = ["RateLimitConfig", "RateLimiter"]
