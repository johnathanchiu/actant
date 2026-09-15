"""One sandbox per thread, for the life of this worker process.

Process-scoped like :class:`~actant.runtime.coordinator.SubThreadRegistry`:
a live handle is cached here, and the sandbox id is persisted on the thread
so another worker (or this one after a restart) reattaches instead of
opening a second sandbox over the same files.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass

from actant.runtime.interfaces.stores import ThreadStore
from actant.sandbox.base import Sandbox, SandboxProvider, SandboxSpec

# How long a verified handle is served without asking the backend again. A
# backend reclaims a sandbox after minutes idle, so one verified seconds ago
# was not reclaimed for idleness; attaching on every call instead put a Modal
# poll in front of every tool call.
VERIFIED_FOR_S = 5.0


@dataclass
class _Verified:
    sandbox: Sandbox
    at: float


class SandboxRegistry:
    def __init__(self, providers: Mapping[str, SandboxProvider], threads: ThreadStore) -> None:
        self._providers = dict(providers)
        self._threads = threads
        self._live: dict[tuple[str, str], _Verified] = {}
        # One resolve per thread at a time, shared by every caller waiting on it.
        self._resolving: dict[tuple[str, str], asyncio.Task[Sandbox]] = {}

    def provider(self, backend: str) -> SandboxProvider:
        try:
            return self._providers[backend]
        except KeyError:
            raise KeyError(
                f"no sandbox provider registered for backend {backend!r}; "
                f"known: {sorted(self._providers)}"
            ) from None

    async def for_thread(self, spec: SandboxSpec, agent_id: str, thread_id: str) -> Sandbox:
        """The thread's sandbox, reattached by its persisted id or newly opened.

        A handle verified within :data:`VERIFIED_FOR_S` is returned as is; past
        that the backend is asked again, so a sandbox it reclaimed is noticed
        and replaced. Concurrent callers for one thread share a single resolve;
        callers for different threads never wait on each other.
        """
        key = (agent_id, thread_id)
        verified = self._live.get(key)
        if verified is not None and time.monotonic() - verified.at < VERIFIED_FOR_S:
            return verified.sandbox
        task = self._resolving.get(key)
        if task is None:
            task = asyncio.ensure_future(self._resolve(spec, agent_id, thread_id))
            self._resolving[key] = task
            task.add_done_callback(lambda done: self._forget_resolve(key, done))
        # Shielded: one caller's cancellation must not fail the others sharing it.
        return await asyncio.shield(task)

    def _forget_resolve(self, key: tuple[str, str], done: asyncio.Task[Sandbox]) -> None:
        if self._resolving.get(key) is done:
            del self._resolving[key]

    async def _resolve(self, spec: SandboxSpec, agent_id: str, thread_id: str) -> Sandbox:
        provider = self.provider(spec.backend)
        while True:
            # The persisted id, not a caller's copy: another worker may have replaced it.
            sandbox_id = (await self._threads.get_or_create(agent_id, thread_id)).sandbox_id
            sandbox = None
            if sandbox_id:
                with contextlib.suppress(KeyError):
                    sandbox = await provider.attach(spec, sandbox_id)
            if sandbox is None:
                opened = await provider.open(spec, agent_id=agent_id, thread_id=thread_id)
                winner = await self._threads.claim_sandbox(
                    agent_id, thread_id, expected=sandbox_id, sandbox_id=opened.id
                )
                if winner != opened.id:
                    # Another worker claimed the thread first: drop ours, attach theirs.
                    with contextlib.suppress(Exception):
                        await opened.close()
                    continue
                sandbox = opened
            self._live[(agent_id, thread_id)] = _Verified(sandbox, time.monotonic())
            return sandbox

    async def close(self, agent_id: str, thread_id: str, *, forget: bool = False) -> None:
        """Release the live handle; with ``forget`` also clear the persisted id.

        The id is cleared first and the close never raises: a sandbox the
        backend already reclaimed must not keep a cancellation retrying. An
        in-flight resolve is awaited first so the handle it produces is the
        one released, not cached after the close.
        """
        key = (agent_id, thread_id)
        resolving = self._resolving.get(key)
        if resolving is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(resolving)
        if forget:
            thread = await self._threads.get_or_create(agent_id, thread_id)
            if thread.sandbox_id is not None:
                await self._threads.claim_sandbox(
                    agent_id, thread_id, expected=thread.sandbox_id, sandbox_id=None
                )
        verified = self._live.pop(key, None)
        if verified is not None:
            with contextlib.suppress(Exception):
                await verified.sandbox.close()
