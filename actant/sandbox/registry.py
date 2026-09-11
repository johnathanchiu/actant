"""One sandbox per thread, for the life of this worker process.

Process-scoped like :class:`~actant.runtime.coordinator.SubThreadRegistry`:
a live handle is cached here, and the sandbox id is persisted on the thread
so another worker (or this one after a restart) reattaches instead of
opening a second sandbox over the same files.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from actant.runtime.interfaces.stores import ThreadStore
from actant.runtime.types.threads import AgentThread
from actant.sandbox.base import Sandbox, SandboxProvider, SandboxSpec


class SandboxRegistry:
    def __init__(self, providers: Mapping[str, SandboxProvider], threads: ThreadStore) -> None:
        self._providers = dict(providers)
        self._threads = threads
        self._live: dict[tuple[str, str], Sandbox] = {}
        # ponytail: one lock; per-key locks if opens ever contend across threads
        self._lock = asyncio.Lock()

    def provider(self, backend: str) -> SandboxProvider:
        try:
            return self._providers[backend]
        except KeyError:
            raise KeyError(
                f"no sandbox provider registered for backend {backend!r}; "
                f"known: {sorted(self._providers)}"
            ) from None

    async def for_thread(self, spec: SandboxSpec, thread: AgentThread) -> Sandbox:
        """The thread's sandbox: live, reattached by its persisted id, or newly opened."""
        key = (thread.agent_id, thread.id)
        async with self._lock:
            live = self._live.get(key)
            if live is not None:
                return live
            provider = self.provider(spec.backend)
            sandbox: Sandbox | None = None
            if thread.sandbox_id:
                try:
                    sandbox = await provider.attach(spec, thread.sandbox_id)
                except KeyError:
                    sandbox = None
            if sandbox is None:
                sandbox = await provider.open(spec, agent_id=thread.agent_id, thread_id=thread.id)
                thread.sandbox_id = sandbox.id
                await self._threads.update(thread)
            self._live[key] = sandbox
            return sandbox

    async def close(self, agent_id: str, thread_id: str, *, forget: bool = False) -> None:
        """Release the live handle; with ``forget`` also clear the persisted id."""
        async with self._lock:
            sandbox = self._live.pop((agent_id, thread_id), None)
        if sandbox is not None:
            await sandbox.close()
        if forget:
            thread = await self._threads.get_or_create(agent_id, thread_id)
            if thread.sandbox_id is not None:
                thread.sandbox_id = None
                await self._threads.update(thread)
