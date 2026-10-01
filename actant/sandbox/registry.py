"""Sandboxes by id: an opaque string their owner picks (a scan's id, a thread's id).

The owner opens the sandbox (:meth:`SandboxRegistry.open`) and closes it
(:meth:`~SandboxRegistry.close`); anything else that works in it attaches by its id
(:meth:`~SandboxRegistry.attach`) and never closes it. The backend's live id
(``provider_id``) and the spec it was opened with are recorded in a
:class:`~actant.sandbox.store.SandboxStore`, so every worker reaches the same sandbox, and
one its backend reclaimed is reopened from that spec.

Process-scoped: a live handle is cached here per sandbox id. This module is part of the
sandbox layer, which the agent runtime depends on and never the reverse: a thread names the
id of the sandbox it runs in, and this module knows nothing of threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass

from actant.sandbox.base import Sandbox, SandboxProvider, SandboxSpec
from actant.sandbox.store import SandboxStore, spec_to_json

# How long a verified handle is served without asking the backend again. A
# backend reclaims a sandbox after minutes idle, so one verified seconds ago
# was not reclaimed for idleness; attaching on every call instead put a Modal
# poll in front of every tool call.
VERIFIED_FOR_S = 5.0
#: How long ``close`` waits for a cancelled in-flight resolve to clean up.
CANCEL_WAIT_S = 60.0


@dataclass
class _Verified:
    sandbox: Sandbox
    at: float


class SandboxRegistry:
    def __init__(self, providers: Mapping[str, SandboxProvider], store: SandboxStore) -> None:
        self._providers = dict(providers)
        self._store = store
        self._live: dict[str, _Verified] = {}
        # The spec this process last saw recorded per sandbox, as JSON: an owner's open with
        # the same spec is served from the cache, one with another spec records it.
        self._recorded: dict[str, str] = {}
        # One resolve per sandbox and spec at a time, shared by every caller waiting on it.
        self._resolving: dict[tuple[str, str | None], asyncio.Task[Sandbox]] = {}

    def provider(self, backend: str) -> SandboxProvider:
        try:
            return self._providers[backend]
        except KeyError:
            raise KeyError(
                f"no sandbox provider registered for backend {backend!r}; "
                f"known: {sorted(self._providers)}"
            ) from None

    async def open(self, sandbox_id: str, spec: SandboxSpec) -> Sandbox:
        """The sandbox, for its owner: the live one kept, else one opened with ``spec``.

        Idempotent, so a retried open reaches the same sandbox. ``spec`` is recorded with
        it: a live sandbox keeps running as it was opened, and ``spec`` is what it reopens
        with if its backend reclaims it.
        """
        wanted = _json(spec)
        if (
            self._recorded.get(sandbox_id) == wanted
            and (live := self._verified(sandbox_id)) is not None
        ):
            return live
        return await self._shared(sandbox_id, wanted, spec)

    async def attach(self, sandbox_id: str) -> Sandbox:
        """The sandbox, for a caller that does not own it and never closes it.

        One its backend reclaimed is reopened from the recorded spec. ``KeyError`` for a
        id nobody opened, or one its owner closed.
        """
        if (live := self._verified(sandbox_id)) is not None:
            return live
        return await self._shared(sandbox_id, None, None)

    async def spec(self, sandbox_id: str) -> SandboxSpec:
        """The spec recorded with the sandbox (without its ``image``)."""
        record = await self._store.get(sandbox_id)
        if record is None:
            raise KeyError(f"no sandbox is open under {sandbox_id!r}")
        return record.spec

    def _verified(self, sandbox_id: str) -> Sandbox | None:
        """A handle verified within :data:`VERIFIED_FOR_S`, served without asking the
        backend again; past that it is asked, so a reclaimed sandbox is noticed."""
        verified = self._live.get(sandbox_id)
        if verified is not None and time.monotonic() - verified.at < VERIFIED_FOR_S:
            return verified.sandbox
        return None

    async def _shared(
        self, sandbox_id: str, wanted: str | None, spec: SandboxSpec | None
    ) -> Sandbox:
        """Concurrent callers for one sandbox share a single resolve; callers for different
        keys never wait on each other."""
        task_key = (sandbox_id, wanted)
        task = self._resolving.get(task_key)
        if task is None:
            task = asyncio.ensure_future(self._resolve(sandbox_id, spec))
            self._resolving[task_key] = task
            task.add_done_callback(lambda done: self._forget_resolve(task_key, done))
        # Shielded: one caller's cancellation must not fail the others sharing it.
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and current is not None and not current.cancelling():
                # ``close`` cancelled the shared resolve, not this caller.
                raise RuntimeError(f"sandbox {sandbox_id} was closed while opening") from None
            raise

    def _forget_resolve(
        self, task_key: tuple[str, str | None], done: asyncio.Task[Sandbox]
    ) -> None:
        if self._resolving.get(task_key) is done:
            del self._resolving[task_key]

    async def _resolve(self, sandbox_id: str, spec: SandboxSpec | None) -> Sandbox:
        while True:
            # The recorded id, not a cached one: another worker may have replaced it.
            record = await self._store.get(sandbox_id)
            if record is not None:
                use = spec if spec is not None else record.spec
            elif spec is not None:
                use = spec
            else:
                raise KeyError(f"no sandbox is open under {sandbox_id!r}")
            provider = self.provider(use.backend)
            expected = record.provider_id if record is not None else None
            sandbox = None
            if record is not None:
                with contextlib.suppress(KeyError):
                    sandbox = await provider.attach(use, record.provider_id)
            if sandbox is not None and record is not None:
                if spec is not None and _json(spec) != _json(record.spec):
                    held = await self._store.claim(
                        sandbox_id, expected=expected, provider_id=record.provider_id, spec=spec
                    )
                    if held is None or held.provider_id != record.provider_id:
                        continue  # replaced or closed meanwhile: look again
                return self._keep(sandbox_id, sandbox, use)
            opened = await provider.open(use, sandbox_id=sandbox_id)
            try:
                held = await self._store.claim(
                    sandbox_id, expected=expected, provider_id=opened.id, spec=use
                )
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await asyncio.shield(opened.close())
                raise
            if held is None or held.provider_id != opened.id:
                # Another worker recorded one first, or the owner closed it: drop ours.
                with contextlib.suppress(Exception):
                    await opened.close()
                if held is None and spec is None:
                    raise KeyError(f"sandbox {sandbox_id} was closed while it was reopened")
                continue
            return self._keep(sandbox_id, opened, use)

    def _keep(self, sandbox_id: str, sandbox: Sandbox, spec: SandboxSpec) -> Sandbox:
        self._live[sandbox_id] = _Verified(sandbox, time.monotonic())
        self._recorded[sandbox_id] = _json(spec)
        return sandbox

    async def close(self, sandbox_id: str) -> None:
        """For the sandbox's owner: its record forgotten, and the sandbox stopped wherever it was
        opened (attached by its recorded id when this process holds no handle).

        The record is forgotten first and the close never raises: a sandbox the backend
        already reclaimed must not keep a cancellation retrying. An in-flight resolve is
        cancelled and forgotten, and waited for at most :data:`CANCEL_WAIT_S` while it
        terminates what it was opening: an open that never finishes must not hold the close.
        """
        resolving = [
            self._resolving.pop(task_key)
            for task_key in [k for k in self._resolving if k[0] == sandbox_id]
        ]
        for task in resolving:
            task.cancel()
        if resolving:
            await asyncio.wait(resolving, timeout=CANCEL_WAIT_S)
        for task in resolving:
            if task.done() and not task.cancelled():
                task.exception()  # retrieved: its callers already saw it
        record = await self._store.get(sandbox_id)
        await self._store.forget(sandbox_id)
        self._recorded.pop(sandbox_id, None)
        verified = self._live.pop(sandbox_id, None)
        sandboxes = [verified.sandbox] if verified is not None else []
        if record is not None and record.provider_id not in {s.id for s in sandboxes}:
            with contextlib.suppress(Exception):
                provider = self.provider(record.spec.backend)
                sandboxes.append(await provider.attach(record.spec, record.provider_id))
        for sandbox in sandboxes:
            with contextlib.suppress(Exception):
                await sandbox.close()


def _json(spec: SandboxSpec) -> str:
    return json.dumps(spec_to_json(spec), sort_keys=True)
