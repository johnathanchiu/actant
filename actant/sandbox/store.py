"""Where a keyed sandbox is recorded: its key, its live id and the spec it was opened with.

A sandbox is owned by a plain string key the product picks (``"scene:<id>"``, a thread
id). The record outlives any one process, so every worker reaches the same sandbox by its
key, and one that was reclaimed is reopened from the spec recorded with it.

This module is part of :mod:`actant.sandbox`, which the agent runtime depends on and never
the reverse: it imports nothing from actant's runtime.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol

from actant.sandbox.base import SandboxSpec


@dataclass(frozen=True)
class SandboxRecord:
    """A key's sandbox: its backend id, and the spec it is reopened with when it is gone."""

    key: str
    sandbox_id: str
    #: Without its ``image``: not data, and the backend's to supply (see :func:`spec_to_json`).
    spec: SandboxSpec


class SandboxStore(Protocol):
    """The records of keyed sandboxes. Every write is a compare-and-set on the key's id, so
    two workers that each opened a sandbox for one key agree on one."""

    async def get(self, key: str) -> SandboxRecord | None: ...

    async def claim(
        self, key: str, *, expected: str | None, sandbox_id: str, spec: SandboxSpec
    ) -> SandboxRecord | None:
        """Record ``sandbox_id`` and ``spec`` under ``key`` if the key still holds
        ``expected`` (``None``: no record yet); return the record the key holds after,
        ``None`` when it has none."""
        ...

    async def forget(self, key: str) -> None:
        """The key's record removed; a key with none is left as it is."""
        ...


def spec_to_json(spec: SandboxSpec) -> dict[str, Any]:
    """The spec as JSON, less ``image``: a backend object (a ``modal.Image``), not data.

    Everything else is recorded as it is, ``env`` included: a secret belongs in
    ``secrets`` (backend secret names) or the provider's own configuration, not here.
    """

    data = {f.name: getattr(spec, f.name) for f in fields(spec) if f.name != "image"}
    return json.loads(json.dumps(data, default=_plain))


def spec_from_json(data: Mapping[str, Any]) -> SandboxSpec:
    """A spec :func:`spec_to_json` wrote. A field this actant does not know is dropped."""

    known = {f.name for f in fields(SandboxSpec)} - {"image"}
    args: dict[str, Any] = {
        k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in known
    }
    return SandboxSpec(**args)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"a sandbox spec holds JSON values, not {type(value).__name__}")


class InMemorySandboxStore:
    """``SandboxStore`` for tests and local runs. A spec is kept as a database keeps it:
    without its image."""

    def __init__(self) -> None:
        self._records: dict[str, SandboxRecord] = {}

    async def get(self, key: str) -> SandboxRecord | None:
        return self._records.get(key)

    async def claim(
        self, key: str, *, expected: str | None, sandbox_id: str, spec: SandboxSpec
    ) -> SandboxRecord | None:
        current = self._records.get(key)
        if (current.sandbox_id if current is not None else None) == expected:
            self._records[key] = SandboxRecord(key, sandbox_id, replace(spec, image=None))
        return self._records.get(key)

    async def forget(self, key: str) -> None:
        self._records.pop(key, None)
