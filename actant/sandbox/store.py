"""Where a sandbox is recorded: its id, its backend's live id and the spec it was opened with.

A sandbox's id is an opaque string its owner picks (a scene's own id, a thread id). The
record outlives any one process, so every worker reaches the same sandbox by its id, and
one its backend reclaimed is reopened from the spec recorded with it.

This module is part of :mod:`actant.sandbox`, which the agent runtime depends on and never
the reverse: it imports nothing from actant's runtime.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Protocol

from actant.sandbox.base import Location, Mount, Restore, SandboxSpec


@dataclass(frozen=True)
class SandboxRecord:
    """A sandbox: its backend's id for the live one (``provider_id``), and the spec it is
    reopened with when that one is gone."""

    sandbox_id: str
    provider_id: str
    #: Without its ``image``: not data, and the backend's to supply (see :func:`spec_to_json`).
    spec: SandboxSpec


class SandboxStore(Protocol):
    """The records of sandboxes by id. Every write is a compare-and-set on the backend's id,
    so two workers that each opened a sandbox for one id agree on one."""

    async def get(self, sandbox_id: str) -> SandboxRecord | None: ...

    async def claim(
        self, sandbox_id: str, *, expected: str | None, provider_id: str, spec: SandboxSpec
    ) -> SandboxRecord | None:
        """Record ``provider_id`` and ``spec`` for ``sandbox_id`` if it still holds
        ``expected`` (``None``: no record yet); return its record after, ``None`` when it
        has none."""
        ...

    async def forget(self, sandbox_id: str) -> None:
        """The sandbox's record removed; an id with none is left as it is."""
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
    args["restore"] = tuple(
        Restore(Location(**r["source"]), r["path"], r["push"]) for r in args.get("restore", ())
    )
    args["mounts"] = tuple(
        Mount(Location(**m["source"]), m["path"]) for m in args.get("mounts", ())
    )
    return SandboxSpec(**args)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"a sandbox spec holds JSON values, not {type(value).__name__}")


class InMemorySandboxStore:
    """``SandboxStore`` for tests and local runs. A spec is kept as a database keeps it:
    without its image."""

    def __init__(self) -> None:
        self._records: dict[str, SandboxRecord] = {}

    async def get(self, sandbox_id: str) -> SandboxRecord | None:
        return self._records.get(sandbox_id)

    async def claim(
        self, sandbox_id: str, *, expected: str | None, provider_id: str, spec: SandboxSpec
    ) -> SandboxRecord | None:
        current = self._records.get(sandbox_id)
        if (current.provider_id if current is not None else None) == expected:
            record = SandboxRecord(sandbox_id, provider_id, replace(spec, image=None))
            self._records[sandbox_id] = record
        return self._records.get(sandbox_id)

    async def forget(self, sandbox_id: str) -> None:
        self._records.pop(sandbox_id, None)
