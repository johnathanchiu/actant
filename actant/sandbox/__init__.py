"""Per-thread sandboxes for tools that run code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from actant.sandbox.base import (
    ArtifactRef,
    ArtifactSink,
    Entry,
    ExecResult,
    Sandbox,
    SandboxProvider,
    SandboxSpec,
)
from actant.sandbox.local import LocalSandbox, LocalSandboxProvider

if TYPE_CHECKING:
    from actant.sandbox.remote import RemoteHost, RemoteResult, RemoteTool

__all__ = [
    "ArtifactRef",
    "ArtifactSink",
    "Entry",
    "ExecResult",
    "LocalSandbox",
    "LocalSandboxProvider",
    "RemoteHost",
    "RemoteResult",
    "RemoteTool",
    "Sandbox",
    "SandboxProvider",
    "SandboxSpec",
]


def __getattr__(name: str) -> Any:
    # Lazy: remote builds on actant.tools, which imports this package's base types.
    if name in {"RemoteHost", "RemoteResult", "RemoteTool"}:
        from actant.sandbox import remote

        return getattr(remote, name)
    raise AttributeError(f"module 'actant.sandbox' has no attribute {name!r}")
