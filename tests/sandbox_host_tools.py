"""Factory and tools the sandbox host tests run inside a real host process."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Ctx:
    key: str
    start: int = 0
    count: int = 0
    seen: list[str] = field(default_factory=list)


async def make_ctx(key: str, init: dict[str, object] | None) -> Ctx:
    return Ctx(key, start=int((init or {}).get("start", 0)))  # pyright: ignore[reportArgumentType]


async def bump(ctx: Ctx, by: int = 1) -> dict[str, int]:
    """Add to the context's counter."""
    ctx.count += by
    return {"count": ctx.start + ctx.count}


async def slow(ctx: Ctx, seconds: float) -> str:
    await asyncio.sleep(seconds)
    return ctx.key


@dataclass
class Rendered:
    text: str
    images: list[str]


def render(ctx: Ctx, name: str) -> Rendered:
    Path(name).write_bytes(b"\x89PNG fake " + ctx.key.encode())
    return Rendered("rendered", [name])


async def boom(ctx: Ctx) -> str:
    raise ValueError("no good")


async def size(ctx: Ctx, text: str) -> int:
    return len(text)


async def env(ctx: Ctx, name: str) -> dict[str, str | None]:
    import os

    from actant.sandbox.host import scrubbed_env

    return {"host": os.environ.get(name), "scrubbed": scrubbed_env().get(name)}
