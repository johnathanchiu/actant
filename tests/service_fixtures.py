"""A generic service the service tests run in-process and inside a real host process."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel

from actant.sandbox.host import Host, script_env
from actant.sandbox.processes import host_function, on_host
from actant.sandbox.protocol import CallRequest


class Box(BaseModel):
    width: float
    height: float


@dataclass
class Picture:
    text: str
    images: list[str | bytes] = field(default_factory=list)


class Counter:
    #: How many instances this process opened, across keys.
    opened: ClassVar[int] = 0

    def __init__(self, start: int = 0) -> None:
        self.count = start

    @classmethod
    async def open(cls, start: int = 0) -> Counter:
        await asyncio.sleep(0.05)  # widen the window two first calls could race in
        cls.opened += 1
        return cls(start)

    async def close(self) -> None:
        Path(f"closed-at-{self.count}").touch()

    def block(self, seconds: float) -> str:
        """A plain ``def``: the host runs it in a thread."""
        time.sleep(seconds)
        return "blocked"

    async def bump(self, by: int = 1) -> dict[str, int]:
        """Add to the counter."""
        self.count += by
        return {"count": self.count}

    async def opens(self) -> int:
        """How many instances were opened."""
        return Counter.opened

    async def wait(self, seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "waited"

    async def picture(self, size: int, as_file: bool = False) -> Picture:
        data = b"\x89PNG\r\n\x1a\n" + os.urandom(size)
        if not as_file:
            return Picture("bytes", [data])
        Path("picture.png").write_bytes(data)
        return Picture("file", ["picture.png", "secret.txt"])

    async def area(self, box: Box, unit: Literal["m", "cm"] = "m") -> str:
        """Methods receive the annotated types, not their JSON."""
        return f"{type(box).__name__} {box.width * box.height:g} {unit}"

    async def raw(self, data: str) -> Picture:
        return Picture("", [data.encode()])

    async def length(self, text: str) -> int:
        return len(text)

    async def fail(self) -> str:
        raise ValueError("no good")

    async def note(self, path: str, text: str) -> str:
        Path(path).write_text(text)
        return path

    async def env(self, name: str) -> dict[str, str | None]:
        return {"host": os.environ.get(name), "script": script_env().get(name)}


#: The host a test serves ``Placed`` from, for :func:`relay`.
HOST: list[Host] = []

#: ``Placed.linger``'s tasks, held so they are not collected while they wait.
LINGERING: set[asyncio.Future[None]] = set()


async def _linger() -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        raise RuntimeError("client closed") from None


@host_function
def where(tag: str) -> str:
    if tag == "bad":
        raise KeyError(tag)
    return f"{tag} {os.getpid()}"


@host_function
async def relay(to: str) -> str:
    """Awaits a worker's call from the host, while that worker waits on this."""
    _, response = await HOST[0].call(CallRequest(service="p", key=to, method="pid"))
    return response.text


class Placed(Counter):
    """``Counter`` with what a test of worker processes asks of one."""

    async def pid(self) -> int:
        return os.getpid()

    async def crash(self) -> str:
        os._exit(3)

    async def hold(self, marker: str) -> str:
        Path(marker).touch()
        try:
            await asyncio.Event().wait()
        finally:
            Path(f"{marker}.cancelled").touch()
        return "never"

    async def linger(self) -> str:
        """Leave a task behind that fails as it is cancelled, as a client closed under it."""
        LINGERING.add(asyncio.ensure_future(_linger()))
        return "lingering"

    async def where(self, tag: str) -> str:
        return await on_host(where, tag)

    async def relay(self, to: str) -> str:
        return await on_host(relay, to)

    async def unregistered(self) -> str:
        return await on_host(os.getcwd)


class Stages:
    """A second service for the product's own calls, served next to ``Counter``."""

    def __init__(self) -> None:
        self.stage = 0

    async def advance(self) -> str:
        self.stage += 1
        return f"stage {self.stage}"
