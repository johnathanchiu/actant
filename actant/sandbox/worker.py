"""One worker process of a placed service (:mod:`actant.sandbox.processes`).

::

    python -m actant.sandbox.worker '<WorkerConfig JSON>' <fd>

Serves its service on its end of the host's socketpair with a :class:`~actant.sandbox.host.Host`
that has no HTTP server: the same instances per key, argument checks, encoding and cancels.
Exits once the socket closes, after closing its instances.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import sys
from typing import cast

from actant.sandbox import host
from actant.sandbox import processes
from actant.sandbox.processes import Kind, frame, receive, send
from actant.sandbox.protocol import CallRequest, WorkerConfig


async def serve(config: WorkerConfig, fd: int) -> None:
    host.scrub(config.scrub)
    serving = host.Host({config.service: host.load(config.path)})
    reader, writer = await asyncio.open_connection(sock=socket.socket(fileno=fd))
    to_host = processes.link(writer)
    running: set[asyncio.Task[object]] = set()

    async def one(number: int, request: CallRequest) -> None:
        _, response = await serving.call(request)
        with contextlib.suppress(ConnectionError):  # the host is gone
            await send(writer, frame(Kind.DONE, number, response))

    while True:
        try:
            kind, number, payload = await receive(reader)
        except (asyncio.IncompleteReadError, ConnectionError):
            break
        if kind == Kind.ANSWER:
            to_host.answered(number, cast(tuple[bool, object], payload))
            continue
        if kind == Kind.CALL:
            assert isinstance(payload, CallRequest)
            task = asyncio.create_task(one(number, payload))
        else:
            assert kind == Kind.CANCEL and isinstance(payload, str)
            task = asyncio.create_task(serving.cancel(payload))
        running.add(task)
        task.add_done_callback(running.discard)
    await serving.close_instances()
    writer.close()


def main(argv: list[str]) -> int:
    processes.log_to_stderr()
    config = WorkerConfig.model_validate_json(argv[0])
    asyncio.run(serve(config, int(argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
