"""A cancellation acknowledgement fences every side effect of its service call."""

from __future__ import annotations

import asyncio
import threading
from http import HTTPStatus

from actant.sandbox import Endpoint, host
from actant.sandbox.protocol import CallRequest, CancelRequest
from actant.sandbox.service import call_host


def request(method: str, call_id: str = "old-call") -> CallRequest:
    return CallRequest(service="author", key="shared-folder", method=method, call_id=call_id)


async def test_async_cancel_ack_waits_for_method_cleanup() -> None:
    started, release, cleanup, finish_cleanup = (asyncio.Event() for _ in range(4))
    effects: list[str] = []

    class Author:
        async def write(self) -> str:
            started.set()
            try:
                await release.wait()
                effects.append("old-write")
                return "written"
            finally:
                cleanup.set()
                await finish_cleanup.wait()
                effects.append("old-cleanup")

    serving = host.Host({"author": Author})
    caller = asyncio.create_task(serving.call(request("write")))
    acknowledgement = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        acknowledgement = asyncio.create_task(serving.cancel(CancelRequest(call_id="old-call")))
        release.set()
        await asyncio.wait_for(cleanup.wait(), 2)
        assert not acknowledgement.done(), "acknowledgement preceded awaited cleanup"
        finish_cleanup.set()
        status, reply = await asyncio.wait_for(acknowledgement, 2)
        assert status == HTTPStatus.OK and reply.text == "drained"
        await asyncio.wait_for(caller, 2)
        effects.append("replacement-start")
        assert effects == ["old-write", "old-cleanup", "replacement-start"]
    finally:
        release.set()
        finish_cleanup.set()
        await asyncio.gather(
            caller, *([acknowledgement] if acknowledgement else []), return_exceptions=True
        )
        await serving.close_instances()


async def test_sync_thread_write_finishes_before_cancel_ack() -> None:
    started, release, finished = (threading.Event() for _ in range(3))
    effects: list[str] = []

    class Author:
        def write(self) -> str:
            started.set()
            assert release.wait(5), "test did not release service thread"
            effects.append("old-thread-write")
            finished.set()
            return "written"

    serving = host.Host({"author": Author})
    caller = asyncio.create_task(serving.call(request("write")))
    acknowledgement = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        acknowledgement = asyncio.create_task(serving.cancel(CancelRequest(call_id="old-call")))
        await asyncio.sleep(0)
        assert not acknowledgement.done() and not finished.is_set()
        release.set()
        status, reply = await asyncio.wait_for(acknowledgement, 2)
        assert status == HTTPStatus.OK and reply.text == "drained" and finished.is_set()
        effects.append("replacement-start")
        await asyncio.wait_for(caller, 2)
        assert effects == ["old-thread-write", "replacement-start"]
    finally:
        release.set()
        await asyncio.gather(
            caller, *([acknowledgement] if acknowledgement else []), return_exceptions=True
        )
        await serving.close_instances()


async def test_cancel_before_call_tombstone_prevents_open_and_method_effects() -> None:
    effects: list[str] = []

    class Author:
        @classmethod
        async def open(cls) -> Author:
            effects.append("opened")
            return cls()

        async def write(self) -> str:
            effects.append("written")
            return "written"

    serving = host.Host({"author": Author})
    try:
        for _ in range(2):
            status, reply = await serving.cancel(CancelRequest(call_id="old-call"))
            assert status == HTTPStatus.OK and reply.text == "drained"
        _, cancelled = await serving.call(request("write"))
        assert cancelled.error is not None and effects == []
        # A replacement is a new invocation even when its service and instance key match.
        status, replacement = await serving.call(request("write", "replacement-call"))
        assert status == HTTPStatus.OK and replacement.text == "written"
        assert effects == ["opened", "written"]
    finally:
        await serving.close_instances()


async def test_repeated_cancel_ack_waits_on_same_running_call() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    effects: list[str] = []

    class Author:
        async def write(self) -> str:
            started.set()
            await release.wait()
            effects.append("old-write")
            return "written"

    serving = host.Host({"author": Author})
    caller = asyncio.create_task(serving.call(request("write")))
    acknowledgements = []
    try:
        await asyncio.wait_for(started.wait(), 2)
        acknowledgements = [
            asyncio.create_task(serving.cancel(CancelRequest(call_id="old-call")))
            for _ in range(2)
        ]
        await asyncio.sleep(0)
        assert all(not acknowledgement.done() for acknowledgement in acknowledgements)
        release.set()
        replies = await asyncio.wait_for(asyncio.gather(*acknowledgements), 2)
        assert all(
            status == HTTPStatus.OK and reply.text == "drained" for status, reply in replies
        )
        await asyncio.wait_for(caller, 2)
        status, reply = await serving.cancel(CancelRequest(call_id="old-call"))
        assert status == HTTPStatus.OK and reply.text == "drained"
        assert effects == ["old-write"]
    finally:
        release.set()
        await asyncio.gather(caller, *acknowledgements, return_exceptions=True)
        await serving.close_instances()


async def test_cancel_waits_for_pending_shared_instance_open() -> None:
    opening, release = asyncio.Event(), asyncio.Event()
    effects: list[str] = []

    class Author:
        @classmethod
        async def open(cls) -> Author:
            opening.set()
            await release.wait()
            effects.append("opened")
            return cls()

        async def write(self) -> str:
            effects.append("written")
            return "written"

    serving = host.Host({"author": Author})
    old = asyncio.create_task(serving.call(request("write")))
    replacement = acknowledgement = None
    try:
        await asyncio.wait_for(opening.wait(), 2)
        replacement = asyncio.create_task(serving.call(request("write", "second-call")))
        acknowledgement = asyncio.create_task(serving.cancel(CancelRequest(call_id="old-call")))
        await asyncio.sleep(0)
        assert not acknowledgement.done()
        release.set()
        await asyncio.wait_for(acknowledgement, 2)
        status, reply = await asyncio.wait_for(replacement, 2)
        assert status == HTTPStatus.OK and reply.text == "written"
        await asyncio.wait_for(old, 2)
        assert effects.count("opened") == 1 and effects.count("written") in (1, 2)
    finally:
        release.set()
        await asyncio.gather(
            old,
            *([replacement] if replacement else []),
            *([acknowledgement] if acknowledgement else []),
            return_exceptions=True,
        )
        await serving.close_instances()


async def test_legacy_calls_without_call_id_keep_normal_results() -> None:
    class Author:
        async def write(self) -> str:
            return "legacy"

    serving = host.Host({"author": Author})
    try:
        legacy = CallRequest(service="author", key="shared-folder", method="write")
        assert legacy.call_id == ""
        status, reply = await serving.call(legacy)
        assert status == HTTPStatus.OK and reply.text == "legacy" and reply.error is None
    finally:
        await serving.close_instances()


async def test_repeated_client_cancellation_waits_for_remote_drain() -> None:
    started, release, finished = (asyncio.Event() for _ in range(3))
    effects: list[str] = []

    class Author:
        async def write(self) -> str:
            started.set()
            await release.wait()
            effects.append("old-write")
            finished.set()
            return "written"

    serving = host.Host({"author": Author})
    connections: set[asyncio.Task[None]] = set()

    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        try:
            await serving.connection(reader, writer)
        finally:
            connections.discard(task)

    server = await asyncio.start_server(connection, "127.0.0.1", 0)
    assert server.sockets
    endpoint = Endpoint(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    caller = asyncio.create_task(
        call_host(endpoint, "author", "write", {}, key="shared", timeout=5)
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        caller.cancel()
        await asyncio.sleep(0)
        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done() and not finished.is_set()
        release.set()
        result = await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 2)
        assert isinstance(result[0], asyncio.CancelledError)
        assert finished.is_set(), "cancelled client returned before remote side effects drained"
        effects.append("replacement-start")
        assert effects == ["old-write", "replacement-start"]
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        server.close()
        await server.wait_closed()
        for writer in list(serving.writers):
            writer.close()
        if connections:
            await asyncio.wait_for(asyncio.gather(*connections, return_exceptions=True), 2)
        await serving.close_instances()


async def test_same_call_id_deduplicates_and_rejects_different_arguments() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    effects: list[str] = []

    class Author:
        async def write(self, text: str = "original") -> str:
            started.set()
            await release.wait()
            effects.append(text)
            return text

    serving = host.Host({"author": Author})
    call = request("write")
    first = asyncio.create_task(serving.call(call))
    duplicate = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        duplicate = asyncio.create_task(serving.call(call))
        status, reply = await serving.call(call.model_copy(update={"args": {"text": "changed"}}))
        assert status == HTTPStatus.CONFLICT and reply.error is not None
        release.set()
        replies = await asyncio.wait_for(asyncio.gather(first, duplicate), 2)
        assert all(reply.text == "original" for _, reply in replies)
        assert effects == ["original"]
    finally:
        release.set()
        await asyncio.gather(first, *([duplicate] if duplicate else []), return_exceptions=True)
        await serving.close_instances()


async def test_shutdown_drains_running_call_before_closing_instance() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    effects: list[str] = []

    class Author:
        async def write(self) -> str:
            started.set()
            await release.wait()
            effects.append("write")
            return "written"

        async def close(self) -> None:
            effects.append("close")

    serving = host.Host({"author": Author})
    caller = asyncio.create_task(serving.call(request("write")))
    try:
        await asyncio.wait_for(started.wait(), 2)
        shutdown = serving.shutdown()
        status, reply = await serving.call(request("write", "late-call"))
        assert status == HTTPStatus.SERVICE_UNAVAILABLE and reply.error is not None
        await asyncio.sleep(0)
        assert not shutdown.done() and effects == []
        release.set()
        await asyncio.wait_for(shutdown, 2)
        await asyncio.wait_for(caller, 2)
        assert effects == ["write", "close"]
    finally:
        release.set()
        await asyncio.gather(caller, serving.shutdown(), return_exceptions=True)
