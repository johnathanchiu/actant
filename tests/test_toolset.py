"""Toolsets: schema from a plain class, and identical results in-process and over a host."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import pytest

from actant.sandbox import Endpoint, LocalSandbox, LocalSandboxProvider, SandboxSpec
from actant.sandbox import StorageStatus, host
from actant.sandbox.protocol import (
    EntryConfig,
    Header,
    HostConfig,
    PushConfig,
    RestoreConfig,
    Route,
)
from actant.tools import LocalRunner, RemoteRunner, SandboxRunner, tools, toolset_schema
from actant.core import JSONObject
from actant.tools.base import CallContext, MetadataKey, ToolResult
from actant.tools.toolset import call_host
from toolset_fixtures import Counter, Stages

if TYPE_CHECKING:
    from some_module_that_only_type_checkers_see import Thing  # noqa: F401

TESTS = str(Path(__file__).parent)
TOOLSETS = {"counter": "toolset_fixtures:Counter", "stages": "toolset_fixtures:Stages"}


def _ctx(sandbox: LocalSandbox | None = None, thread: str = "t") -> CallContext:
    return CallContext("a", thread, "r", "tc", "turn", sandbox=sandbox)


class Base:
    async def inherited(self, flag: bool = False) -> str:
        """From the base."""
        return str(flag)

    async def close(self) -> None: ...


class Sample(Base):
    @classmethod
    async def open(cls) -> Sample:
        return cls()

    async def search(self, query: str, limit: int = 5, tag: str | None = None) -> Thing:
        """Find things.

        More detail."""
        raise NotImplementedError

    async def _private(self) -> str:
        return ""

    def sync_helper(self, x: int) -> int:
        return x

    @staticmethod
    async def static(x: int) -> int:
        return x


def test_schema_comes_from_the_signature_without_self() -> None:
    schemas: dict[str, Any] = {
        s["function"]["name"]: s["function"]  # pyright: ignore[reportIndexIssue]
        for s in toolset_schema(Sample)
    }
    assert list(schemas) == ["inherited", "search", "sync_helper"]
    search = schemas["search"]
    assert search["description"] == "Find things.\n\nMore detail."
    params = search["parameters"]
    assert params["required"] == ["query"]
    assert set(params["properties"]) == {"query", "limit", "tag"}
    assert params["properties"]["limit"]["default"] == 5
    assert {"type": "null"} in params["properties"]["tag"]["anyOf"]
    assert params["additionalProperties"] is False
    assert schemas["inherited"]["description"] == "From the base."


def test_unannotated_parameters_are_rejected() -> None:
    class Bad:
        async def go(self, x) -> str:  # pyright: ignore[reportMissingParameterType]
            return x

    with pytest.raises(TypeError, match="needs a type"):
        toolset_schema(Bad)


async def test_invalid_arguments_fail_at_build() -> None:
    (bump,) = [t for t in tools(Counter, LocalRunner(Counter())) if t.name == "bump"]
    with pytest.raises(ValueError, match="Invalid arguments"):
        await bump.build({"by": "many"}, _ctx())


@pytest.fixture
async def sandbox(tmp_path: Path) -> AsyncIterator[LocalSandbox]:
    provider = LocalSandboxProvider(tmp_path)
    spec = SandboxSpec(
        toolsets=TOOLSETS,
        env={"PYTHONPATH": TESTS, "SERVICE_KEY": "k"},
        scrub_env=("SERVICE_KEY",),
    )
    opened = await provider.open(spec, agent_id="a", thread_id="t")
    assert isinstance(opened, LocalSandbox)
    yield opened
    await opened.close()


async def _call(
    runner: object, method: str, ctx: CallContext | None = None, **args: object
) -> ToolResult:
    (tool,) = [t for t in tools(Counter, runner) if t.name == method]  # pyright: ignore[reportArgumentType]
    ctx = ctx or _ctx()
    return await (await tool.build(cast(JSONObject, args), ctx)).execute()


async def test_local_and_remote_runners_give_identical_results(
    sandbox: LocalSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    monkeypatch.chdir(sandbox.root)
    local = LocalRunner(Counter())
    remote = RemoteRunner(endpoint, "counter", key="k1")
    for method, args in [
        ("bump", {"by": 2}),
        ("length", {"text": "hello"}),
        ("fail", {}),
    ]:
        mine, theirs = await _call(local, method, **args), await _call(remote, method, **args)
        assert mine.output == theirs.output and mine.content_blocks == theirs.content_blocks
        assert (mine.error is None) == (theirs.error is None)
        if mine.error:
            assert theirs.error and theirs.error.startswith("ValueError: no good")

    # Paths name files in the host's working directory (the sandbox root); a non-image is dropped.
    (sandbox.root / "secret.txt").write_text("key")
    mine = await _call(local, "picture", size=10, as_file=True)
    theirs = await _call(remote, "picture", size=10, as_file=True)
    assert mine.output == theirs.output and "dropped secret.txt" in str(theirs.output)
    assert theirs.content_blocks and len(theirs.content_blocks) == 3
    assert theirs.content_blocks[2]["source"]["media_type"] == "image/png"  # pyright: ignore[reportIndexIssue]


async def test_instances_are_per_key_opened_once_and_calls_run_in_parallel(
    sandbox: LocalSandbox,
) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    a = RemoteRunner(endpoint, "counter", key="a", init={"start": 10})
    b = RemoteRunner(endpoint, "counter", key="b")
    # Two first calls on one key race; both must land on one instance.
    first = await asyncio.gather(_call(a, "bump"), _call(a, "bump"))
    assert sorted(json.loads(str(r.output))["count"] for r in first) == [11, 12]
    assert json.loads(str((await _call(b, "bump")).output)) == {"count": 1}
    assert (await _call(a, "opens")).output == "2"

    started = time.monotonic()
    await asyncio.gather(*[_call(a, "wait", seconds=0.5) for _ in range(5)])
    assert time.monotonic() - started < 2


async def test_sync_methods_run_in_threads_over_kept_alive_connections(
    sandbox: LocalSandbox,
) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    runner = RemoteRunner(endpoint, "counter", key="threads")
    started = time.monotonic()
    blocked = asyncio.gather(*[_call(runner, "block", seconds=0.5) for _ in range(5)])
    await asyncio.sleep(0.1)
    assert (await _call(runner, "length", text="abc")).output == "3"
    assert time.monotonic() - started < 0.4  # the loop was free while threads blocked
    assert [r.output for r in await blocked] == ["blocked"] * 5
    assert time.monotonic() - started < 2

    # Sequential calls reuse one pooled connection.
    slot = ("http", endpoint.url.removeprefix("http://"))
    host._idle.pop(slot, None)  # pyright: ignore[reportPrivateUsage]
    await _call(runner, "length", text="a")
    ((first, _),) = host._idle[slot]  # pyright: ignore[reportPrivateUsage]
    await _call(runner, "length", text="b")
    assert [c for c, _ in host._idle[slot]] == [first]  # pyright: ignore[reportPrivateUsage]


async def test_a_connection_lost_after_the_request_is_sent_is_an_error_not_a_retry(
    sandbox: LocalSandbox,
) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    target = urlsplit(endpoint.url)
    drop = asyncio.Event()

    async def relay(client_reader: asyncio.StreamReader, client: asyncio.StreamWriter) -> None:
        """Forward to the host; while ``drop`` is set, swallow the reply and hang up."""
        host_reader, upstream = await asyncio.open_connection(target.hostname, target.port)

        async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while data := await reader.read(65536):
                if writer is client and drop.is_set():
                    drop.clear()
                    break
                writer.write(data)
                await writer.drain()
            writer.close()
            (upstream if writer is client else client).close()

        await asyncio.gather(
            pump(client_reader, upstream), pump(host_reader, client), return_exceptions=True
        )

    server = await asyncio.start_server(relay, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        relayed = RemoteRunner(
            Endpoint(f"http://127.0.0.1:{port}", endpoint.headers), "counter", key="once"
        )
        assert (await _call(relayed, "length", text="warm")).output == "4"  # pools a connection
        drop.set()
        lost = await _call(relayed, "bump")
        assert lost.error and "may have run" in lost.error
        direct = await call_host(endpoint, "counter", "bump", {}, key="once")
        assert json.loads(str(direct.output)) == {"count": 2}  # the lost call ran exactly once
    finally:
        server.close()


async def test_close_stops_the_host_gracefully(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(tmp_path)
    opened = await provider.open(
        SandboxSpec(toolsets=TOOLSETS, env={"PYTHONPATH": TESTS}), agent_id="a", thread_id="t"
    )
    assert isinstance(opened, LocalSandbox)
    endpoint = await opened.endpoint()
    assert endpoint is not None
    await call_host(endpoint, "counter", "bump", {"by": 3}, key="k")
    await opened.close()  # SIGTERM
    assert (opened.root / "closed-at-3").exists()


async def test_sandbox_runner_uses_the_thread_sandbox(sandbox: LocalSandbox) -> None:
    runner = SandboxRunner("counter", init={"start": 5})
    (bump,) = [t for t in tools(Counter, runner) if t.name == "bump"]
    assert bump.needs_sandbox
    result = await _call(runner, "bump", _ctx(sandbox, thread="x"))
    assert json.loads(str(result.output)) == {"count": 6}
    failed = await _call(runner, "bump", _ctx(None))
    assert failed.error and "serves no toolset" in failed.error


async def test_arguments_arrive_as_the_annotated_types_and_bytes_must_be_images(
    sandbox: LocalSandbox,
) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    for runner in (LocalRunner(Counter()), RemoteRunner(endpoint, "counter", key="types")):
        box = {"width": 2, "height": 3}
        assert (await _call(runner, "area", box=box, unit="cm")).output == "Box 6 cm"
        dropped = await _call(runner, "raw", data="not an image")
        assert "dropped image-0" in str(dropped.output) and not dropped.content_blocks
    # The host validates too, for callers that skip ``build``.
    bad = await call_host(endpoint, "counter", "area", {"box": {"width": 1}}, key="types")
    assert bad.error and "ValidationError" in bad.error


async def test_sandbox_runner_refreshes_a_rejected_credential_once(sandbox: LocalSandbox) -> None:
    good = await sandbox.endpoint()
    assert good is not None

    class Rotating:
        refreshes = 0

        async def endpoint(self, *, refresh: bool = False) -> Endpoint:
            assert good is not None
            if refresh:
                Rotating.refreshes += 1
            return good if Rotating.refreshes else Endpoint(good.url, {Header.AUTHORIZATION: "x"})

    stale = cast(LocalSandbox, Rotating())
    result = await _call(SandboxRunner("counter"), "bump", _ctx(stale))
    assert json.loads(str(result.output)) == {"count": 1} and Rotating.refreshes == 1
    await _call(SandboxRunner("counter"), "bump", _ctx(stale))
    assert Rotating.refreshes == 1


async def test_bad_token_unknown_method_and_host_survives_errors(sandbox: LocalSandbox) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    wrong = Endpoint(endpoint.url, {Header.AUTHORIZATION: "Bearer nope"})
    denied = await call_host(wrong, "counter", "bump", {}, key="k")
    assert denied.error and "bearer" in denied.error
    unknown = await call_host(endpoint, "counter", "__init__", {}, key="k")
    assert unknown.error and "unknown tool method" in unknown.error
    assert (await call_host(endpoint, "counter", "fail", {}, key="k")).error
    body = json.dumps({"toolset": "counter", "key": "k", "method": ["bump"]}).encode()
    status, _ = await asyncio.to_thread(host.post, endpoint, Route.CALL, body, 30)
    assert status == 400
    ok = await call_host(endpoint, "counter", "length", {"text": "ok"}, key="k")
    assert ok.output == "2" and MetadataKey.STORAGE not in ok.metadata  # no push, no status
    assert "Bearer" not in repr(endpoint)


async def test_toolsets_on_one_host_are_separate_by_name(sandbox: LocalSandbox) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    bumped = await call_host(endpoint, "counter", "bump", {}, key="k")
    assert json.loads(str(bumped.output)) == {"count": 1}
    staged = await call_host(endpoint, "stages", "advance", {}, key="k")
    assert staged.output == "stage 1"
    assert [t.name for t in tools(Stages, RemoteRunner(endpoint, "stages", key="k"))] == [
        "advance"
    ]
    # A toolset serves only its own methods; an unknown name is refused.
    crossed = await call_host(endpoint, "stages", "bump", {}, key="k")
    assert crossed.error and "unknown tool method" in crossed.error
    missing = await call_host(endpoint, "nope", "bump", {}, key="k")
    assert missing.error and "unknown toolset" in missing.error


async def test_large_arguments_and_images_round_trip(sandbox: LocalSandbox) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    runner = RemoteRunner(endpoint, "counter", key="big")
    text = "x" * (2 * 1024 * 1024)
    assert (await _call(runner, "length", text=text)).output == str(len(text))
    result = await _call(runner, "picture", size=5 * 1024 * 1024)
    assert result.content_blocks
    data = base64.b64decode(result.content_blocks[2]["source"]["data"])  # pyright: ignore[reportIndexIssue]
    assert len(data) == 5 * 1024 * 1024 + 8 and data.startswith(b"\x89PNG")


async def test_script_env_scrubs_what_the_host_keeps(sandbox: LocalSandbox) -> None:
    endpoint = await sandbox.endpoint()
    assert endpoint is not None
    runner = RemoteRunner(endpoint, "counter", key="env")
    result = json.loads(str((await _call(runner, "env", name="SERVICE_KEY")).output))
    assert result == {"host": "k", "script": None}
    token = json.loads(str((await _call(runner, "env", name=host.TOKEN_ENV)).output))
    assert token == {"host": None, "script": None}
    ran = await sandbox.exec(["sh", "-c", "echo ${SERVICE_KEY:-gone}"], timeout=10)
    assert ran.stdout.strip() == "gone"


async def test_attach_reuses_the_running_host_and_close_stops_it(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(tmp_path)
    spec = SandboxSpec(toolsets=TOOLSETS, env={"PYTHONPATH": TESTS})
    opened = await provider.open(spec, agent_id="a", thread_id="t")
    try:
        attached = await provider.attach(spec, opened.id)
        endpoint = await opened.endpoint()
        assert attached is opened and endpoint is not None
        await _call(RemoteRunner(endpoint, "counter", key="k"), "bump")
    finally:
        await opened.close()
    restarted = await provider.attach(spec, opened.id)
    try:
        assert restarted is not opened and await restarted.endpoint() != await opened.endpoint()
    finally:
        await restarted.close()


@pytest.fixture
async def pushing_host(
    tmp_path: Path,
) -> AsyncIterator[tuple[Endpoint, Path, asyncio.subprocess.Process]]:
    """A host started the way a container backend starts one: entry restores, then serves."""
    log = tmp_path / "pushes"
    restore = [sys.executable, "-c", "open('restored.txt', 'w').write('from storage')"]
    push = [sys.executable, "-c", f"open({str(log)!r}, 'a').write('push\\n'); raise SystemExit(3)"]
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "actant.sandbox.entry",
        EntryConfig(
            restore=RestoreConfig(argv=restore),
            host=HostConfig(
                toolsets={"counter": "toolset_fixtures:Counter"},
                port=0,
                bind="127.0.0.1",
                push=PushConfig(argv=push),
            ),
        ).model_dump_json(),
        cwd=tmp_path,
        env={"PYTHONPATH": TESTS, "PATH": "/usr/bin:/bin"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    line = (await asyncio.wait_for(process.stdout.readline(), 30)).decode()
    prefix, _, port = line.strip().partition(" ")
    assert prefix == host.READY_PREFIX, line
    yield Endpoint(f"http://127.0.0.1:{port}"), log, process
    if process.returncode is None:
        process.terminate()
        await process.wait()


async def test_entry_restores_before_serving_and_pushes_after_calls(
    pushing_host: tuple[Endpoint, Path, asyncio.subprocess.Process], tmp_path: Path
) -> None:
    endpoint, log, process = pushing_host
    assert (tmp_path / "restored.txt").read_text() == "from storage"
    await call_host(endpoint, "counter", "note", {"path": "a.txt", "text": "1"}, key="k")
    for _ in range(50):
        if log.exists():
            break
        await asyncio.sleep(0.1)
    assert log.read_text().startswith("push")
    # A burst coalesces into far fewer pushes than calls: one running, one pending.
    await asyncio.gather(*[call_host(endpoint, "counter", "bump", {}, key="k") for _ in range(20)])
    await asyncio.sleep(1.5)
    assert 2 <= len(log.read_text().splitlines()) < 10
    process.terminate()
    await process.wait()
    assert process.stderr is not None
    assert "storage push failed: exited 3" in (await process.stderr.read()).decode()


async def test_shutdown_closes_instances_pushes_and_exits(
    pushing_host: tuple[Endpoint, Path, asyncio.subprocess.Process], tmp_path: Path
) -> None:
    endpoint, log, process = pushing_host
    await call_host(endpoint, "counter", "bump", {"by": 2}, key="k")
    for _ in range(50):  # the push that follows the call
        if log.exists():
            break
        await asyncio.sleep(0.1)
    status, _ = await asyncio.to_thread(host.post, endpoint, Route.SHUTDOWN, b"{}", 30)
    assert status == 200 and (tmp_path / "closed-at-2").exists()
    assert await asyncio.wait_for(process.wait(), 10) == 0
    assert log.read_text().count("push") == 2  # after the call, and once more at shutdown


def test_entry_exits_when_restore_fails(tmp_path: Path) -> None:
    restore = [sys.executable, "-c", "import sys; sys.stderr.write('access denied'); sys.exit(1)"]
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "actant.sandbox.entry",
            EntryConfig(restore=RestoreConfig(argv=restore)).model_dump_json(),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 1 and "access denied" in done.stderr


#: A push that logs, then hangs while ``hang`` exists and exits with the code in ``code``.
PUSH_SCRIPT = """
import os, sys, time
open('pushes', 'a').write('push\\n')
while os.path.exists('hang'):
    time.sleep(0.05)
sys.exit(int(open('code').read()) if os.path.exists('code') else 0)
"""


async def _start_host(tmp_path: Path) -> tuple[Endpoint, asyncio.subprocess.Process]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "actant.sandbox.entry",
        EntryConfig(
            host=HostConfig(
                toolsets={"counter": "toolset_fixtures:Counter"},
                port=0,
                bind="127.0.0.1",
                push=PushConfig(
                    argv=[sys.executable, "-c", PUSH_SCRIPT], interval_s=0.3, timeout_s=1
                ),
            )
        ).model_dump_json(),
        cwd=tmp_path,
        env={"PYTHONPATH": TESTS, "PATH": "/usr/bin:/bin"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert process.stdout is not None
    line = (await asyncio.wait_for(process.stdout.readline(), 30)).decode()
    prefix, _, port = line.strip().partition(" ")
    assert prefix == host.READY_PREFIX, line
    return Endpoint(f"http://127.0.0.1:{port}"), process


async def _until(check: Any, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.05)


async def test_push_failures_and_hangs_never_touch_calls_and_show_in_status(
    tmp_path: Path,
) -> None:
    endpoint, process = await _start_host(tmp_path)
    pushes = tmp_path / "pushes"

    def count() -> int:
        return len(pushes.read_text().splitlines()) if pushes.exists() else 0

    async def storage() -> StorageStatus:
        result = await call_host(endpoint, "counter", "length", {"text": "abc"}, key="k")
        assert result.output == "3" and result.error is None  # the call never suffers
        return StorageStatus.model_validate(result.metadata[MetadataKey.STORAGE])

    try:
        # A hanging push is killed at the timeout; the pusher carries on and retries.
        (tmp_path / "hang").touch()
        first = await storage()
        assert first.pending and first.consecutive_failures == 0
        await _until(lambda: count() >= 2, 10)  # the first timed out, the next started
        status = await storage()
        assert status.last_error == "timed out after 1s" and status.consecutive_failures >= 1

        # A failing push: calls still succeed and the status names the exit.
        (tmp_path / "code").write_text("3")
        (tmp_path / "hang").unlink()
        await asyncio.sleep(1)
        status = await storage()
        assert (status.last_error or "").startswith("exited 3") and status.last_success_at is None

        # Periodic: with no calls at all, the unpushed work is retried until it lands.
        before = count()
        await _until(lambda: count() >= before + 2, 10)
        (tmp_path / "code").unlink()
        before = count()
        await _until(lambda: count() > before, 10)
        await asyncio.sleep(0.5)
        status = await storage()
        assert status.last_error is None and status.consecutive_failures == 0
        assert status.last_success_at is not None
        # Once pushed, a quiet host stops pushing.
        await asyncio.sleep(1)
        quiet = count()
        await asyncio.sleep(1)
        assert count() == quiet
    finally:
        process.terminate()
        await process.wait()


async def test_shutdown_finishes_within_the_timeout_when_the_push_hangs(tmp_path: Path) -> None:
    endpoint, process = await _start_host(tmp_path)
    (tmp_path / "hang").touch()
    await call_host(endpoint, "counter", "bump", {}, key="k")
    started = time.monotonic()
    status, _ = await asyncio.to_thread(host.post, endpoint, Route.SHUTDOWN, b"{}", 30)
    assert status == 200 and await asyncio.wait_for(process.wait(), 10) == 0
    assert time.monotonic() - started < 5  # a running push plus the final one, 1 s each


def test_entry_fails_when_restore_times_out(tmp_path: Path) -> None:
    restore = [sys.executable, "-c", "import time; time.sleep(60)"]
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-m", "actant.sandbox.entry",
         EntryConfig(restore=RestoreConfig(argv=restore, timeout_s=0.5)).model_dump_json()],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )  # fmt: skip
    assert done.returncode == 1 and "restore timed out after 0.5s" in done.stderr
    assert time.monotonic() - started < 20


def test_stamp_gives_restored_files_their_object_mtimes(tmp_path: Path) -> None:
    from actant.sandbox.entry import stamp_mtimes

    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "a.json").write_text("{}")
    (tmp_path / "changed.txt").write_text("longer than remote")
    (tmp_path / "empty").write_text("")
    prefix = "s3://b/sandboxes/t1/"
    listing = [
        json.dumps({"key": f"{prefix}dir/a.json", "last_modified": "2026-01-02T03:04:05.123456789Z", "size": 2}),
        json.dumps({"key": f"{prefix}changed.txt", "last_modified": "2026-01-02T03:04:05Z", "size": 3}),
        json.dumps({"key": f"{prefix}empty", "last_modified": "2026-01-02T03:04:05+00:00"}),
        json.dumps({"key": f"{prefix}missing", "last_modified": "2026-01-02T03:04:05Z", "size": 1}),
        "not json",
    ]  # fmt: skip
    assert stamp_mtimes(listing, prefix, tmp_path) == 2
    stamp = 1767323045
    # Exact to the microsecond, never rounded past the object's time.
    assert (tmp_path / "dir" / "a.json").stat().st_mtime_ns == stamp * 10**9 + 123456000
    assert int((tmp_path / "empty").stat().st_mtime) == stamp
    # A size mismatch keeps its fresh mtime, so the next push uploads it.
    assert int((tmp_path / "changed.txt").stat().st_mtime) > stamp
