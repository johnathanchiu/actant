"""Remote tools: a real ``actant.sandbox.host`` server behind a LocalSandbox."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

import sandbox_host_tools as tools
from actant.sandbox import LocalSandbox, RemoteHost, RemoteTool
from actant.tools.base import CallContext

TESTS = str(Path(__file__).parent)
FACTORY = "sandbox_host_tools:make_ctx"


@pytest_asyncio.fixture
async def host(tmp_path: Path) -> AsyncIterator[RemoteHost]:
    # A unix socket path must fit in ~104 bytes; pytest's tmp_path does not.
    short = tempfile.mkdtemp(prefix="ah", dir="/tmp")
    remote = RemoteHost(
        LocalSandbox(tmp_path, env={"PYTHONPATH": TESTS}), socket=f"{short}/h.sock"
    )
    await remote.start(FACTORY, after=f"touch {tmp_path}/after-ran")
    yield remote
    await remote.sandbox.exec(["pkill", "-f", f"host serve --socket {remote.socket}"], timeout=10)
    shutil.rmtree(short, ignore_errors=True)


def _path(fn: object) -> str:
    return f"sandbox_host_tools:{fn.__qualname__}"  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_context_state_persists_and_after_runs(host: RemoteHost, tmp_path: Path) -> None:
    first = await host.call("c1", _path(tools.bump), {"by": 2}, init={"start": 10})
    second = await host.call("c1", _path(tools.bump), {})
    other = await host.call("c2", _path(tools.bump), {})
    assert (first.text, second.text, other.text) == (
        '{"count": 12}',
        '{"count": 13}',
        '{"count": 1}',
    )
    for _ in range(50):
        if (tmp_path / "after-ran").exists():
            break
        await asyncio.sleep(0.05)
    assert (tmp_path / "after-ran").exists()


@pytest.mark.asyncio
async def test_calls_run_in_parallel(host: RemoteHost) -> None:
    started = time.monotonic()
    results = await asyncio.gather(
        *(host.call(f"p{i}", _path(tools.slow), {"seconds": 0.5}) for i in range(4))
    )
    assert [r.text for r in results] == ["p0", "p1", "p2", "p3"]
    assert time.monotonic() - started < 1.8


@pytest.mark.asyncio
async def test_images_round_trip_and_errors_keep_the_server_up(host: RemoteHost) -> None:
    image = await host.call("img", _path(tools.render), {"name": "out.png"})
    assert image.error is None and image.images == [("out.png", b"\x89PNG fake img")]

    failed = await host.call("img", _path(tools.boom), {})
    assert failed.error == "ValueError: no good" and "Traceback" not in (failed.error or "")
    assert (await host.call("img", _path(tools.bump), {})).text == '{"count": 1}'


@pytest.mark.asyncio
async def test_start_is_idempotent_and_remote_tool_returns_image_blocks(host: RemoteHost) -> None:
    await host.start(FACTORY)  # answers the ping; no second server
    ps = await host.sandbox.exec(["pgrep", "-f", f"host serve --socket {host.socket}"], timeout=10)
    assert len(ps.stdout.split()) == 1

    remote = RemoteTool(tools.render, context="rt", host=host)
    ctx = CallContext("a", "t", "r", "tc", "turn")
    result = await (await remote.build({"name": "r.png"}, ctx)).execute()
    assert result.output == "rendered" and result.content_blocks is not None
    assert result.content_blocks[-1]["source"]["media_type"] == "image/png"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_remote_tool_schema_drops_ctx_and_local_mode_runs_in_process() -> None:
    local = tools.Ctx("here")
    remote = RemoteTool(tools.bump, context="x", local=local)
    parameters = remote.schema["function"]["parameters"]  # pyright: ignore[reportIndexIssue]
    assert list(parameters["properties"]) == ["by"]  # pyright: ignore[reportIndexIssue]
    assert remote.path == "sandbox_host_tools:bump"

    ctx = CallContext("a", "t", "r", "tc", "turn")
    result = await (await remote.build({"by": 3}, ctx)).execute()
    assert result.output == '{"count": 3}' and local.count == 3
    failed = await (
        await RemoteTool(tools.boom, context="x", local=local).build({}, ctx)
    ).execute()
    assert failed.error is not None and failed.error.startswith("ValueError: no good")


@pytest.mark.asyncio
async def test_call_overhead(host: RemoteHost) -> None:
    await host.call("o", _path(tools.bump), {})
    started = time.monotonic()
    for _ in range(10):
        await host.call("o", _path(tools.bump), {})
    print(f"\nper-call overhead: {(time.monotonic() - started) * 100:.1f} ms")


async def _stop(remote: RemoteHost) -> None:
    await remote.sandbox.exec(["pkill", "-f", f"host serve --socket {remote.socket}"], timeout=10)


@pytest.mark.asyncio
async def test_large_arguments_travel_by_file_and_exec_errors_become_results(
    host: RemoteHost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await host.call("big", _path(tools.size), {"text": "x" * 2_000_000})
    assert (result.error, result.text) == (None, "2000000")
    assert list((tmp_path / ".actant" / "requests").iterdir()) == []

    async def broken(*_: object, **__: object) -> object:
        raise OSError("Argument list too long")

    monkeypatch.setattr(host.sandbox, "exec", broken)
    failed = await host.call("big", _path(tools.bump), {})
    assert failed.error is not None and "Argument list too long" in failed.error


@pytest.mark.asyncio
async def test_scrub_env_hides_keys_from_commands_but_not_the_host(tmp_path: Path) -> None:
    sandbox = LocalSandbox(
        tmp_path, env={"PYTHONPATH": TESTS, "KEY_A": "a", "KEY_B": "b"}, scrub_env=("KEY_A",)
    )
    assert (await sandbox.exec(["sh", "-c", "echo ${KEY_A:-gone}"], timeout=10)).stdout == "gone\n"
    kept = await sandbox.exec(["sh", "-c", "echo $KEY_A"], timeout=10, keep_env=True)
    assert kept.stdout == "a\n"

    remote = RemoteHost(sandbox)
    try:
        await remote.start(FACTORY, scrub=("KEY_B",))
        a = await remote.call("e", _path(tools.env), {"name": "KEY_A"})
        b = await remote.call("e", _path(tools.env), {"name": "KEY_B"})
        scrub = await remote.call("e", _path(tools.env), {"name": "ACTANT_SCRUB_ENV"})
        assert a.text == '{"host": "a", "scrubbed": null}'
        assert b.text == '{"host": "b", "scrubbed": null}'
        assert '"host": "KEY_A,KEY_B"' in scrub.text
        await _stop(remote)
        await asyncio.sleep(0.2)
        await remote.start(FACTORY)  # no scrub: the spec's list survives
        scrub = await remote.call("e", _path(tools.env), {"name": "ACTANT_SCRUB_ENV"})
        assert '"host": "KEY_A"' in scrub.text
    finally:
        await _stop(remote)


@pytest.mark.asyncio
async def test_hosts_of_two_sandboxes_are_isolated_and_concurrent_starts_share_one(
    tmp_path: Path,
) -> None:
    a = LocalSandbox(tmp_path / "a", env={"PYTHONPATH": TESTS})
    b = LocalSandbox(tmp_path / "b", env={"PYTHONPATH": TESTS})
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    hosts = [RemoteHost(a), RemoteHost(a), RemoteHost(b)]
    assert hosts[0].socket == hosts[1].socket != hosts[2].socket
    assert len(hosts[0].socket) < 100
    try:
        await asyncio.gather(*(h.start(FACTORY) for h in hosts), hosts[0].start(FACTORY))
        for h in (hosts[0], hosts[2]):
            ps = await h.sandbox.exec(
                ["pgrep", "-f", f"host serve --socket {h.socket}"], timeout=10
            )
            assert len(ps.stdout.split()) == 1
        assert (await hosts[0].call("c", _path(tools.bump), {})).text == '{"count": 1}'
        assert (await hosts[1].call("c", _path(tools.bump), {})).text == '{"count": 2}'
        assert (await hosts[2].call("c", _path(tools.bump), {})).text == '{"count": 1}'
    finally:
        for h in (hosts[0], hosts[2]):
            await _stop(h)


@pytest.mark.asyncio
async def test_a_failing_after_hook_is_logged_and_a_dead_host_restarts(tmp_path: Path) -> None:
    remote = RemoteHost(LocalSandbox(tmp_path, env={"PYTHONPATH": TESTS}))
    try:
        await remote.start(FACTORY, after="echo push denied >&2; exit 3")
        await remote.call("d", _path(tools.bump), {})
        for _ in range(50):
            if "after hook exited 3" in await remote.log():
                break
            await asyncio.sleep(0.05)
        assert "push denied" in await remote.log()

        await _stop(remote)
        await asyncio.sleep(0.2)
        result = await remote.call("d", _path(tools.bump), {})
        assert (result.error, result.text) == (None, '{"count": 1}')
    finally:
        await _stop(remote)


def test_adapt_reads_only_images_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from actant.sandbox.host import adapt

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ok.png").write_bytes(b"png")
    (tmp_path / "notes.txt").write_text("secret")
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"png")
    response = adapt(tools.Rendered("r", ["ok.png", "notes.txt", str(outside), "../outside.png"]))
    assert [image["path"] for image in response["images"]] == ["ok.png"]  # pyright: ignore[reportGeneralTypeIssues]
    assert str(response["text"]).count("dropped") == 3


@pytest.mark.asyncio
async def test_a_failed_factory_does_not_evict_a_newer_context() -> None:
    from actant.sandbox.host import Server

    server = Server(FACTORY, None)
    gate = asyncio.Event()

    async def failing(key: str, init: object) -> object:
        await gate.wait()
        raise RuntimeError("factory failed")

    server.factory = failing
    first = asyncio.create_task(server.context("k", None))
    await asyncio.sleep(0)
    newer: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    server.contexts["k"] = newer
    gate.set()
    with pytest.raises(RuntimeError):
        await first
    assert server.contexts["k"] is newer
