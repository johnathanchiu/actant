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
