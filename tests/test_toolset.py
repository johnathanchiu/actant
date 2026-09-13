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
from typing import TYPE_CHECKING

import pytest

from actant.sandbox import Endpoint, LocalSandbox, LocalSandboxProvider, SandboxSpec
from actant.sandbox import host
from actant.tools import LocalRunner, RemoteRunner, SandboxRunner, tools, toolset_schema
from actant.tools.base import CallContext, ToolResult
from actant.tools.toolset import call_host
from toolset_fixtures import Counter

if TYPE_CHECKING:
    from some_module_that_only_type_checkers_see import Thing  # noqa: F401

TESTS = str(Path(__file__).parent)
TOOLSET = "toolset_fixtures:Counter"


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
    schemas = {s["function"]["name"]: s["function"] for s in toolset_schema(Sample)}  # pyright: ignore[reportIndexIssue]
    assert set(schemas) == {"inherited", "search"}
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
        toolset=TOOLSET,
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
    return await (await tool.build(args, ctx)).execute()


async def test_local_and_remote_runners_give_identical_results(
    sandbox: LocalSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sandbox.endpoint is not None
    monkeypatch.chdir(sandbox.root)
    local = LocalRunner(Counter())
    remote = RemoteRunner(sandbox.endpoint, key="k1")
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
    assert sandbox.endpoint is not None
    a = RemoteRunner(sandbox.endpoint, key="a", init={"start": 10})
    b = RemoteRunner(sandbox.endpoint, key="b")
    # Two first calls on one key race; both must land on one instance.
    first = await asyncio.gather(_call(a, "bump"), _call(a, "bump"))
    assert sorted(json.loads(str(r.output))["count"] for r in first) == [11, 12]
    assert json.loads(str((await _call(b, "bump")).output)) == {"count": 1}
    assert (await _call(a, "opens")).output == "2"

    started = time.monotonic()
    await asyncio.gather(*[_call(a, "wait", seconds=0.5) for _ in range(5)])
    assert time.monotonic() - started < 2


async def test_sandbox_runner_uses_the_thread_sandbox(sandbox: LocalSandbox) -> None:
    runner = SandboxRunner(init={"start": 5})
    (bump,) = [t for t in tools(Counter, runner) if t.name == "bump"]
    assert bump.needs_sandbox
    result = await _call(runner, "bump", _ctx(sandbox, thread="x"))
    assert json.loads(str(result.output)) == {"count": 6}
    failed = await _call(runner, "bump", _ctx(None))
    assert failed.error and "serves no toolset" in failed.error


async def test_bad_token_unknown_method_and_host_survives_errors(sandbox: LocalSandbox) -> None:
    assert sandbox.endpoint is not None
    wrong = Endpoint(sandbox.endpoint.url, {"Authorization": "Bearer nope"})
    denied = await call_host(wrong, "k", {}, "bump", {})
    assert denied.error and "bearer" in denied.error
    unknown = await call_host(sandbox.endpoint, "k", {}, "__init__", {})
    assert unknown.error and "unknown tool method" in unknown.error
    assert (await call_host(sandbox.endpoint, "k", {}, "fail", {})).error
    assert (await call_host(sandbox.endpoint, "k", {}, "length", {"text": "ok"})).output == "2"


async def test_large_arguments_and_images_round_trip(sandbox: LocalSandbox) -> None:
    assert sandbox.endpoint is not None
    runner = RemoteRunner(sandbox.endpoint, key="big")
    text = "x" * (2 * 1024 * 1024)
    assert (await _call(runner, "length", text=text)).output == str(len(text))
    result = await _call(runner, "picture", size=5 * 1024 * 1024)
    assert result.content_blocks
    data = base64.b64decode(result.content_blocks[2]["source"]["data"])  # pyright: ignore[reportIndexIssue]
    assert len(data) == 5 * 1024 * 1024 + 8 and data.startswith(b"\x89PNG")


async def test_script_env_scrubs_what_the_host_keeps(sandbox: LocalSandbox) -> None:
    assert sandbox.endpoint is not None
    runner = RemoteRunner(sandbox.endpoint, key="env")
    result = json.loads(str((await _call(runner, "env", name="SERVICE_KEY")).output))
    assert result == {"host": "k", "script": None}
    token = json.loads(str((await _call(runner, "env", name=host.TOKEN_ENV)).output))
    assert token == {"host": None, "script": None}
    ran = await sandbox.exec(["sh", "-c", "echo ${SERVICE_KEY:-gone}"], timeout=10)
    assert ran.stdout.strip() == "gone"


async def test_attach_reuses_the_running_host_and_close_stops_it(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(tmp_path)
    spec = SandboxSpec(toolset=TOOLSET, env={"PYTHONPATH": TESTS})
    opened = await provider.open(spec, agent_id="a", thread_id="t")
    try:
        attached = await provider.attach(spec, opened.id)
        assert attached is opened and opened.endpoint is not None
        await _call(RemoteRunner(opened.endpoint, key="k"), "bump")
    finally:
        await opened.close()
    restarted = await provider.attach(spec, opened.id)
    try:
        assert restarted is not opened and restarted.endpoint != opened.endpoint
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
        "--restore",
        json.dumps(restore),
        "--",
        "--toolset",
        TOOLSET,
        "--bind",
        "127.0.0.1",
        "--port",
        "0",
        "--push",
        json.dumps(push),
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
    await call_host(endpoint, "k", {}, "note", {"path": "a.txt", "text": "1"})
    for _ in range(50):
        if log.exists():
            break
        await asyncio.sleep(0.1)
    assert log.read_text().startswith("push")
    # A burst coalesces into far fewer pushes than calls: one running, one pending.
    await asyncio.gather(*[call_host(endpoint, "k", {}, "bump", {}) for _ in range(20)])
    await asyncio.sleep(1.5)
    assert 2 <= len(log.read_text().splitlines()) < 10
    process.terminate()
    await process.wait()
    assert process.stderr is not None
    assert "storage push exited 3" in (await process.stderr.read()).decode()


def test_entry_exits_when_restore_fails(tmp_path: Path) -> None:
    restore = [sys.executable, "-c", "import sys; sys.stderr.write('access denied'); sys.exit(1)"]
    done = subprocess.run(
        [sys.executable, "-m", "actant.sandbox.entry", "--restore", json.dumps(restore)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 1 and "access denied" in done.stderr
