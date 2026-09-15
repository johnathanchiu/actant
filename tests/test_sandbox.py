"""Sandboxes: the local backend, the registry, and tool injection."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from actant.runtime.stores import InMemoryRuntimeStores
from actant.sandbox import LocalSandbox, LocalSandboxProvider, Sandbox, SandboxSpec
from actant.sandbox.registry import VERIFIED_FOR_S, SandboxRegistry
from actant.tools.base import CallContext
from actant.tools.function import tool


def _ctx(sandbox: Sandbox | None = None) -> CallContext:
    return CallContext("a", "t", "r", "tc", "turn", sandbox=sandbox)


# === local backend ===


@pytest.mark.asyncio
async def test_local_sandbox_reads_writes_lists_and_runs(tmp_path: Path) -> None:
    sb = LocalSandbox(tmp_path)
    await sb.write("work/hello.txt", b"hi")
    assert await sb.read("work/hello.txt") == b"hi"
    [entry] = await sb.ls("work/*.txt")
    assert (entry.path, entry.size) == ("work/hello.txt", 2)

    result = await sb.exec(
        ["python", "-c", "print(open('hello.txt').read())"], cwd="work", timeout=30
    )
    assert (result.returncode, result.stdout.strip(), result.timed_out) == (0, "hi", False)


@pytest.mark.asyncio
async def test_local_sandbox_times_out_with_the_conventional_code(tmp_path: Path) -> None:
    result = await LocalSandbox(tmp_path).exec(
        ["python", "-c", "import time; time.sleep(5)"], timeout=0.5
    )
    assert result.timed_out and result.returncode == 124


@pytest.mark.asyncio
async def test_local_sandbox_rejects_paths_that_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        await LocalSandbox(tmp_path / "root").read("../outside")


@pytest.mark.asyncio
async def test_local_provider_keys_one_directory_per_thread(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(tmp_path)
    sb = await provider.open(SandboxSpec(), agent_id="a", thread_id="t1")
    assert Path(sb.id) == tmp_path / "t1"
    assert (await provider.attach(SandboxSpec(), sb.id)).id == sb.id
    with pytest.raises(KeyError):
        await provider.attach(SandboxSpec(), str(tmp_path / "never"))


# === registry ===


@dataclass
class _Recording:
    """A provider that counts opens and can forget a sandbox on demand."""

    root: Path
    opened: int = 0
    attached: list[str] = field(default_factory=list)
    gone: set[str] = field(default_factory=set)
    latency_s: float = 0.0

    async def open(self, spec: SandboxSpec, *, agent_id: str, thread_id: str) -> Sandbox:
        self.opened += 1
        root = self.root / f"{thread_id}-{self.opened}"
        root.mkdir()
        await asyncio.sleep(self.latency_s)
        return _Closable(root)

    async def attach(self, spec: SandboxSpec, sandbox_id: str) -> Sandbox:
        self.attached.append(sandbox_id)
        await asyncio.sleep(self.latency_s)
        if sandbox_id in self.gone:
            raise KeyError(sandbox_id)
        return _Closable(Path(sandbox_id))


class _Closable(LocalSandbox):
    closed: set[str] = set()

    async def close(self) -> None:
        _Closable.closed.add(self.id)


def _expire(registry: SandboxRegistry) -> None:
    """Age every verified handle past the window, as if the seconds had passed."""
    for verified in registry._live.values():
        verified.at -= VERIFIED_FOR_S


@pytest.mark.asyncio
async def test_registry_opens_once_persists_the_id_and_reattaches(tmp_path: Path) -> None:
    stores = InMemoryRuntimeStores()
    provider = _Recording(tmp_path)
    spec = SandboxSpec(backend="fake")
    registry = SandboxRegistry({"fake": provider}, stores.threads)

    first = await registry.for_thread(spec, "a", "t")
    assert provider.opened == 1
    assert (await stores.threads.get("a", "t")).sandbox_id == first.id

    # Verified moments ago: served without asking the backend.
    assert (await registry.for_thread(spec, "a", "t")).id == first.id
    assert provider.opened == 1 and provider.attached == []

    # Another worker attaches by the persisted id.
    other = SandboxRegistry({"fake": provider}, stores.threads)
    assert (await other.for_thread(spec, "a", "t")).id == first.id
    assert provider.attached == [first.id] and provider.opened == 1

    # A whole-thread update never clobbers the id a claim set.
    thread = await stores.threads.get("a", "t")
    thread.sandbox_id = None
    await stores.threads.update(thread)
    assert (await stores.threads.get("a", "t")).sandbox_id == first.id

    await registry.close("a", "t", forget=True)
    assert (await stores.threads.get("a", "t")).sandbox_id is None


@pytest.mark.asyncio
async def test_registry_replaces_a_reclaimed_sandbox_after_the_window(tmp_path: Path) -> None:
    stores = InMemoryRuntimeStores()
    provider = _Recording(tmp_path)
    spec = SandboxSpec(backend="fake")
    registry = SandboxRegistry({"fake": provider}, stores.threads)
    first = await registry.for_thread(spec, "a", "t")

    provider.gone.add(first.id)
    _expire(registry)
    fresh = await registry.for_thread(spec, "a", "t")
    assert fresh.id != first.id and provider.opened == 2
    assert provider.attached == [first.id]
    assert (await stores.threads.get("a", "t")).sandbox_id == fresh.id


@pytest.mark.asyncio
async def test_registry_threads_never_wait_on_each_other(tmp_path: Path) -> None:
    stores = InMemoryRuntimeStores()
    provider = _Recording(tmp_path, latency_s=0.05)
    spec = SandboxSpec(backend="fake")
    registry = SandboxRegistry({"fake": provider}, stores.threads)
    threads = [f"t{i}" for i in range(100)]
    await asyncio.gather(*(registry.for_thread(spec, "a", t) for t in threads))
    _expire(registry)

    started = time.monotonic()
    await asyncio.gather(*(registry.for_thread(spec, "a", t) for t in threads))
    assert len(provider.attached) == 100
    assert time.monotonic() - started < 0.5  # serialized, it would take 5 s


@pytest.mark.asyncio
async def test_registry_shares_one_attach_per_thread(tmp_path: Path) -> None:
    stores = InMemoryRuntimeStores()
    provider = _Recording(tmp_path, latency_s=0.05)
    spec = SandboxSpec(backend="fake")
    registry = SandboxRegistry({"fake": provider}, stores.threads)
    first = await registry.for_thread(spec, "a", "t")
    _expire(registry)

    got = await asyncio.gather(*(registry.for_thread(spec, "a", "t") for _ in range(50)))
    assert {sb.id for sb in got} == {first.id} and provider.attached == [first.id]


@pytest.mark.asyncio
async def test_two_workers_opening_one_thread_keep_one_sandbox(tmp_path: Path) -> None:
    stores = InMemoryRuntimeStores()
    provider = _Recording(tmp_path, latency_s=0.05)
    spec = SandboxSpec(backend="fake")
    await stores.threads.get_or_create("a", "t")
    workers = [SandboxRegistry({"fake": provider}, stores.threads) for _ in range(2)]

    got = await asyncio.gather(*(w.for_thread(spec, "a", "t") for w in workers))
    winner = (await stores.threads.get("a", "t")).sandbox_id
    assert provider.opened == 2
    assert [sb.id for sb in got] == [winner, winner]
    [loser] = {str(tmp_path / "t-1"), str(tmp_path / "t-2")} - {winner}
    assert loser in _Closable.closed and winner not in _Closable.closed


@pytest.mark.asyncio
async def test_registry_names_the_missing_backend() -> None:
    registry = SandboxRegistry({}, InMemoryRuntimeStores().threads)
    with pytest.raises(KeyError, match="modal"):
        registry.provider("modal")


# === function tools receive the context and the sandbox ===


@pytest.mark.asyncio
async def test_a_sandbox_parameter_is_injected_and_hidden_from_the_schema(tmp_path: Path) -> None:
    @tool
    async def write_note(text: str, sandbox: Sandbox) -> str:
        """Write a note."""
        await sandbox.write("note.txt", text.encode())
        return "written"

    properties = write_note.schema["function"]["parameters"]["properties"]  # type: ignore[index]
    assert set(properties) == {"text"}
    assert write_note.needs_sandbox

    sb = LocalSandbox(tmp_path)
    result = await (await write_note.build({"text": "hi"}, _ctx(sb))).execute()
    assert result.output == "written" and (tmp_path / "note.txt").read_bytes() == b"hi"

    with pytest.raises(RuntimeError, match="takes a Sandbox"):
        await write_note.build({"text": "hi"}, _ctx())


@pytest.mark.asyncio
async def test_a_call_context_parameter_tells_a_tool_its_thread() -> None:
    @tool
    def whoami(ctx: CallContext) -> str:
        """Which thread am I on?"""
        return ctx.thread_id

    assert not whoami.needs_sandbox
    assert whoami.schema["function"]["parameters"]["properties"] == {}  # type: ignore[index]
    result = await (await whoami.build({}, _ctx())).execute()
    assert result.output == "t"


def test_local_ls_cannot_leave_the_root(tmp_path) -> None:
    from actant.sandbox.local import LocalSandbox

    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("x")
    sandbox = LocalSandbox(root)
    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(sandbox.ls("../*"))
    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(sandbox.ls("/etc/*"))
