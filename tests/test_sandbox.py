"""Sandboxes: the local backend, the registry, and tool injection."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from actant.sandbox import LocalSandbox, LocalSandboxProvider, Sandbox, SandboxSpec
from actant.sandbox.registry import VERIFIED_FOR_S, SandboxRegistry
from actant.sandbox.store import InMemorySandboxStore
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
    sb = await provider.open(SandboxSpec(), sandbox_id="t1")
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

    specs: list[SandboxSpec] = field(default_factory=list)

    async def open(self, spec: SandboxSpec, *, sandbox_id: str) -> Sandbox:
        self.opened += 1
        self.specs.append(spec)
        root = self.root / f"{sandbox_id}-{self.opened}"
        root.mkdir()
        await asyncio.sleep(self.latency_s)
        return _Closable(root)

    async def attach(self, spec: SandboxSpec, provider_id: str) -> Sandbox:
        self.attached.append(provider_id)
        await asyncio.sleep(self.latency_s)
        if provider_id in self.gone:
            raise KeyError(provider_id)
        return _Closable(Path(provider_id))


class _Closable(LocalSandbox):
    closed: set[str] = set()

    async def close(self) -> None:
        _Closable.closed.add(self.id)


def _expire(registry: SandboxRegistry) -> None:
    """Age every verified handle past the window, as if the seconds had passed."""
    for verified in registry._live.values():
        verified.at -= VERIFIED_FOR_S


@pytest.mark.asyncio
async def test_registry_opens_a_key_once_records_it_and_others_attach(tmp_path: Path) -> None:
    store = InMemorySandboxStore()
    provider = _Recording(tmp_path)
    spec = SandboxSpec(backend="fake", env={"A": "1"})
    registry = SandboxRegistry({"fake": provider}, store)

    first = await registry.open("scan_1", spec)
    assert provider.opened == 1
    record = await store.get("scan_1")
    assert record is not None and record.provider_id == first.id and record.spec == spec

    # Verified moments ago: served without asking the backend.
    assert (await registry.open("scan_1", spec)).id == first.id
    assert (await registry.attach("scan_1")).id == first.id
    assert provider.opened == 1 and provider.attached == []

    # Another worker attaches by the key alone, with the recorded spec.
    other = SandboxRegistry({"fake": provider}, store)
    assert (await other.attach("scan_1")).id == first.id
    assert provider.attached == [first.id] and provider.opened == 1
    assert await other.spec("scan_1") == spec


@pytest.mark.asyncio
async def test_attaching_a_key_nobody_opened_is_refused(tmp_path: Path) -> None:
    registry = SandboxRegistry({"fake": _Recording(tmp_path)}, InMemorySandboxStore())
    with pytest.raises(KeyError, match="scan_1"):
        await registry.attach("scan_1")


@pytest.mark.asyncio
async def test_an_owner_open_with_a_new_spec_records_it_and_keeps_the_sandbox(
    tmp_path: Path,
) -> None:
    store = InMemorySandboxStore()
    provider = _Recording(tmp_path)
    registry = SandboxRegistry({"fake": provider}, store)
    first = await registry.open("k", SandboxSpec(backend="fake"))
    later = SandboxSpec(backend="fake", push_exclude=("capture",))

    assert (await registry.open("k", later)).id == first.id and provider.opened == 1
    record = await store.get("k")
    assert record is not None and record.spec == later and record.provider_id == first.id


@pytest.mark.asyncio
async def test_a_reclaimed_sandbox_is_reopened_from_its_recorded_spec(tmp_path: Path) -> None:
    store = InMemorySandboxStore()
    provider = _Recording(tmp_path)
    spec = SandboxSpec(backend="fake", env={"A": "1"})
    first = await SandboxRegistry({"fake": provider}, store).open("k", spec)
    provider.gone.add(first.id)

    fresh = await SandboxRegistry({"fake": provider}, store).attach("k")
    assert fresh.id != first.id and provider.opened == 2
    assert provider.specs[-1] == spec
    record = await store.get("k")
    assert record is not None and record.provider_id == fresh.id


@pytest.mark.asyncio
async def test_registry_keys_never_wait_on_each_other(tmp_path: Path) -> None:
    provider = _Recording(tmp_path, latency_s=0.05)
    spec = SandboxSpec(backend="fake")
    registry = SandboxRegistry({"fake": provider}, InMemorySandboxStore())
    keys = [f"t{i}" for i in range(100)]
    await asyncio.gather(*(registry.open(k, spec) for k in keys))
    _expire(registry)

    started = time.monotonic()
    await asyncio.gather(*(registry.attach(k) for k in keys))
    assert len(provider.attached) == 100
    assert time.monotonic() - started < 0.5  # serialized, it would take 5 s


@pytest.mark.asyncio
async def test_registry_shares_one_attach_per_key(tmp_path: Path) -> None:
    provider = _Recording(tmp_path, latency_s=0.05)
    registry = SandboxRegistry({"fake": provider}, InMemorySandboxStore())
    first = await registry.open("t", SandboxSpec(backend="fake"))
    _expire(registry)

    got = await asyncio.gather(*(registry.attach("t") for _ in range(50)))
    assert {sb.id for sb in got} == {first.id} and provider.attached == [first.id]


@pytest.mark.asyncio
async def test_two_workers_opening_one_key_keep_one_sandbox(tmp_path: Path) -> None:
    store = InMemorySandboxStore()
    provider = _Recording(tmp_path, latency_s=0.05)
    spec = SandboxSpec(backend="fake")
    workers = [SandboxRegistry({"fake": provider}, store) for _ in range(2)]

    got = await asyncio.gather(*(w.open("t", spec) for w in workers))
    record = await store.get("t")
    assert record is not None and provider.opened == 2
    assert [sb.id for sb in got] == [record.provider_id, record.provider_id]
    [loser] = {str(tmp_path / "t-1"), str(tmp_path / "t-2")} - {record.provider_id}
    assert loser in _Closable.closed and record.provider_id not in _Closable.closed


@pytest.mark.asyncio
async def test_registry_names_the_missing_backend() -> None:
    registry = SandboxRegistry({}, InMemorySandboxStore())
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


@dataclass
class _Hanging:
    """A provider whose open never finishes on its own, and records being cancelled."""

    started: bool = False
    cancelled: bool = False

    async def open(self, spec: SandboxSpec, *, sandbox_id: str) -> Sandbox:
        self.started = True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")

    async def attach(self, spec: SandboxSpec, provider_id: str) -> Sandbox:
        raise KeyError(provider_id)


@pytest.mark.asyncio
async def test_close_cancels_an_in_flight_open_instead_of_waiting_for_it() -> None:
    provider = _Hanging()
    registry = SandboxRegistry({"fake": provider}, InMemorySandboxStore())
    opening = asyncio.ensure_future(registry.open("t", SandboxSpec(backend="fake")))
    while not provider.started:
        await asyncio.sleep(0)

    await asyncio.wait_for(registry.close("t"), 5)
    assert provider.cancelled and registry._resolving == {}
    # The caller that was waiting is told why, not handed a cancellation of its own.
    with pytest.raises(RuntimeError, match="closed while opening"):
        await opening


@pytest.mark.asyncio
async def test_the_owner_closes_a_sandbox_another_worker_opened(tmp_path: Path) -> None:
    store = InMemorySandboxStore()
    provider = _Recording(tmp_path)
    spec = SandboxSpec(backend="fake")
    opened = await SandboxRegistry({"fake": provider}, store).open("k", spec)
    await SandboxRegistry({"fake": provider}, store).close("k")
    assert opened.id in _Closable.closed and provider.attached == [opened.id]
    assert await store.get("k") is None
    with pytest.raises(KeyError):
        await SandboxRegistry({"fake": provider}, store).attach("k")


@pytest.mark.asyncio
async def test_closing_survives_a_reclaimed_sandbox_and_an_unknown_key(tmp_path: Path) -> None:
    store = InMemorySandboxStore()
    provider = _Recording(tmp_path)
    opened = await SandboxRegistry({"fake": provider}, store).open(
        "k", SandboxSpec(backend="fake")
    )
    provider.gone.add(opened.id)
    await SandboxRegistry({"fake": provider}, store).close("k")
    await SandboxRegistry({"fake": provider}, store).close("never")
    assert await store.get("k") is None
