"""ModalSandboxProvider against a fake ``modal`` module: what it asks Modal for."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from actant.sandbox import Endpoint, SandboxSpec, Storage
from actant.sandbox import host
from actant.sandbox.entry import READY_FILE, stamp_mtimes
import actant.sandbox.modal as modal_backend
from actant.sandbox.protocol import (
    EntryConfig,
    Header,
    HostConfig,
    ImageUploadConfig,
    PushConfig,
    RestoreConfig,
    Route,
    StampConfig,
)
from actant.sandbox.modal import (
    DISK_PATH,
    MOUNT_PATH,
    PULL_FLAGS,
    S5CMD_URL,
    Location,
    ModalSandbox,
    ModalSandboxProvider,
    Mount,
    Restore,
    with_s5cmd,
)


class _Aio:
    def __init__(self, fn: Any) -> None:
        self.aio = fn


def _async(value: Any) -> Any:
    async def fn(*_: Any, **__: Any) -> Any:
        return value

    return fn


@dataclass(frozen=True)
class _Probe:
    tcp: int | None = None
    exec_argv: tuple[str, ...] | None = None


class _FakeModal:
    def __init__(
        self, *, ready_error: Exception | None = None, exit_code: int | None = None
    ) -> None:
        self.created: tuple[tuple[str, ...], dict[str, Any]] = ((), {})
        self.execs: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.tokens: list[int] = []
        self.terminated = False
        fake = self

        async def exec_(*argv: str, **kwargs: Any) -> Any:
            fake.execs.append((argv, kwargs))
            return SimpleNamespace(
                stdout=SimpleNamespace(read=_Aio(_async(""))),
                stderr=SimpleNamespace(read=_Aio(_async(""))),
                wait=_Aio(_async(0)),
            )

        async def wait_until_ready(timeout: int) -> None:
            del timeout
            if ready_error is not None:
                raise ready_error

        async def create_connect_token(port: int) -> Any:
            fake.tokens.append(port)
            return SimpleNamespace(url="https://sb.modal.host", token=f"tok{len(fake.tokens)}")

        async def terminate() -> None:
            fake.terminated = True

        self.sandbox = SimpleNamespace(
            object_id="sb-1",
            exec=_Aio(exec_),
            poll=_Aio(_async(exit_code)),
            get_tags=_Aio(lambda: _async(fake.created[1]["tags"])()),
            wait_until_ready=_Aio(wait_until_ready),
            create_connect_token=_Aio(create_connect_token),
            stderr=SimpleNamespace(read=_Aio(_async("restore exited 1: access denied"))),
            terminate=_Aio(terminate),
            wait=_Aio(_async(None)),
        )

        async def create(*args: str, **kwargs: Any) -> Any:
            fake.created = (args, kwargs)
            return fake.sandbox

        self.App = SimpleNamespace(lookup=_Aio(_async("app")))
        self.Sandbox = SimpleNamespace(create=_Aio(create), from_id=_Aio(_async(self.sandbox)))
        self.Secret = SimpleNamespace(
            from_name=lambda name: f"secret:{name}", from_dict=lambda env: ("dict", env)
        )
        self.CloudBucketMount = lambda bucket, **kw: ("mount", bucket, kw)
        self.Probe = SimpleNamespace(
            with_tcp=lambda port: _Probe(tcp=port),
            with_exec=lambda *argv: _Probe(exec_argv=argv),
        )


@pytest.fixture
def provider() -> ModalSandboxProvider:
    return ModalSandboxProvider(
        app_name="app",
        bucket="b",
        endpoint_url="https://r2.example",
        secret_name="r2",
    )


def _use(monkeypatch: pytest.MonkeyPatch, fake: _FakeModal) -> None:
    real = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", lambda name, *a: fake if name == "modal" else real(name, *a)
    )


S5 = ("s5cmd", "--endpoint-url", "https://r2.example")
PULL = (*S5, *PULL_FLAGS)


async def test_mount_without_service_has_no_entrypoint(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    sandbox = await provider.open(SandboxSpec(backend="modal"), agent_id="a", thread_id="t1")
    assert isinstance(sandbox, ModalSandbox) and await sandbox.endpoint() is None
    args, kw = fake.created
    assert args == () and kw["readiness_probe"] is None
    assert kw["outbound_cidr_allowlist"] == [] and kw["gpu"] is None and kw["secrets"] == []
    assert kw["region"] is None  # anywhere, unless the spec says
    assert kw["workdir"] == MOUNT_PATH and kw["env"] is None
    assert kw["volumes"][MOUNT_PATH] == (
        "mount",
        "b",
        {
            "key_prefix": "sandboxes/t1/",
            "bucket_endpoint_url": "https://r2.example",
            "secret": "secret:r2",
        },
    )
    assert fake.tokens == []
    assert (await sandbox.sync()).returncode == 0 and fake.execs == []


async def test_service_with_disk_sync_restores_then_serves_behind_a_connect_token(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    spec = SandboxSpec(
        backend="modal",
        storage=Storage.DISK_SYNC,
        services={"notes": "pkg.tools:Notes"},
        service_port=9000,
        gpu="L4",
        region=("us-east", "us-west"),
        network=True,
        secrets=("service",),
        scrub_env=("SERVICE_KEY", "AWS_SECRET_ACCESS_KEY"),
        env={"MODE": "test"},
        sync_interval_s=30,
        sync_timeout_s=120,
    )
    sandbox = await provider.open(spec, agent_id="a", thread_id="t1")
    assert isinstance(sandbox, ModalSandbox)
    args, kw = fake.created
    push = [*S5, "sync", f"{DISK_PATH}/", "s3://b/sandboxes/t1/"]
    restore = [*PULL, "s3://b/sandboxes/t1/*", f"{DISK_PATH}/"]
    assert list(args[:3]) == ["python", "-m", "actant.sandbox.entry"] and len(args) == 4
    assert EntryConfig.model_validate_json(args[3]) == EntryConfig(
        restore=RestoreConfig(
            argv=restore,
            timeout_s=1800,
            stamp=StampConfig(
                argv=["s5cmd", "--json", *S5[1:], "ls", "s3://b/sandboxes/t1/*"],
                prefix="s3://b/sandboxes/t1/",
                root=DISK_PATH,
            ),
        ),
        host=HostConfig(
            services={"notes": "pkg.tools:Notes"},
            port=9000,
            bind="0.0.0.0",
            scrub=["SERVICE_KEY", "AWS_SECRET_ACCESS_KEY"],
            push=PushConfig(argv=push, interval_s=30, timeout_s=120),
            # Images upload beside the thread's prefix, never inside what the push mirrors.
            images=ImageUploadConfig(
                destination="s3://b/actant-images/t1/",
                endpoint_url="https://r2.example",
                timeout_s=10,
            ),
        ),
    )
    assert kw["readiness_probe"] == _Probe(tcp=9000)
    assert "encrypted_ports" not in kw and "unencrypted_ports" not in kw
    assert kw["gpu"] == "L4" and "outbound_cidr_allowlist" not in kw
    assert kw["region"] == ["us-east", "us-west"]
    assert kw["secrets"] == ["secret:service", "secret:r2"]
    assert kw["env"] == {"MODE": "test"} and kw["tags"] == {"actant_thread": "t1"}
    assert kw["volumes"] == {} and kw["workdir"] == DISK_PATH
    # The token is minted on first use, then cached until a refresh.
    assert fake.tokens == []
    tok1 = Endpoint("https://sb.modal.host", {Header.AUTHORIZATION: "Bearer tok1"})
    assert await sandbox.endpoint() == tok1 and await sandbox.endpoint() == tok1
    assert fake.tokens == [9000]

    # Agent commands lose the scrubbed keys (argv, no shell) unless passed explicitly.
    await sandbox.exec(["python", "run.py"], timeout=5)
    await sandbox.exec(["python", "run.py"], timeout=5, env={"SERVICE_KEY": "mine"})
    assert [argv for argv, _ in fake.execs] == [
        ("timeout", "5", "env", "-uSERVICE_KEY", "-uAWS_SECRET_ACCESS_KEY", "python", "run.py"),
        ("timeout", "5", "env", "-uAWS_SECRET_ACCESS_KEY", "python", "run.py"),
    ]

    # sync keeps the bucket keys s5cmd needs.
    fake.execs.clear()
    await sandbox.sync()
    ((argv, kwargs),) = fake.execs
    assert (
        argv[:2] == ("timeout", "120")
        and list(argv[2:]) == push
        and kwargs["workdir"] == DISK_PATH
    )

    # attach returns the live handle (one poll, no new token); refresh re-mints.
    assert await provider.attach(spec, "sb-1") is sandbox and fake.tokens == [9000]
    tok2 = Endpoint("https://sb.modal.host", {Header.AUTHORIZATION: "Bearer tok2"})
    assert await sandbox.endpoint(refresh=True) == tok2 and await sandbox.endpoint() == tok2

    # Another worker's attach recovers the prefix from the tag.
    attached = await ModalSandboxProvider(
        app_name="app", bucket="b", endpoint_url="https://r2.example", secret_name="r2"
    ).attach(spec, "sb-1")
    assert isinstance(attached, ModalSandbox) and attached is not sandbox
    fake.execs.clear()
    await attached.sync()
    assert fake.execs[0][0][-1] == "s3://b/sandboxes/t1/"

    # close asks the host to shut down (it closes instances and pushes); if the host
    # cannot confirm, close pushes itself. Either way it terminates.
    posts: list[tuple[str, str]] = []

    def post(endpoint: Endpoint, path: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        posts.append((endpoint.headers[Header.AUTHORIZATION], path))
        return (401, b"") if len(posts) == 1 else (200, b"")

    monkeypatch.setattr(host, "post", post)
    fake.execs.clear()
    await sandbox.close()
    assert [p for _, p in posts] == [Route.SHUTDOWN] * 2 and posts[1][0] == "Bearer tok3"
    assert fake.execs == [] and fake.terminated
    monkeypatch.setattr(host, "post", lambda *_: (_ for _ in ()).throw(OSError("gone")))
    await sandbox.close()
    assert list(fake.execs[0][0][2:]) == push
    fake.sandbox.poll = _Aio(_async(0))
    with pytest.raises(KeyError):
        await provider.attach(spec, "sb-1")


async def test_disk_sync_without_service_waits_for_the_restore_marker(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC)
    sandbox = await provider.open(spec, agent_id="a", thread_id="t1")
    args, kw = fake.created
    config = EntryConfig.model_validate_json(args[3])
    assert config.restore is not None and config.host is None
    assert kw["readiness_probe"] == _Probe(exec_argv=("test", "-f", READY_FILE))
    assert await sandbox.endpoint() is None and fake.execs == []
    # Without network, s5cmd may reach only the bucket endpoint.
    assert kw["outbound_domain_allowlist"] == ["r2.example"]
    with pytest.raises(ValueError, match="endpoint_url"):
        await ModalSandboxProvider(app_name="app", bucket="b").open(
            spec, agent_id="a", thread_id="t"
        )


async def test_open_fails_with_the_entrypoint_stderr_when_never_ready(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal(ready_error=TimeoutError("probe"), exit_code=1)
    _use(monkeypatch, fake)
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, services={"t": "pkg:T"})
    with pytest.raises(RuntimeError, match="access denied"):
        await provider.open(spec, agent_id="a", thread_id="t1")
    assert fake.terminated


async def test_open_cancelled_before_ready_terminates_the_sandbox(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()

    async def never_ready(timeout: int) -> None:
        del timeout
        await asyncio.Event().wait()

    fake.sandbox.wait_until_ready = _Aio(never_ready)
    _use(monkeypatch, fake)
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC)
    opening = asyncio.ensure_future(provider.open(spec, agent_id="a", thread_id="t1"))
    while not fake.created[1]:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert fake.terminated


async def test_attach_to_an_exited_sandbox_is_gone(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    _use(monkeypatch, _FakeModal(exit_code=0))
    with pytest.raises(KeyError):
        await provider.attach(SandboxSpec(backend="modal"), "sb-1")


async def test_inline_bucket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    provider = ModalSandboxProvider(
        app_name="app",
        bucket="b",
        endpoint_url="https://abc.trycloudflare.com",
        bucket_env={"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"},
    )
    await provider.open(
        SandboxSpec(backend="modal", storage=Storage.DISK_SYNC), agent_id="a", thread_id="t"
    )
    creds = {"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"}
    assert fake.created[1]["secrets"] == [("dict", creds)]

    await provider.open(SandboxSpec(backend="modal"), agent_id="a", thread_id="t")
    assert fake.created[1]["secrets"] == []
    assert fake.created[1]["volumes"][MOUNT_PATH][2]["secret"] == ("dict", creds)


async def test_disk_sync_uploads_references_without_public_signing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    provider = ModalSandboxProvider(
        app_name="app", bucket="b", endpoint_url="http://10.0.0.5:9000"
    )
    spec = SandboxSpec(
        backend="modal", storage=Storage.DISK_SYNC, services={"t": "pkg:T"}, network=True
    )
    await provider.open(spec, agent_id="a", thread_id="t")
    config = EntryConfig.model_validate_json(fake.created[0][3]).host
    assert config is not None and config.images is not None
    assert config.images.destination == "s3://b/actant-images/t/"
    bytes_only = SandboxSpec(
        backend="modal",
        storage=Storage.DISK_SYNC,
        services={"t": "pkg:T"},
        network=True,
        upload_images=False,
    )
    await provider.open(bytes_only, agent_id="a", thread_id="t")
    host_config = EntryConfig.model_validate_json(fake.created[0][3]).host
    assert host_config is not None and host_config.images is None
    # Mounted storage never uploads images, so it needs no public endpoint either.
    mounted = SandboxSpec(backend="modal", services={"t": "pkg:T"})
    await provider.open(mounted, agent_id="a", thread_id="t")
    host_config = EntryConfig.model_validate_json(fake.created[0][3]).host
    assert host_config is not None and host_config.images is None

    public = ModalSandboxProvider(
        app_name="app",
        bucket="b",
        endpoint_url="http://10.0.0.5:9000",
    )
    timed = SandboxSpec(
        backend="modal",
        storage=Storage.DISK_SYNC,
        services={"t": "pkg:T"},
        network=True,
        image_upload_timeout_s=2.5,
    )
    await public.open(timed, agent_id="a", thread_id="t")
    host_config = EntryConfig.model_validate_json(fake.created[0][3]).host
    assert host_config is not None and host_config.images is not None
    assert host_config.images.timeout_s == 2.5
    assert host_config.images.endpoint_url == "http://10.0.0.5:9000"


def test_with_s5cmd_installs_the_pinned_binary() -> None:
    image = SimpleNamespace(run_commands=lambda *cmds: cmds)
    cmds = with_s5cmd(image)
    assert S5CMD_URL.endswith("/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz")
    assert S5CMD_URL in cmds[0]


def test_storage_accepts_the_enum_or_its_string_and_rejects_others() -> None:
    assert SandboxSpec(storage="disk_sync").storage is Storage.DISK_SYNC  # pyright: ignore[reportArgumentType]
    assert SandboxSpec().storage is Storage.MOUNT
    with pytest.raises(ValueError):
        SandboxSpec(storage="nfs")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="positive"):
        SandboxSpec(sync_timeout_s=0)


async def test_sync_and_close_are_bounded_when_modal_hangs(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, sync_timeout_s=0.1)
    sandbox = await provider.open(spec, agent_id="a", thread_id="t1")
    monkeypatch.setattr(modal_backend, "API_SLACK_S", 0.1)

    async def hang(*_: Any, **__: Any) -> Any:
        await asyncio.sleep(3600)

    fake.sandbox.exec = _Aio(hang)
    result = await asyncio.wait_for(sandbox.sync(), 5)
    assert result.timed_out and result.returncode == 124

    async def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("modal is down")

    fake.sandbox.terminate = _Aio(broken)
    await asyncio.wait_for(sandbox.close(), 5)  # neither hangs nor raises


async def test_a_seed_restores_and_copies_only_into_an_empty_thread_prefix(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, seed="templates/base/")
    await provider.open(spec, agent_id="a", thread_id="t1")
    config = EntryConfig.model_validate_json(fake.created[0][3])
    restore = config.restore
    assert restore is not None and restore.seed is not None and restore.seed.stamp is not None
    # The thread's own prefix first; the seed only if that is empty.
    assert restore.argv == [*PULL, "s3://b/sandboxes/t1/*", f"{DISK_PATH}/"]
    assert restore.seed.argv == [*PULL, "s3://b/templates/base/*", f"{DISK_PATH}/"]
    assert restore.seed.copy_argv == [*S5, "cp", "s3://b/templates/base/*", "s3://b/sandboxes/t1/"]
    assert restore.seed.stamp.prefix == "s3://b/templates/base/"
    assert restore.seed.stamp.argv[-1] == "s3://b/templates/base/*"
    push = provider.sync_argv("t1")
    assert push == [*S5, "sync", f"{DISK_PATH}/", "s3://b/sandboxes/t1/"]


def test_seed_is_a_disk_sync_key_prefix() -> None:
    with pytest.raises(ValueError, match="seed"):
        SandboxSpec(backend="modal", storage=Storage.MOUNT, seed="templates/base/")
    with pytest.raises(ValueError, match="seed"):
        SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, seed="templates/room")


class _ScenePlan(ModalSandboxProvider):
    """A scene: its own prefix at the root, and a capture from another prefix, read-only."""

    def restore_plan(self, thread_id: str) -> list[Restore]:
        return [
            Restore(Location("b", "captures/c1/"), "capture", push=False),
            Restore(self.thread_location(thread_id), "", push=True),
        ]


def test_a_restore_plan_pulls_every_entry_and_pushes_only_its_own() -> None:
    provider = _ScenePlan(app_name="app", bucket="b", endpoint_url="https://r2.example")
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, seed="templates/base/")
    restore = provider.entry_config(spec, "t1").restore
    assert restore is not None and restore.seed is not None
    assert restore.argv == [*PULL, "s3://b/sandboxes/t1/*", f"{DISK_PATH}/"]
    [capture] = restore.also
    assert capture.argv == [*PULL, "s3://b/captures/c1/*", f"{DISK_PATH}/capture/"]
    assert capture.stamp is not None and capture.stamp.root == f"{DISK_PATH}/capture"
    # The seed fills the pushed entry and is marked beside its prefix.
    assert restore.seed.copy_argv[-1] == "s3://b/sandboxes/t1/"
    assert provider.seed_marker("t1") == "s3://b/sandboxes/t1.actant-seeded"
    assert provider.sync_argv("t1") == [
        *S5, "sync", "--exclude", "root/sandbox/capture/*", f"{DISK_PATH}/", "s3://b/sandboxes/t1/"
    ]  # fmt: skip


def test_the_spec_sets_the_restore_budget() -> None:
    provider = _ScenePlan(app_name="app", bucket="b", endpoint_url="https://r2.example")
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, restore_timeout_s=420)
    restore = provider.entry_config(spec, "t1").restore
    assert restore is not None and restore.timeout_s == 420
    with pytest.raises(ValueError, match="restore_timeout_s"):
        SandboxSpec(restore_timeout_s=0)


def test_a_restore_pulled_again_keeps_files_already_on_the_disk_at_their_size() -> None:
    provider = _ScenePlan(app_name="app", bucket="b", endpoint_url="https://r2.example")
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC)
    first = provider.entry_config(spec, "t1").restore
    again = provider.entry_config(spec, "t1", size_only=True).restore
    assert first is not None and again is not None
    pulls = [(first.argv, again.argv), *((a.argv, b.argv) for a, b in zip(first.also, again.also))]
    for before, after in pulls:
        assert "--size-only" not in before
        at = before.index("sync") + 1
        assert after == [*before[:at], "--size-only", *before[at:]]


def test_a_restore_plan_is_checked() -> None:
    with pytest.raises(ValueError, match="prefix ending"):
        Location("b", "captures")
    with pytest.raises(ValueError, match="relative"):
        Restore(Location("b", ""), "/abs", push=False)
    with pytest.raises(ValueError, match="relative"):
        Restore(Location("b", ""), "a/../b", push=False)

    class _NoPush(ModalSandboxProvider):
        def restore_plan(self, thread_id: str) -> list[Restore]:
            return [Restore(self.thread_location(thread_id), "", push=False)]

    with pytest.raises(ValueError, match="exactly one"):
        _NoPush(app_name="app", bucket="b").sync_argv("t1")


class _MountedScene(ModalSandboxProvider):
    """A scene: its own prefix at the root, and a capture mounted read-only inside it."""

    def bucket_mounts(self, thread_id: str) -> list[Mount]:
        return [Mount(Location("b", f"captures/{thread_id}/"), "capture")]


async def test_a_mount_is_read_only_and_no_pull_push_or_stamp_touches_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    provider = _MountedScene(
        app_name="app", bucket="b", endpoint_url="https://r2.example", secret_name="r2"
    )
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, seed="templates/base/")
    await provider.open(spec, agent_id="a", thread_id="t1")
    args, kw = fake.created
    assert kw["volumes"] == {
        f"{DISK_PATH}/capture": (
            "mount",
            "b",
            {
                "key_prefix": "captures/t1/",
                "bucket_endpoint_url": "https://r2.example",
                "secret": "secret:r2",
                "read_only": True,
            },
        )
    }
    restore = EntryConfig.model_validate_json(args[3]).restore
    assert restore is not None and restore.seed is not None and restore.stamp is not None
    assert restore.argv == [
        *PULL, "--exclude", "sandboxes/t1/capture/*", "s3://b/sandboxes/t1/*", f"{DISK_PATH}/"
    ]  # fmt: skip
    assert restore.seed.argv == [
        *PULL, "--exclude", "templates/base/capture/*", "s3://b/templates/base/*",
        f"{DISK_PATH}/",
    ]  # fmt: skip
    assert restore.stamp.skip == ["s3://b/sandboxes/t1/capture/"]
    assert provider.sync_argv("t1") == [
        *S5, "sync", "--exclude", "root/sandbox/capture/*", f"{DISK_PATH}/", "s3://b/sandboxes/t1/"
    ]  # fmt: skip


def test_a_mount_skips_stamping_its_keys(tmp_path: Path) -> None:
    (tmp_path / "capture").mkdir()
    for name in ("room.py", "capture/f.jpg"):
        (tmp_path / name).write_text("x")
    listing = [
        f'{{"key": "s3://b/t/{name}", "size": 1, "last_modified": "2020-01-01T00:00:00Z"}}'
        for name in ("room.py", "capture/f.jpg")
    ]
    assert stamp_mtimes(listing, "s3://b/t/", tmp_path, ["s3://b/t/capture/"]) == 1


def test_mounts_are_checked_against_the_plan_and_each_other() -> None:
    with pytest.raises(ValueError, match="root"):
        Mount(Location("b", "c/"), "")
    with pytest.raises(ValueError, match="relative"):
        Mount(Location("b", "c/"), "a/../b")

    @dataclass
    class _Over(ModalSandboxProvider):
        mounts: tuple[str, ...] = ()
        restores: tuple[str, ...] = ()

        def restore_plan(self, thread_id: str) -> list[Restore]:
            others = [
                Restore(Location("b", f"r{i}/"), p, push=False)
                for i, p in enumerate(self.restores)
            ]
            return [Restore(self.thread_location(thread_id), "", push=True), *others]

        def bucket_mounts(self, thread_id: str) -> list[Mount]:
            del thread_id
            return [Mount(Location("b", f"m{i}/"), p) for i, p in enumerate(self.mounts)]

    for mounts, restores, match in [
        (("capture",), ("capture",), "under mount"),
        (("capture",), ("capture/x",), "under mount"),
        (("capture", "capture/x"), (), "overlaps"),
        (("capture/x", "capture"), (), "overlaps"),
    ]:
        provider = _Over(app_name="app", bucket="b", mounts=mounts, restores=restores)
        with pytest.raises(ValueError, match=match):
            provider.sync_argv("t1")
    fine = _Over(app_name="app", bucket="b", mounts=("capture",), restores=("inputs",))
    assert "root/sandbox/capture/*" in fine.sync_argv("t1")
