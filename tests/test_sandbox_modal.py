"""ModalSandboxProvider against a fake ``modal`` module: what it asks Modal for."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from actant.sandbox import Endpoint, SandboxSpec, Storage
from actant.sandbox import host
from actant.sandbox.entry import READY_FILE
import actant.sandbox.modal as modal_backend
from actant.sandbox.protocol import (
    EntryConfig,
    Header,
    HostConfig,
    PushConfig,
    RestoreConfig,
    Route,
    StampConfig,
)
from actant.sandbox.modal import (
    DISK_PATH,
    MOUNT_PATH,
    S5CMD_URL,
    ModalSandbox,
    ModalSandboxProvider,
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
        app_name="app", bucket="b", endpoint_url="https://r2.example", secret_name="r2"
    )


def _use(monkeypatch: pytest.MonkeyPatch, fake: _FakeModal) -> None:
    real = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", lambda name, *a: fake if name == "modal" else real(name, *a)
    )


S5 = ("s5cmd", "--endpoint-url", "https://r2.example")


async def test_mount_without_toolset_has_no_entrypoint(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    sandbox = await provider.open(SandboxSpec(backend="modal"), agent_id="a", thread_id="t1")
    assert isinstance(sandbox, ModalSandbox) and await sandbox.endpoint() is None
    args, kw = fake.created
    assert args == () and kw["readiness_probe"] is None
    assert kw["outbound_cidr_allowlist"] == [] and kw["gpu"] is None and kw["secrets"] == []
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


async def test_toolset_with_disk_sync_restores_then_serves_behind_a_connect_token(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    spec = SandboxSpec(
        backend="modal",
        storage=Storage.DISK_SYNC,
        toolsets={"notes": "pkg.tools:Notes"},
        toolset_port=9000,
        gpu="L4",
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
    push = [*S5, "sync", "--delete", f"{DISK_PATH}/", "s3://b/sandboxes/t1/"]
    restore = [*S5, "sync", "s3://b/sandboxes/t1/*", f"{DISK_PATH}/"]
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
            toolsets={"notes": "pkg.tools:Notes"},
            port=9000,
            bind="0.0.0.0",
            scrub=["SERVICE_KEY", "AWS_SECRET_ACCESS_KEY"],
            push=PushConfig(argv=push, interval_s=30, timeout_s=120),
        ),
    )
    assert kw["readiness_probe"] == _Probe(tcp=9000)
    assert "encrypted_ports" not in kw and "unencrypted_ports" not in kw
    assert kw["gpu"] == "L4" and "outbound_cidr_allowlist" not in kw
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


async def test_disk_sync_without_toolset_waits_for_the_restore_marker(
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
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, toolsets={"t": "pkg:T"})
    with pytest.raises(RuntimeError, match="access denied"):
        await provider.open(spec, agent_id="a", thread_id="t1")
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
