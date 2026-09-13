"""ModalSandboxProvider against a fake ``modal`` module: what it asks Modal for."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from actant.sandbox import SandboxSpec
from actant.sandbox.local import LocalSandbox
from actant.sandbox.modal import (
    DISK_PATH,
    MOUNT_PATH,
    ModalSandbox,
    S5CMD_URL,
    ModalSandboxProvider,
    sync_command,
    with_s5cmd,
)


class _Aio:
    def __init__(self, fn: Any) -> None:
        self.aio = fn


class _Process:
    def __init__(self, code: int, stderr: str) -> None:
        self.stdout = SimpleNamespace(read=_Aio(_async("")))
        self.stderr = SimpleNamespace(read=_Aio(_async(stderr)))
        self.wait = _Aio(_async(code))


def _async(value: Any) -> Any:
    async def fn(*_: Any, **__: Any) -> Any:
        return value

    return fn


class _FakeModal:
    def __init__(self, restore_code: int = 0, restore_stderr: str = "") -> None:
        self.created: dict[str, Any] = {}
        self.execs: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.tags: dict[str, str] = {}
        fake = self

        async def exec_(*argv: str, **kwargs: Any) -> _Process:
            fake.execs.append((argv, kwargs))
            return _Process(restore_code, restore_stderr)

        async def get_tags() -> dict[str, str]:
            return fake.tags

        self.sandbox = SimpleNamespace(
            object_id="sb-1",
            exec=_Aio(exec_),
            poll=_Aio(_async(None)),
            get_tags=_Aio(get_tags),
            terminate=_Aio(_async(None)),
            wait=_Aio(_async(None)),
        )

        async def create(**kwargs: Any) -> Any:
            fake.created = kwargs
            fake.tags = kwargs["tags"]
            return fake.sandbox

        self.App = SimpleNamespace(lookup=_Aio(_async("app")))
        self.Sandbox = SimpleNamespace(create=_Aio(create), from_id=_Aio(_async(self.sandbox)))
        self.Secret = SimpleNamespace(
            from_name=lambda name: f"secret:{name}", from_dict=lambda env: ("dict", env)
        )
        self.CloudBucketMount = lambda bucket, **kw: ("mount", bucket, kw)


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


async def test_mount_keeps_the_bucket_secret_on_the_mount(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    sandbox = await provider.open(SandboxSpec(backend="modal"), agent_id="a", thread_id="t1")
    assert isinstance(sandbox, ModalSandbox)
    kw = fake.created
    assert kw["block_network"] is True and kw["gpu"] is None and kw["secrets"] == []
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
    assert fake.execs == []  # no restore
    assert (await sandbox.sync()).returncode == 0 and fake.execs == []


async def test_disk_sync_restores_on_open_and_pushes_on_sync(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    fake = _FakeModal(restore_code=1, restore_stderr="ERROR no object found")
    _use(monkeypatch, fake)
    spec = SandboxSpec(
        backend="modal",
        storage="disk_sync",
        gpu="L4",
        network=True,
        secrets=("openai",),
        scrub_env=("OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"),
    )
    sandbox = await provider.open(spec, agent_id="a", thread_id="t1")
    assert isinstance(sandbox, ModalSandbox)
    kw = fake.created
    assert kw["gpu"] == "L4" and kw["block_network"] is False
    assert kw["secrets"] == ["secret:openai", "secret:r2"]
    assert kw["volumes"] == {} and kw["workdir"] == DISK_PATH
    assert kw["env"] == {"ACTANT_SCRUB_ENV": "OPENAI_API_KEY,AWS_SECRET_ACCESS_KEY"}

    endpoint = ("s5cmd", "--endpoint-url", "https://r2.example")
    ((argv, _),) = fake.execs
    exclude = ("--exclude", ".actant/*")
    assert argv[2:] == (*endpoint, "sync", *exclude, "s3://b/sandboxes/t1/*", f"{DISK_PATH}/")

    assert sync_command(provider, "t1") == (
        "s5cmd --endpoint-url https://r2.example sync --delete --exclude '.actant/*' "
        f"{DISK_PATH}/ s3://b/sandboxes/t1/"
    )
    fake.execs.clear()
    await sandbox.sync()
    ((argv, kwargs),) = fake.execs
    assert argv[2:] == (
        *endpoint,
        "sync",
        "--delete",
        *exclude,
        f"{DISK_PATH}/",
        "s3://b/sandboxes/t1/",
    )
    assert kwargs["workdir"] == DISK_PATH

    # Reattaching a live sandbox skips the restore but still knows its prefix.
    fake.execs.clear()
    attached = await provider.attach(spec, "sb-1")
    assert fake.execs == []
    assert isinstance(attached, ModalSandbox)
    await attached.sync()
    assert fake.execs[0][0][-1] == "s3://b/sandboxes/t1/"

    # Commands lose the scrubbed keys unless kept (or passed explicitly).
    fake.execs.clear()
    await sandbox.exec(["python", "run.py"], timeout=5)
    await sandbox.exec(["python", "run.py"], timeout=5, env={"OPENAI_API_KEY": "mine"})
    await sandbox.exec(["python", "run.py"], timeout=5, keep_env=True)
    assert [argv for argv, _ in fake.execs] == [
        ("env", "-uOPENAI_API_KEY", "-uAWS_SECRET_ACCESS_KEY", "timeout", "5", "python", "run.py"),
        ("env", "-uAWS_SECRET_ACCESS_KEY", "timeout", "5", "python", "run.py"),
        ("timeout", "5", "python", "run.py"),
    ]

    # close pushes the disk before terminating, and terminates even though this push exits 1.
    order: list[str] = []
    fake.execs.clear()
    fake.sandbox.terminate = _Aio(lambda: _record(order, "terminate"))
    await sandbox.close()
    assert fake.execs[0][0][2:6] == (*endpoint, "sync") and "--delete" in fake.execs[0][0]
    assert order == ["terminate"]


async def _record(order: list[str], name: str) -> None:
    order.append(name)


async def test_disk_sync_open_fails_when_the_restore_fails(
    monkeypatch: pytest.MonkeyPatch, provider: ModalSandboxProvider
) -> None:
    _use(monkeypatch, _FakeModal(restore_code=1, restore_stderr="access denied"))
    with pytest.raises(RuntimeError, match="access denied"):
        await provider.open(
            SandboxSpec(backend="modal", storage="disk_sync"), agent_id="a", thread_id="t1"
        )


async def test_inline_bucket_env_against_a_tunnelled_minio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModal()
    _use(monkeypatch, fake)
    provider = ModalSandboxProvider(
        app_name="app",
        bucket="b",
        endpoint_url="https://abc.trycloudflare.com",
        bucket_env={"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"},
    )
    await provider.open(
        SandboxSpec(backend="modal", storage="disk_sync"), agent_id="a", thread_id="t"
    )
    creds = {"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"}
    assert fake.created["secrets"] == [("dict", creds)]
    assert fake.execs[0][0][2:5] == ("s5cmd", "--endpoint-url", "https://abc.trycloudflare.com")

    await provider.open(SandboxSpec(backend="modal"), agent_id="a", thread_id="t")
    assert fake.created["secrets"] == []
    assert fake.created["volumes"][MOUNT_PATH][2]["secret"] == ("dict", creds)


def test_with_s5cmd_installs_the_pinned_binary() -> None:
    image = SimpleNamespace(run_commands=lambda *cmds: cmds)
    cmds = with_s5cmd(image)
    assert S5CMD_URL.endswith("/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz")
    assert S5CMD_URL in cmds[0]


async def test_local_sync_is_a_no_op(tmp_path: Any) -> None:
    assert (await LocalSandbox(tmp_path).sync()).returncode == 0
