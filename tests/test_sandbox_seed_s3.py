"""A seeded ``disk_sync`` restore against a real S3 API (moto) with the real s5cmd.

Skipped unless s5cmd is on PATH and moto's server is importable
(``uv run --with 'moto[server]' pytest tests/test_sandbox_seed_s3.py``).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from actant.sandbox import SandboxSpec, Storage
from actant.sandbox.entry import INCOMPLETE_SEED, restore
from actant.sandbox.modal import DISK_PATH, Location, ModalSandboxProvider, Mount, Restore
from actant.sandbox.protocol import EntryConfig, RestoreConfig

if shutil.which("s5cmd") is None:
    pytest.skip("needs s5cmd", allow_module_level=True)
moto_server = pytest.importorskip("moto.server")

SEED = {"a.txt": "a" * 50, "d/b.txt": "b" * 70, "d/e/c.bin": "c" * 90}


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    for name, value in {
        "AWS_ACCESS_KEY_ID": "x", "AWS_SECRET_ACCESS_KEY": "x", "AWS_REGION": "us-east-1"
    }.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


def _s5(endpoint: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["s5cmd", "--endpoint-url", endpoint, *args], capture_output=True, text=True, check=False
    )


def _keys(endpoint: str, pattern: str) -> list[str]:
    return sorted(line.split()[-1] for line in _s5(endpoint, "ls", pattern).stdout.splitlines())


def _restore(provider: ModalSandboxProvider, disk: Path, **overrides: object) -> RestoreConfig:
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC, seed="seed/")
    raw = provider.entry_config(spec, "t1").model_dump_json().replace(DISK_PATH, str(disk))
    config = EntryConfig.model_validate_json(raw).restore
    assert config is not None and config.seed is not None
    return config.model_copy(update={"seed": config.seed.model_copy(update=overrides)})


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


@pytest.fixture
def provider(endpoint: str, tmp_path: Path) -> ModalSandboxProvider:
    bucket = f"seed-{uuid.uuid4().hex[:12]}"  # moto's state outlives each server
    assert _s5(endpoint, "mb", f"s3://{bucket}").returncode == 0
    for name, text in SEED.items():
        (tmp_path / "seed" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "seed" / name).write_text(text)
    assert _s5(endpoint, "cp", f"{tmp_path}/seed/*", f"s3://{bucket}/seed/").returncode == 0
    return ModalSandboxProvider(app_name="x", bucket=bucket, endpoint_url=endpoint)


def test_a_finished_seed_copy_is_marked_and_reopens_as_an_existing_run(
    provider: ModalSandboxProvider, endpoint: str, tmp_path: Path
) -> None:
    bucket = provider.bucket
    first = tmp_path / "first"
    assert restore(_restore(provider, first))
    assert _files(first) == sorted(SEED)
    assert _keys(endpoint, f"s3://{bucket}/sandboxes/t1/*") == sorted(SEED)
    assert _s5(endpoint, "ls", provider.seed_marker("t1")).returncode == 0
    # Stamped from the seed: the first push uploads nothing, and never touches the marker.
    push = [arg.replace(DISK_PATH, str(first)) for arg in provider.sync_argv("t1")]
    (first / "a.txt").unlink()
    pushed = subprocess.run(push, capture_output=True, text=True, check=True).stdout
    assert pushed.split() == ["rm", f"s3://{bucket}/sandboxes/t1/a.txt"]
    assert _s5(endpoint, "ls", provider.seed_marker("t1")).returncode == 0

    # Reopened: the run's own files, not the seed's (a.txt stays deleted).
    assert _s5(endpoint, "cp", f"{first}/d/b.txt", f"s3://{bucket}/seed/new.txt").returncode == 0
    second = tmp_path / "second"
    assert restore(_restore(provider, second))
    assert _files(second) == ["d/b.txt", "d/e/c.bin"]


def test_an_interrupted_seed_copy_fails_every_later_startup(
    provider: ModalSandboxProvider,
    endpoint: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket = provider.bucket
    # The copy dies after one object, the way a killed container leaves it.
    one = f"s3://{bucket}/seed/a.txt"
    cut = [
        "sh",
        "-c",
        f"s5cmd --endpoint-url {endpoint} cp {one} s3://{bucket}/sandboxes/t1/ && exit 1",
    ]
    assert not restore(_restore(provider, tmp_path / "first", copy_argv=cut))
    assert _keys(endpoint, f"s3://{bucket}/sandboxes/t1/*") == ["a.txt"]
    capsys.readouterr()

    assert not restore(_restore(provider, tmp_path / "second"))
    assert INCOMPLETE_SEED in capsys.readouterr().err
    # Not copied again over the partial run.
    assert _keys(endpoint, f"s3://{bucket}/sandboxes/t1/*") == ["a.txt"]


def test_a_two_entry_plan_restores_both_and_pushes_only_its_own(
    endpoint: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bucket = f"plan-{uuid.uuid4().hex[:12]}"
    assert _s5(endpoint, "mb", f"s3://{bucket}").returncode == 0
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "scan.json").write_text("{}")
    (tmp_path / "src" / "room.py").write_text("room")
    assert (
        _s5(endpoint, "cp", f"{tmp_path}/src/scan.json", f"s3://{bucket}/captures/c1/").returncode
        == 0
    )
    assert (
        _s5(endpoint, "cp", f"{tmp_path}/src/room.py", f"s3://{bucket}/sandboxes/t1/").returncode
        == 0
    )

    class Scene(ModalSandboxProvider):
        def restore_plan(self, thread_id: str) -> list[Restore]:
            return [
                Restore(Location(bucket, "captures/c1/"), "capture", push=False),
                Restore(self.thread_location(thread_id), "", push=True),
            ]

    provider = Scene(app_name="x", bucket=bucket, endpoint_url=endpoint)
    disk = tmp_path / "disk"
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC)
    # Without the leading "/", so s5cmd's exclude patterns move to the test's disk too.
    local = DISK_PATH[1:], str(disk)[1:]
    raw = provider.entry_config(spec, "t1").model_dump_json().replace(*local)
    config = EntryConfig.model_validate_json(raw).restore
    assert config is not None and restore(config)
    assert _files(disk) == ["capture/scan.json", "room.py"]
    assert "restore: 1/1 objects (attempt 1/3)" in capsys.readouterr().err

    (disk / "new.txt").write_text("new")
    (disk / "capture" / "mask.png").write_text("mask")
    push = [arg.replace(*local) for arg in provider.sync_argv("t1")]
    subprocess.run(push, capture_output=True, text=True, check=True)
    assert _keys(endpoint, f"s3://{bucket}/sandboxes/t1/*") == ["new.txt", "room.py"]
    assert _keys(endpoint, f"s3://{bucket}/captures/c1/*") == ["scan.json"]


def test_a_mount_is_neither_restored_nor_pushed(endpoint: str, tmp_path: Path) -> None:
    """A scene whose prefix still holds an old copy of its capture: with the capture
    mounted at ``capture``, the restore leaves that path to the mount and a push never
    uploads what the mount shows."""
    bucket = f"mount-{uuid.uuid4().hex[:12]}"
    assert _s5(endpoint, "mb", f"s3://{bucket}").returncode == 0
    (tmp_path / "src" / "capture").mkdir(parents=True)
    (tmp_path / "src" / "room.py").write_text("room")
    (tmp_path / "src" / "capture" / "f.jpg").write_text("old")
    assert _s5(endpoint, "cp", f"{tmp_path}/src/*", f"s3://{bucket}/sandboxes/t1/").returncode == 0

    class Scene(ModalSandboxProvider):
        def bucket_mounts(self, thread_id: str) -> list[Mount]:
            return [Mount(Location(bucket, "captures/c1/"), "capture")]

    provider = Scene(app_name="x", bucket=bucket, endpoint_url=endpoint)
    disk = tmp_path / "disk"
    spec = SandboxSpec(backend="modal", storage=Storage.DISK_SYNC)
    local = DISK_PATH[1:], str(disk)[1:]
    raw = provider.entry_config(spec, "t1").model_dump_json().replace(*local)
    config = EntryConfig.model_validate_json(raw).restore
    assert config is not None and restore(config)
    assert _files(disk) == ["room.py"]

    # What the mount would show, plus the scene's own new file.
    (disk / "capture").mkdir()
    (disk / "capture" / "g.jpg").write_text("mounted")
    (disk / "new.txt").write_text("new")
    push = [arg.replace(*local) for arg in provider.sync_argv("t1")]
    subprocess.run(push, capture_output=True, text=True, check=True)
    assert _keys(endpoint, f"s3://{bucket}/sandboxes/t1/*") == [
        "capture/f.jpg", "new.txt", "room.py"
    ]  # fmt: skip
