"""A restore's ranged resume: a large object continues from its last byte, never from zero."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from botocore.exceptions import ReadTimeoutError

from actant.sandbox import entry, ranged
from actant.sandbox.protocol import RestoreConfig, StampConfig

URL = "s3://b/t/structform.npz"


class _Body:
    """Hands out ``data`` in ``chunk``-byte reads, then stalls (raises) after ``stall_after``
    bytes, as a read timing out does."""

    def __init__(self, data: bytes, chunk: int, stall_after: int | None) -> None:
        self._data = data
        self._chunk = chunk
        self._stall_after = stall_after
        self._at = 0

    def read(self, amt: int) -> bytes:
        if self._stall_after is not None and self._at >= self._stall_after:
            raise ReadTimeoutError(endpoint_url="https://r2.example")
        piece = self._data[self._at : self._at + min(amt, self._chunk)]
        self._at += len(piece)
        return piece

    def close(self) -> None:
        pass


class _Client:
    """An object store serving ``data``; the first ``stalls`` GETs stall after ``stall_after``
    bytes. Records each requested range."""

    def __init__(self, data: bytes, *, stalls: int = 0, stall_after: int = 0) -> None:
        self.data = data
        self.stalls = stalls
        self.stall_after = stall_after
        self.ranges: list[tuple[int, int]] = []

    def get_object(self, **request: str) -> ranged.GetObjectOutput:
        assert request["Bucket"] == "b" and request["Key"] == "t/structform.npz"
        start, end = (int(n) for n in request["Range"].removeprefix("bytes=").split("-"))
        self.ranges.append((start, end))
        stall = None
        if self.stalls:
            self.stalls -= 1
            stall = self.stall_after
        body = _Body(self.data[start : end + 1], 10, stall)
        return {
            "Body": body,
            "ETag": '"e1"',
            "ContentRange": f"bytes {start}-{end}/{len(self.data)}",
        }


def test_a_stalled_part_resumes_from_its_last_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ranged, "PART_BYTES", 100)
    data = bytes(range(256)) * 2  # 512 bytes, six parts
    s3 = _Client(data, stalls=2, stall_after=40)
    target = tmp_path / "structform.npz"
    fetched = ranged.fetch(s3, URL, len(data), target, deadline=time.monotonic() + 10, tries=3)
    assert target.read_bytes() == data and fetched == len(data)
    assert not ranged.partial_path(target).exists()
    # Each stall resumed where it stopped: 40 bytes in, then 40 more; nothing fetched twice.
    assert s3.ranges[:3] == [(0, 99), (40, 139), (80, 179)]
    assert sum(end - start + 1 for start, end in s3.ranges[2:]) == len(data) - 80


def test_a_part_that_never_moves_fails_and_keeps_its_partial_for_the_next_try(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ranged, "PART_BYTES", 100)
    data = b"x" * 300
    target = tmp_path / "structform.npz"
    stuck = _Client(data, stalls=99, stall_after=0)
    ranged.partial_path(target).write_bytes(data[:120])  # an earlier try got this far
    with pytest.raises(ranged.FetchError, match="3 tries without progress at 120/300"):
        ranged.fetch(stuck, URL, len(data), target, deadline=time.monotonic() + 10, tries=3)
    assert stuck.ranges == [(120, 219)] * 3
    healthy = _Client(data)
    assert ranged.fetch(healthy, URL, 300, target, deadline=time.monotonic() + 10, tries=3) == 180
    assert healthy.ranges[0] == (120, 219) and target.read_bytes() == data


def test_a_retry_resumes_the_large_object_and_refetches_nothing_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first attempt pulls the small files and stalls on the large one; the retry
    fetches the large one by parts, and the rerun pull (like s5cmd's sync) skips all of it."""
    monkeypatch.setattr(entry, "CHECK_INTERVAL_S", 0.1)
    monkeypatch.setattr(entry, "RESUME_BYTES", 500)
    monkeypatch.setattr(ranged, "PART_BYTES", 256)
    data = bytes(range(256)) * 4
    s3 = _Client(data)
    monkeypatch.setattr(ranged, "client", lambda endpoint_url, stall_s: s3)
    runs = tmp_path / "runs"
    pull = (
        "import os, sys, time\n"
        f"os.chdir({str(tmp_path)!r})\n"
        "open('runs', 'a').write('.')\n"
        "for name in ('a.txt', 'b.txt'):\n"
        "    if os.path.exists(name): continue\n"
        "    open(name, 'w').write('abc')\n"
        "    print(f'cp s3://b/t/{name} {name}', flush=True)\n"
        "open('structform.npz4179753230', 'w').write('part')  # s5cmd's temporary file\n"
        "big = 'structform.npz'\n"
        "if not os.path.exists(big) or os.path.getsize(big) < 1024: time.sleep(60)\n"
    )
    when = "2026-01-02T03:04:05Z"
    listed = "\n".join(
        json.dumps({"key": f"s3://b/t/{name}", "last_modified": when, "size": size})
        for name, size in (("a.txt", 3), ("b.txt", 3), ("structform.npz", len(data)))
    )
    config = RestoreConfig(
        argv=[sys.executable, "-c", pull],
        stall_s=1,
        attempts=3,
        stamp=StampConfig(
            argv=[sys.executable, "-c", f"print({listed!r})"],
            prefix="s3://b/t/",
            root=str(tmp_path),
        ),
    )
    summary = entry.restore_summary(config)
    assert summary.ok and runs.read_text() == ".."
    assert (tmp_path / "structform.npz").read_bytes() == data
    assert s3.ranges == [(0, 255), (256, 511), (512, 767), (768, 1023)]  # each byte once
    [pulled] = summary.pulls
    assert (pulled.objects, pulled.resumed, pulled.stalls, pulled.attempts) == (3, 1, 1, 2)
    assert pulled.bytes == len(data) + 6
    assert int((tmp_path / "structform.npz").stat().st_mtime) == 1767323045
    # The killed attempt's temporary file is gone, so no push uploads it.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt", "runs", "structform.npz"]


def test_the_old_attempt_timeout_is_read_as_the_stall_window() -> None:
    from actant.sandbox import SandboxSpec, Storage

    assert SandboxSpec(backend="modal", storage=Storage.DISK_SYNC).restore_stall_s == 60
    with pytest.warns(DeprecationWarning, match="restore_stall_s"):
        spec = SandboxSpec(backend="modal", restore_attempt_timeout_s=240)
    assert spec.restore_stall_s == 240
