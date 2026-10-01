"""A container sandbox's entrypoint: restore storage, then serve the services (or idle).

::

    python -m actant.sandbox.entry '<EntryConfig JSON>'

One :class:`~actant.sandbox.protocol.EntryConfig` document, validated before
anything runs. The restore runs to completion before the host binds its port, so
a readiness probe on that port (or on :data:`READY_FILE` without a host) only
passes once the files are in place. The services' modules are imported while the
restore runs, so importing one must not read the restored files. A failed or
timed-out restore exits non-zero, which ends the sandbox. An empty bucket prefix
(a new run) is not a failure.

``restore.also``: other inputs (a read-only capture from another prefix) pulled
alongside the run's own, each an independent pull that may be empty.

``restore.seed``: when the run's prefix is empty, the seed is pulled onto the disk
while ``copy_argv`` copies it into the run's prefix; startup waits for both, so
the host's first push never races the copy. Once the copy finishes, a marker
object (outside the run's prefix, so never pulled or pushed) records it. A run
with files ignores its seed, but without the marker its copy was cut off, and
startup fails rather than serve a partial workspace.

``stamp`` lists the prefix a pull reads (``s5cmd --json ls`` lines, alongside the
pull) and then sets each pulled file's mtime to its object's ``last_modified``.
Downloads get the current time, and s5cmd's sync uploads any file newer than its
object, so without this every push after a restore re-uploads the whole
workspace. A seeded file takes the seed object's time, never later than its copy
in the run's prefix, so a push skips it too. A stamp failure only costs that
re-upload, so it is logged, not fatal.

Each pull runs once, bounded by ``restore.timeout_s``. Blips are the transfer
client's to handle: s5cmd retries a failed request (one part of an object, a ranged
read) without dropping the parts it already has, so a slow but moving object
finishes. Progress (objects fetched of those listed) goes to stderr every
:data:`PROGRESS_INTERVAL_S`, so a stall shows in the sandbox's logs.
"""

from __future__ import annotations

import calendar
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ValidationError

import actant.sandbox.host as host
from actant.sandbox.protocol import EntryConfig, RestoreConfig, StampConfig

READY_FILE = "/tmp/actant-ready"
#: How often a running pull reports its progress.
PROGRESS_INTERVAL_S = 10.0
#: What s5cmd prints for a prefix with no objects.
EMPTY_PREFIX = "no object found"
#: Why startup fails on a run's prefix with files and a seed but no seed marker.
INCOMPLETE_SEED = (
    "the run's prefix has files but no seed marker: its seed copy never finished; "
    "delete the prefix to seed it again"
)


class ListedObject(BaseModel):
    """One line of ``s5cmd --json ls``; ``size`` is omitted for an empty object."""

    key: str
    last_modified: datetime
    size: int = 0


def stamp_mtimes(listing: Iterable[str], prefix: str, root: Path, skip: Sequence[str] = ()) -> int:
    """Set each local file's mtime to its object's ``last_modified`` when the sizes match;
    how many were stamped. ``listing`` is ``s5cmd --json ls`` output, one object per line.
    Keys under a ``skip`` prefix (a read-only mount) are left alone."""
    stamped = 0
    for line in listing:
        try:
            item = ListedObject.model_validate_json(line)
        except ValidationError:
            continue
        if not item.key.startswith(prefix) or item.key.startswith(tuple(skip)):
            continue
        when = item.last_modified  # parsing floors to the microsecond
        # Integer nanoseconds: a float can round a stamp past its object's time, and
        # s5cmd uploads any file even a nanosecond newer.
        modified = calendar.timegm(when.utctimetuple()) * 10**9 + when.microsecond * 1000
        path = root / item.key[len(prefix) :]
        try:
            if path.stat().st_size == item.size:
                os.utime(path, ns=(modified, modified))
                stamped += 1
        except OSError:
            continue
    return stamped


def _run(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str] | str:
    """The finished command, or why it did not finish."""
    try:
        return subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout:g}s"
    except OSError as error:
        return f"could not start: {error}"


def _failure(done: subprocess.CompletedProcess[str] | str) -> str | None:
    """Why an s5cmd command failed; ``None`` on success or an empty prefix."""
    if isinstance(done, str):
        return done
    if done.returncode != 0 and EMPTY_PREFIX not in done.stderr:
        return f"exited {done.returncode}: {done.stderr[-4000:]}"
    return None


class Pulled(StrEnum):
    """How a pull ended."""

    FILES = "files"
    #: The prefix has no objects: a new run.
    EMPTY = "empty"
    FAILED = "failed"


def restore(config: RestoreConfig) -> bool:
    """Pull the run's prefix onto the disk (a new run's seed instead, while it is copied
    into the run's prefix) and ``config.also`` alongside, then stamp mtimes; whether
    startup may continue."""
    with ThreadPoolExecutor(max(1, len(config.also))) as pool:
        others = [pool.submit(pull, also.argv, also.stamp, config) for also in config.also]
        own = _restore_own(config)
        pulled = [other.result() for other in others]
    return own and Pulled.FAILED not in pulled


def _restore_own(config: RestoreConfig) -> bool:
    seed = config.seed
    if seed is None:
        return pull(config.argv, config.stamp, config) != Pulled.FAILED
    with ThreadPoolExecutor(1) as pool:
        checking = pool.submit(_run, seed.check_marker_argv, config.timeout_s)
        pulled = pull(config.argv, config.stamp, config)
        checked = checking.result()
    if pulled == Pulled.FAILED:
        return False
    if pulled == Pulled.FILES:
        if not isinstance(checked, str) and EMPTY_PREFIX in checked.stderr:
            print(f"restore: {INCOMPLETE_SEED}", file=sys.stderr)
            return False
        if (failed := _failure(checked)) is not None:
            print(f"seed marker check {failed}", file=sys.stderr)
            return False
        return True
    # Both finish before the host starts, so no push can race the copy.
    with ThreadPoolExecutor(1) as pool:
        copying = pool.submit(_run, seed.copy_argv, config.timeout_s)
        pulled = pull(seed.argv, seed.stamp, config)
        copied = copying.result()
    if (failed := _failure(copied)) is not None:
        print(f"seed copy {failed}", file=sys.stderr)
        return False
    # Only a finished copy is marked, so an interrupted one fails the next startup.
    if (failed := _failure(_run(seed.write_marker_argv, config.timeout_s))) is not None:
        print(f"seed marker {failed}", file=sys.stderr)
        return False
    return pulled != Pulled.FAILED


def pull(argv: Sequence[str], stamp_config: StampConfig | None, config: RestoreConfig) -> Pulled:
    """Run the pull ``argv``, bounded by ``config.timeout_s``, with the stamp's listing
    alongside it (into a file: a pipe would stall it), then apply the listing."""
    with tempfile.TemporaryFile("w+") as listing:
        lister = None
        if stamp_config is not None:
            try:
                lister = subprocess.Popen(stamp_config.argv, stdout=listing, stderr=listing)
            except OSError as error:
                print(f"mtime stamp skipped: could not start: {error}", file=sys.stderr)
        progress = _Progress(listing if lister is not None else None)
        done = _pull_once(argv, config.timeout_s, progress)
        failed = _failure(done)
        # s5cmd 2.3's sync exits 0 on an empty prefix, but still says so.
        empty = not isinstance(done, str) and EMPTY_PREFIX in done.stderr
        if failed is not None or empty:
            if lister is not None:
                lister.kill()
                lister.wait()
            if failed is None:
                return Pulled.EMPTY
            print(f"restore {failed}", file=sys.stderr)
            return Pulled.FAILED
        if lister is not None and stamp_config is not None:
            stamp(lister, listing, stamp_config, config.timeout_s)
    return Pulled.FILES


def _pull_once(
    argv: Sequence[str], timeout: float, progress: _Progress
) -> subprocess.CompletedProcess[str] | str:
    """Run the pull, killed after ``timeout``, reporting progress while it runs."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        try:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        except OSError as error:
            return f"could not start: {error}"
        deadline = time.monotonic() + timeout
        while True:
            try:
                process.wait(max(0.0, min(PROGRESS_INTERVAL_S, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    progress.report(out)
                    return f"timed out after {timeout:g}s"
                progress.report(out)
        progress.report(out)
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(argv, process.returncode, out.read(), err.read())


def _lines(file: IO[str]) -> list[str]:
    """``file``'s lines so far, read without moving the offset a child process writes at."""
    size = os.fstat(file.fileno()).st_size
    return os.pread(file.fileno(), size, 0).decode(errors="replace").splitlines()


class _Progress:
    """Objects a pull fetched (s5cmd prints ``cp <source> <target>`` for each) of those
    the stamp's listing has found so far."""

    def __init__(self, listing: IO[str] | None) -> None:
        self._listing = listing

    def report(self, output: IO[str]) -> None:
        fetched = sum(line.startswith("cp ") for line in _lines(output))
        listed = "?" if self._listing is None else str(len(_lines(self._listing)))
        print(f"restore: {fetched}/{listed} objects", file=sys.stderr)


def stamp(
    lister: subprocess.Popen[bytes], listing: IO[str], config: StampConfig, timeout: float
) -> None:
    """Wait for the listing (``lister``, writing to ``listing``) and apply it."""
    try:
        lister.wait(timeout)
    except subprocess.TimeoutExpired:
        lister.kill()
        lister.wait()
        print(f"mtime stamp skipped: timed out after {timeout:g}s", file=sys.stderr)
        return
    listing.seek(0)
    lines = listing.read().splitlines()
    if lister.returncode != 0:
        output = "\n".join(lines)
        if EMPTY_PREFIX not in output:  # an empty prefix is a new thread: nothing to stamp
            print(f"mtime stamp skipped: {output[-1000:]}", file=sys.stderr)
        return
    stamp_mtimes(lines, config.prefix, Path(config.root), config.skip)


def main(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        print("usage: python -m actant.sandbox.entry '<EntryConfig JSON>'", file=sys.stderr)
        return 2
    try:
        config = EntryConfig.model_validate_json(args[0])
    except ValidationError as error:
        print(f"invalid entry config: {error}", file=sys.stderr)
        return 2
    with ThreadPoolExecutor(1) as pool:  # the pull there, the services' imports here
        restored = pool.submit(restore, config.restore) if config.restore is not None else None
        services = host.load_services(config.host) if config.host is not None else None
        if restored is not None and not restored.result():
            return 1
    if config.host is not None:
        return host.main(config.host, services)
    Path(READY_FILE).touch()
    signal.pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
