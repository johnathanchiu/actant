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

Each pull runs at most ``restore.attempts`` times, all within ``restore.timeout_s``.
An attempt is killed only when it stalls: nothing arrives (no output, no bytes on disk)
for ``restore.stall_s``. A slow pull that keeps moving runs on, however long one object
takes. A retry skips the files already pulled and first resumes each large object still
missing with ranged downloads (:mod:`actant.sandbox.ranged`), from where an earlier retry
stopped. Progress (objects fetched of those listed) goes to stderr every
:data:`PROGRESS_INTERVAL_S`; a :class:`~actant.sandbox.protocol.RestoreSummary` goes to
stderr last and, as JSON, to ``restore.summary_path``, where the provider reads it.
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
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ValidationError

import actant.sandbox.host as host
from actant.sandbox import ranged
from actant.sandbox.protocol import (
    EntryConfig,
    Pulled,
    PullSummary,
    RestoreConfig,
    RestoreSummary,
    StampConfig,
)

READY_FILE = "/tmp/actant-ready"
#: Where the provider asks the entrypoint to write its restore summary.
SUMMARY_FILE = "/tmp/actant-restore.json"
#: How often a running pull reports its progress.
PROGRESS_INTERVAL_S = 10.0
#: How often a running pull is checked for a stall.
CHECK_INTERVAL_S = 2.0
#: A retry resumes a missing object at least this large with ranged downloads instead
#: of starting it over with s5cmd.
RESUME_BYTES = 32 * 1024 * 1024
#: Why an attempt was killed: nothing arrived for ``stall_s``, or the restore's time ran out.
STALLED = "stalled"
TIMED_OUT = "timed out"
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
    etag: str | None = None


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


def restore(config: RestoreConfig) -> bool:
    """Pull the run's prefix onto the disk (a new run's seed instead, while it is copied
    into the run's prefix) and ``config.also`` alongside, then stamp mtimes; whether
    startup may continue."""
    return restore_summary(config).ok


def restore_summary(config: RestoreConfig) -> RestoreSummary:
    """:func:`restore`, with what each pull did; the summary goes to stderr last, so the
    tail of a failed sandbox's stderr has it."""
    started = time.monotonic()
    deadline = started + config.timeout_s
    pulls: list[PullSummary] = []
    with ThreadPoolExecutor(max(1, len(config.also))) as pool:
        others = [
            pool.submit(pull, also.argv, also.stamp, config, pulls, deadline)
            for also in config.also
        ]
        own = _restore_own(config, pulls, deadline)
        pulled = [other.result() for other in others]
    summary = RestoreSummary(
        ok=own and Pulled.FAILED not in pulled,
        seconds=round(time.monotonic() - started, 3),
        pulls=pulls,
    )
    print(summary.line(), file=sys.stderr)
    return summary


def _restore_own(config: RestoreConfig, pulls: list[PullSummary], deadline: float) -> bool:
    seed = config.seed
    if seed is None:
        return pull(config.argv, config.stamp, config, pulls, deadline) != Pulled.FAILED
    with ThreadPoolExecutor(1) as pool:
        checking = pool.submit(_run, seed.check_marker_argv, _left(deadline))
        pulled = pull(config.argv, config.stamp, config, pulls, deadline)
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
        copying = pool.submit(_run, seed.copy_argv, _left(deadline))
        pulled = pull(seed.argv, seed.stamp, config, pulls, deadline)
        copied = copying.result()
    if (failed := _failure(copied)) is not None:
        print(f"seed copy {failed}", file=sys.stderr)
        return False
    # Only a finished copy is marked, so an interrupted one fails the next startup.
    if (failed := _failure(_run(seed.write_marker_argv, _left(deadline)))) is not None:
        print(f"seed marker {failed}", file=sys.stderr)
        return False
    return pulled != Pulled.FAILED


def _left(deadline: float) -> float:
    """Seconds until ``deadline``, never quite zero (a zero timeout is no timeout)."""
    return max(0.001, deadline - time.monotonic())


def pull(
    argv: Sequence[str],
    stamp_config: StampConfig | None,
    config: RestoreConfig,
    pulls: list[PullSummary] | None = None,
    deadline: float | None = None,
) -> Pulled:
    """Run the pull ``argv`` (again after a failed or stalled attempt, up to
    ``config.attempts`` runs, until ``deadline``), with the stamp's listing alongside it
    (into a file: a pipe would stall it), then apply the listing. A retry first resumes
    the large objects still missing. What it did is appended to ``pulls``."""
    started = time.monotonic()
    if deadline is None:
        deadline = started + config.timeout_s
    root = Path(stamp_config.root) if stamp_config is not None else None
    summary = _PullStats(stamp_config.prefix if stamp_config is not None else argv[-1])
    with tempfile.TemporaryFile("w+") as listing:
        lister = None
        if stamp_config is not None:
            try:
                lister = subprocess.Popen(stamp_config.argv, stdout=listing, stderr=listing)
            except OSError as error:
                print(f"mtime stamp skipped: could not start: {error}", file=sys.stderr)
        progress = _Progress(listing if lister is not None else None, config.attempts)
        attempt = 1
        while True:
            summary.attempts = attempt
            done = _pull_once(argv, config, deadline, progress, attempt, root)
            failed = _failure(done)
            if failed is not None and failed.startswith(STALLED):
                summary.stalls += 1
            out_of_time = failed is not None and failed.startswith(TIMED_OUT)
            if failed is None or attempt == config.attempts or out_of_time:
                break
            print(f"restore attempt {attempt}/{config.attempts} {failed}", file=sys.stderr)
            attempt += 1
            if lister is not None and stamp_config is not None and root is not None:
                summary.resumed += _resume_large(lister, listing, stamp_config, config, deadline)
        summary.objects = progress.fetched + summary.resumed
        # s5cmd 2.3's sync exits 0 on an empty prefix, but still says so.
        empty = not isinstance(done, str) and EMPTY_PREFIX in done.stderr
        if failed is not None or empty:
            if lister is not None:
                lister.kill()
                lister.wait()
                if stamp_config is not None and failed is not None:  # what was listed so far
                    objects = _listed(_lines(listing), stamp_config)
                    summary.listed = len(objects)
                    summary.bytes = sum(item.size for item in objects)
            if failed is None:
                outcome = Pulled.EMPTY
            else:
                print(f"restore {failed}", file=sys.stderr)
                outcome = Pulled.FAILED
        else:
            outcome = Pulled.FILES
            if lister is not None and stamp_config is not None:
                if stamp(lister, listing, stamp_config, _left(deadline)):
                    objects = _listed(_lines(listing), stamp_config)
                    summary.listed = len(objects)
                    summary.bytes = sum(item.size for item in objects)
                    if attempt > 1:
                        _discard_leftovers(Path(stamp_config.root), objects, stamp_config.prefix)
    if pulls is not None:
        pulls.append(summary.done(outcome, failed, time.monotonic() - started))
    return outcome


class _PullStats:
    def __init__(self, source: str) -> None:
        self.source = source
        self.attempts = 0
        self.stalls = 0
        self.resumed = 0
        self.objects = 0
        self.listed: int | None = None
        self.bytes: int | None = None

    def done(self, outcome: Pulled, error: str | None, seconds: float) -> PullSummary:
        return PullSummary(
            source=self.source,
            outcome=outcome,
            objects=self.objects,
            listed=self.listed,
            bytes=self.bytes,
            seconds=round(seconds, 3),
            attempts=self.attempts,
            stalls=self.stalls,
            resumed=self.resumed,
            error=error,
        )


def _listed(lines: Iterable[str], config: StampConfig) -> list[ListedObject]:
    """The listing's objects under the stamp's prefix, less its skipped prefixes."""
    objects = []
    for line in lines:
        try:
            item = ListedObject.model_validate_json(line)
        except ValidationError:
            continue
        if item.key.startswith(config.prefix) and not item.key.startswith(tuple(config.skip)):
            objects.append(item)
    return objects


def _resume_large(
    lister: subprocess.Popen[bytes],
    listing: IO[str],
    stamp_config: StampConfig,
    config: RestoreConfig,
    deadline: float,
) -> int:
    """Before a retry: fetch each listed object of at least :data:`RESUME_BYTES` that is
    not yet on disk with ranged downloads, resuming its partial file from an earlier
    retry; how many finished. s5cmd's retry then skips them. Anything not finished here
    is left to that retry."""
    try:
        lister.wait(_left(deadline))
    except subprocess.TimeoutExpired:
        return 0
    if lister.returncode != 0:
        return 0
    root = Path(stamp_config.root)
    large = [
        (item, root / item.key[len(stamp_config.prefix) :])
        for item in _listed(_lines(listing), stamp_config)
        if item.size >= RESUME_BYTES
    ]
    missing = [(item, target) for item, target in large if not _on_disk(target, item.size)]
    if not missing:
        return 0
    try:
        s3 = ranged.client(config.endpoint_url, config.stall_s)
    except ImportError as error:  # no ``sandbox`` extra: s5cmd starts each over instead
        print(f"restore: no ranged resume ({error})", file=sys.stderr)
        return 0
    finished = 0
    for item, target in missing:
        try:
            fetched = ranged.fetch(
                s3,
                item.key,
                item.size,
                target,
                deadline=deadline,
                tries=config.attempts,
                etag=item.etag,
            )
        except ranged.FetchError as error:
            print(f"restore: resume {error}", file=sys.stderr)
            continue
        finished += 1
        print(
            f"restore: resumed {item.key} ({fetched}/{item.size} bytes fetched)", file=sys.stderr
        )
    return finished


def _discard_leftovers(root: Path, objects: Iterable[ListedObject], prefix: str) -> None:
    """Remove what killed attempts left beside the listed objects: s5cmd's temporary
    files (``<name><digits>``, renamed over ``<name>`` only once complete) and unfinished
    ranged partial files. Left alone, a push would upload them into the bucket."""
    names: dict[Path, set[str]] = {}
    for item in objects:
        path = root / item.key[len(prefix) :]
        names.setdefault(path.parent, set()).add(path.name)
    for directory, listed in names.items():
        try:
            present = os.listdir(directory)
        except OSError:
            continue
        for name in present:
            if name in listed:
                continue
            leftover = name.endswith(ranged.PARTIAL_SUFFIX) and (
                name.removesuffix(ranged.PARTIAL_SUFFIX) in listed
            )
            leftover = leftover or any(
                name.startswith(base) and name[len(base) :].isdigit() for base in listed
            )
            if leftover:
                (directory / name).unlink(missing_ok=True)


def _on_disk(path: Path, size: int) -> bool:
    try:
        return path.stat().st_size == size
    except OSError:
        return False


def _pull_once(
    argv: Sequence[str],
    config: RestoreConfig,
    deadline: float,
    progress: _Progress,
    attempt: int,
    root: Path | None,
) -> subprocess.CompletedProcess[str] | str:
    """One pull attempt, killed once nothing arrives for ``config.stall_s`` (no output, no
    bytes on disk under ``root``) or at ``deadline``, reporting progress while it runs."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        try:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        except OSError as error:
            return f"could not start: {error}"
        seen = _arrived(out, root)
        moved = time.monotonic()
        report = moved + PROGRESS_INTERVAL_S
        while True:
            now = time.monotonic()
            wait = min(
                CHECK_INTERVAL_S, report - now, deadline - now, moved + config.stall_s - now
            )
            try:
                process.wait(max(0.0, wait))
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if (arrived := _arrived(out, root)) != seen:
                seen, moved = arrived, now
            cut = None
            if now >= deadline:
                cut = f"{TIMED_OUT} after {config.timeout_s:g}s"
            elif now - moved >= config.stall_s:
                cut = f"{STALLED}: nothing arrived for {config.stall_s:g}s"
            if cut is not None:
                process.kill()
                process.wait()
                progress.report(out, attempt, final=True)
                return cut
            if now >= report:
                progress.report(out, attempt, final=False)
                report = now + PROGRESS_INTERVAL_S
        progress.report(out, attempt, final=True)
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(argv, process.returncode, out.read(), err.read())


def _arrived(output: IO[str], root: Path | None) -> tuple[int, int, int]:
    """What a pull has produced so far: its output's length, and the bytes (by size and by
    blocks, as parts land out of order) of the files under ``root``. Any change is
    progress. ``root`` holds the other pulls' paths too, so their bytes count as well."""
    size = blocks = 0
    if root is not None:
        for directory, _, names in os.walk(root):
            for name in names:
                try:
                    stat = os.lstat(os.path.join(directory, name))
                except OSError:
                    continue
                size += stat.st_size
                blocks += stat.st_blocks
    return os.fstat(output.fileno()).st_size, size, blocks


def _lines(file: IO[str]) -> list[str]:
    """``file``'s lines so far, read without moving the offset a child process writes at."""
    size = os.fstat(file.fileno()).st_size
    return os.pread(file.fileno(), size, 0).decode(errors="replace").splitlines()


class _Progress:
    """Objects a pull fetched (s5cmd prints ``cp <source> <target>`` for each), across its
    attempts, of those the stamp's listing has found so far."""

    def __init__(self, listing: IO[str] | None, attempts: int) -> None:
        self._listing = listing
        self._attempts = attempts
        self.fetched = 0  # by finished attempts

    def report(self, output: IO[str], attempt: int, *, final: bool) -> None:
        fetched = self.fetched + sum(line.startswith("cp ") for line in _lines(output))
        if final:
            self.fetched = fetched
        listed = "?" if self._listing is None else str(len(_lines(self._listing)))
        print(
            f"restore: {fetched}/{listed} objects (attempt {attempt}/{self._attempts})",
            file=sys.stderr,
        )


def stamp(
    lister: subprocess.Popen[bytes], listing: IO[str], config: StampConfig, timeout: float
) -> bool:
    """Wait for the listing (``lister``, writing to ``listing``) and apply it; whether
    there was a listing to apply."""
    try:
        lister.wait(timeout)
    except subprocess.TimeoutExpired:
        lister.kill()
        lister.wait()
        print(f"mtime stamp skipped: timed out after {timeout:g}s", file=sys.stderr)
        return False
    listing.seek(0)
    lines = listing.read().splitlines()
    if lister.returncode != 0:
        output = "\n".join(lines)
        if EMPTY_PREFIX not in output:  # an empty prefix is a new thread: nothing to stamp
            print(f"mtime stamp skipped: {output[-1000:]}", file=sys.stderr)
        return False
    stamp_mtimes(lines, config.prefix, Path(config.root), config.skip)
    return True


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
        restoring = config.restore
        restored = pool.submit(restore_summary, restoring) if restoring is not None else None
        services = host.load_services(config.host) if config.host is not None else None
        if restoring is not None and restored is not None:
            summary = restored.result()
            if restoring.summary_path is not None:
                Path(restoring.summary_path).write_text(summary.model_dump_json())
            if not summary.ok:
                return 1
    if config.host is not None:
        return host.main(config.host, services)
    Path(READY_FILE).touch()
    signal.pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
