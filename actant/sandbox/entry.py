"""A container sandbox's entrypoint: restore storage, then serve the services (or idle).

::

    python -m actant.sandbox.entry '<EntryConfig JSON>'

One :class:`~actant.sandbox.protocol.EntryConfig` document, validated before
anything runs. The restore runs to completion before the host binds its port, so
a readiness probe on that port (or on :data:`READY_FILE` without a host) only
passes once the files are in place. A failed or timed-out restore exits non-zero,
which ends the sandbox. An empty bucket prefix (a new thread) is not a failure.

``restore.stamp`` then lists the prefix (``s5cmd --json ls`` lines) and sets each
restored file's mtime to its object's ``last_modified``. Downloads get the current
time, and s5cmd's sync uploads any file newer than its object, so without this
every push after a restore re-uploads the whole workspace. A stamp failure only
costs that re-upload, so it is logged, not fatal.
"""

from __future__ import annotations

import calendar
import os
import signal
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

import actant.sandbox.host as host
from actant.sandbox.protocol import EntryConfig, RestoreConfig, StampConfig

READY_FILE = "/tmp/actant-ready"
#: What s5cmd prints for a prefix with no objects.
EMPTY_PREFIX = "no object found"


class ListedObject(BaseModel):
    """One line of ``s5cmd --json ls``; ``size`` is omitted for an empty object."""

    key: str
    last_modified: datetime
    size: int = 0


def stamp_mtimes(listing: Iterable[str], prefix: str, root: Path) -> int:
    """Set each local file's mtime to its object's ``last_modified`` when the sizes match;
    how many were stamped. ``listing`` is ``s5cmd --json ls`` output, one object per line."""
    stamped = 0
    for line in listing:
        try:
            item = ListedObject.model_validate_json(line)
        except ValidationError:
            continue
        if not item.key.startswith(prefix):
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
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout:g}s"
    except OSError as error:
        return f"could not start: {error}"


def restore(config: RestoreConfig) -> bool:
    """Pull storage onto the disk, then stamp mtimes; whether startup may continue."""
    done = _run(config.argv, config.timeout_s)
    if isinstance(done, str):
        print(f"restore {done}", file=sys.stderr)
        return False
    if done.returncode != 0 and EMPTY_PREFIX not in done.stderr:
        print(f"restore exited {done.returncode}: {done.stderr[-4000:]}", file=sys.stderr)
        return False
    if config.stamp is not None:
        stamp(config.stamp, config.timeout_s)
    return True


def stamp(config: StampConfig, timeout: float) -> None:
    listed = _run(config.argv, timeout)
    if not isinstance(listed, str) and EMPTY_PREFIX in listed.stderr:
        return  # a new thread: nothing restored, nothing to stamp
    if isinstance(listed, str) or listed.returncode != 0:
        detail = listed if isinstance(listed, str) else listed.stderr[-1000:]
        print(f"mtime stamp skipped: {detail}", file=sys.stderr)
        return
    stamp_mtimes(listed.stdout.splitlines(), config.prefix, Path(config.root))


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
    if config.restore is not None and not restore(config.restore):
        return 1
    if config.host is not None:
        return host.main(config.host)
    Path(READY_FILE).touch()
    signal.pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
