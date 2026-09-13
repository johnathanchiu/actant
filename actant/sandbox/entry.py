"""A container sandbox's entrypoint: restore storage, then serve the toolset (or idle).

::

    python -m actant.sandbox.entry [--restore JSON-ARGV] [--restore-timeout S]
        [--stamp JSON] [-- HOST-ARGS...]

The restore runs to completion before the host binds its port, so a readiness
probe on that port (or on :data:`READY_FILE` without a toolset) only passes once
the files are in place. A failed or timed-out restore exits non-zero, which ends
the sandbox. An empty bucket prefix (a new thread) is not a failure.

``--stamp {"argv", "prefix", "root"}`` then lists the prefix (``s5cmd --json ls``
lines) and sets each restored file's mtime to its object's ``last_modified``.
Downloads get the current time, and s5cmd's sync re-uploads any file newer than
its object, so without this every push after a restore re-uploads the whole
workspace. A stamp failure only costs that re-upload, so it is logged, not fatal.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import actant.sandbox.host as host

READY_FILE = "/tmp/actant-ready"
#: What s5cmd prints for a prefix with no objects.
EMPTY_PREFIX = "no object found"
RESTORE_TIMEOUT_S = 1800.0


def stamp_mtimes(listing: Iterable[str], prefix: str, root: Path) -> int:
    """Set each local file's mtime to its object's ``last_modified`` when the sizes match;
    how many were stamped. ``listing`` is ``s5cmd --json ls`` output, one object per line."""
    stamped = 0
    for line in listing:
        try:
            item = json.loads(line)
            key = str(item["key"])
            when = datetime.fromisoformat(item["last_modified"])
            # Integer nanoseconds, truncated: a float can round a stamp past its object's
            # time, and s5cmd uploads any file even a nanosecond newer.
            modified = calendar.timegm(when.utctimetuple()) * 10**9 + when.microsecond * 1000
        except (ValueError, KeyError, TypeError):
            continue
        if not key.startswith(prefix):
            continue
        path = root / key[len(prefix) :]
        try:
            if path.stat().st_size == int(item.get("size") or 0):
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m actant.sandbox.entry")
    parser.add_argument("--restore", help="JSON argv that pulls storage onto the disk")
    parser.add_argument("--restore-timeout", type=float, default=RESTORE_TIMEOUT_S)
    parser.add_argument("--stamp", help='JSON {"argv", "prefix", "root"}: restored mtimes')
    parser.add_argument("host_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.restore:
        restore = _run(json.loads(args.restore), args.restore_timeout)
        if isinstance(restore, str):
            print(f"restore {restore}", file=sys.stderr)
            return 1
        if restore.returncode != 0 and EMPTY_PREFIX not in restore.stderr:
            print(
                f"restore exited {restore.returncode}: {restore.stderr[-4000:]}", file=sys.stderr
            )
            return 1
    if args.stamp:
        stamp = json.loads(args.stamp)
        listed = _run(stamp["argv"], args.restore_timeout)
        if not isinstance(listed, str) and EMPTY_PREFIX in listed.stderr:
            pass  # a new thread: nothing restored, nothing to stamp
        elif isinstance(listed, str) or listed.returncode != 0:
            detail = listed if isinstance(listed, str) else listed.stderr[-1000:]
            print(f"mtime stamp skipped: {detail}", file=sys.stderr)
        else:
            stamp_mtimes(listed.stdout.splitlines(), stamp["prefix"], Path(stamp["root"]))
    host_args = args.host_args[1:] if args.host_args[:1] == ["--"] else args.host_args
    if host_args:
        return host.main(host_args)
    Path(READY_FILE).touch()
    signal.pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
