"""A container sandbox's entrypoint: restore storage, then serve the toolset (or idle).

::

    python -m actant.sandbox.entry [--restore JSON-ARGV] [-- HOST-ARGS...]

The restore runs to completion before the host binds its port, so a readiness
probe on that port (or on :data:`READY_FILE` without a toolset) only passes once
the files are in place. A failed restore exits non-zero, which ends the sandbox.
An empty bucket prefix (a new thread) is not a failure.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import actant.sandbox.host as host

READY_FILE = "/tmp/actant-ready"
#: What s5cmd prints for a prefix with no objects.
EMPTY_PREFIX = "no object found"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m actant.sandbox.entry")
    parser.add_argument("--restore", help="JSON argv that pulls storage onto the disk")
    parser.add_argument("host_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.restore:
        restore = subprocess.run(json.loads(args.restore), capture_output=True, text=True)
        if restore.returncode != 0 and EMPTY_PREFIX not in restore.stderr:
            print(
                f"restore exited {restore.returncode}: {restore.stderr[-4000:]}", file=sys.stderr
            )
            return 1
    host_args = args.host_args[1:] if args.host_args[:1] == ["--"] else args.host_args
    if host_args:
        return host.main(host_args)
    Path(READY_FILE).touch()
    signal.pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
