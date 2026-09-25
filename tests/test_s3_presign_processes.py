"""Separate worker processes hand out the same presigned URL, with nothing shared.

Each resolution runs in a fresh interpreter that builds its own ``S3AssetResolver`` from the
same static credentials, with its clock set to the time under test. Byte-identical output
across those processes is what lets the provider's prompt cache match, with no stored URL.
"""

from __future__ import annotations

import json
import subprocess
import sys

CHILD = """
import asyncio, json, sys, time

from actant.assets import AssetContext, AssetReference
from actant.storage.s3 import S3AssetResolver
from actant.storage.sigv4 import SigningKeys


class Client:
    def head_object(self, *, Bucket, Key):
        return {}


now = float(sys.argv[1])
resolver = S3AssetResolver(
    Client(),
    endpoint_url="https://account.r2.cloudflarestorage.com",
    region="auto",
    keys=SigningKeys("test-key", "test-secret"),
    bucket="b",
    prefix="images/",
    window_s=3600,
    buffer_s=1800,
    clock=lambda: now,
)


async def main():
    out = {}
    for key in sys.argv[2:]:
        image = await resolver.resolve(AssetReference(key, "image/png"), AssetContext("a", "t", "r", "turn"))
        out[key] = [image.url, image.expires_at]
    print(json.dumps(out))


asyncio.run(main())
"""

KEYS = ["images/a.png", "images/a b+c.png", "images/ümlaut/こんにちは.jpg"]
WINDOW, BUFFER = 3600, 1800
START = 1_800_000_000 - 1_800_000_000 % WINDOW


def resolve_in_new_process(now: float) -> dict[str, tuple[str, float]]:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(now), *KEYS],
        capture_output=True,
        text=True,
        check=True,
    )
    return {key: (url, expires_at) for key, (url, expires_at) in json.loads(done.stdout).items()}


def test_separate_processes_agree_within_a_window_and_move_on_together() -> None:
    times = [START, START + 1.5, START + WINDOW - 0.001]
    within = [resolve_in_new_process(now) for now in times]
    assert within[0] == within[1] == within[2]
    assert len({url for url, _ in within[0].values()}) == len(KEYS)

    after = [
        resolve_in_new_process(START + WINDOW),
        resolve_in_new_process(START + 2 * WINDOW - 1),
    ]
    assert after[0] == after[1]
    assert all(after[0][key][0] != within[0][key][0] for key in KEYS)

    for now, urls in zip([*times, START + WINDOW, START + 2 * WINDOW - 1], [*within, *after]):
        assert all(expires_at - now >= BUFFER for _, expires_at in urls.values())
