"""Presigned image URLs from a real local sandbox host against a real S3 API (MinIO) and s5cmd.

Skipped unless s5cmd is on PATH and ``ACTANT_TEST_S3_ENDPOINT`` names a reachable
MinIO (with ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` for it)::

    ACTANT_TEST_S3_ENDPOINT=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=... \\
        AWS_SECRET_ACCESS_KEY=... uv run pytest tests/test_sandbox_images_minio.py

``ACTANT_TEST_S3_PUBLIC_ENDPOINT`` (a tunnel to that MinIO, e.g. ``cloudflared tunnel
--url``) also presigns against the public host and fetches through it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from actant.sandbox import ImageBucket, LocalSandbox, LocalSandboxProvider, SandboxSpec
from actant.sandbox import SandboxRunner
from actant.sandbox.protocol import InlineSource, UrlSource
from actant.tools import image_block

ENDPOINT = os.environ.get("ACTANT_TEST_S3_ENDPOINT")
PUBLIC = os.environ.get("ACTANT_TEST_S3_PUBLIC_ENDPOINT")
TESTS = str(Path(__file__).parent)

if shutil.which("s5cmd") is None or not ENDPOINT:
    pytest.skip("needs s5cmd and ACTANT_TEST_S3_ENDPOINT", allow_module_level=True)
if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
    pytest.skip("needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY", allow_module_level=True)


def _s5(*args: str) -> subprocess.CompletedProcess[str]:
    assert ENDPOINT
    return subprocess.run(
        ["s5cmd", "--endpoint-url", ENDPOINT, *args], capture_output=True, text=True, check=False
    )


def _get(url: str) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=30) as reply:  # noqa: S310 -- the test's own URL
        return reply.read(), reply.headers["Content-Type"]


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("AWS_REGION", os.environ.get("AWS_REGION", "us-east-1"))
    name = f"actant-images-{uuid.uuid4().hex[:12]}"
    made = _s5("mb", f"s3://{name}")
    if made.returncode != 0:
        pytest.skip(f"MinIO at {ENDPOINT} unavailable: {made.stderr.strip()}")
    yield name
    _s5("rm", f"s3://{name}/*")
    _s5("rb", f"s3://{name}")


async def _open(tmp_path: Path, images: ImageBucket) -> AsyncIterator[LocalSandbox]:
    provider = LocalSandboxProvider(tmp_path, images=images)
    spec = SandboxSpec(
        services={"counter": "service_fixtures:Counter"},
        env={"PYTHONPATH": TESTS},
        image_url_ttl_s=600,
    )
    sandbox = await provider.open(spec, agent_id="a", thread_id="t1")
    assert isinstance(sandbox, LocalSandbox)
    try:
        yield sandbox
    finally:
        await sandbox.close()


@pytest.fixture
async def sandbox(tmp_path: Path, bucket: str) -> AsyncIterator[LocalSandbox]:
    async for opened in _open(tmp_path, ImageBucket(bucket, endpoint_url=ENDPOINT)):
        yield opened


async def test_returned_images_arrive_as_urls_a_plain_get_fetches(
    sandbox: LocalSandbox, bucket: str
) -> None:
    runner = SandboxRunner("counter")
    for as_file in (False, True):
        response = await runner.call(
            "picture", {"size": 64, "as_file": as_file}, key="t1", sandbox=sandbox
        )
        assert response.storage is not None and response.storage.image_error is None
        [image] = response.images
        assert isinstance(image.source, UrlSource)
        data, content_type = _get(image.source.url)
        assert data.startswith(b"\x89PNG") and len(data) == 64 + 8
        assert content_type == "image/png"
        assert image_block(image)["source"] == {"type": "url", "url": image.source.url}
    # Beside the thread's prefix, never inside it: a push never mirrors or deletes images.
    listed = _s5("ls", f"s3://{bucket}/sandboxes/*").stdout
    assert "t1.actant-images/" in listed and "t1/" not in listed.replace("t1.actant-images/", "")


async def test_an_unreachable_bucket_sends_bytes_and_says_why(tmp_path: Path, bucket: str) -> None:
    missing = ImageBucket(f"{bucket}-missing", endpoint_url=ENDPOINT)
    async for sandbox in _open(tmp_path, missing):
        response = await SandboxRunner("counter").call(
            "picture", {"size": 16}, key="t1", sandbox=sandbox
        )
        [image] = response.images
        assert isinstance(image.source, InlineSource)
        assert response.storage is not None and "upload exited" in str(
            response.storage.image_error
        )


@pytest.mark.skipif(not PUBLIC, reason="needs ACTANT_TEST_S3_PUBLIC_ENDPOINT")
async def test_urls_presigned_for_a_public_endpoint_fetch_through_it(
    tmp_path: Path, bucket: str
) -> None:
    images = ImageBucket(bucket, endpoint_url=ENDPOINT, public_endpoint_url=PUBLIC)
    async for sandbox in _open(tmp_path, images):
        response = await SandboxRunner("counter").call(
            "picture", {"size": 32}, key="t1", sandbox=sandbox
        )
        [image] = response.images
        assert isinstance(image.source, UrlSource) and PUBLIC
        assert image.source.url.startswith(PUBLIC.rstrip("/") + "/")
        data, _ = _get(image.source.url)
        assert data.startswith(b"\x89PNG") and len(data) == 32 + 8
