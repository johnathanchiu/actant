"""Images a service returns: storage keys from a host that uploads, bytes otherwise."""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from actant.sandbox import ImageBucket, SandboxSpec, host
from actant.sandbox.protocol import (
    CallRequest,
    CallResponse,
    Image,
    ImageUploadConfig,
    InlineSource,
    AssetSource,
)
from actant.blocks import AssetBlock, Base64Source, InlineImageBlock
from actant.tools import image_block
from actant.tools.base import MetadataKey
from actant.tools.service import to_tool_result
from service_fixtures import Counter

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 32

# Records its argv and stdin; ``FAKE_S5CMD_FAIL`` names the subcommand that fails.
FAKE_S5CMD = f"""#!{sys.executable}
import json, os, sys
args = sys.argv[1:]
log = os.environ["FAKE_S5CMD_LOG"]
command = next(a for a in args if a in ("pipe", "presign"))
import time
time.sleep(float(os.environ.get("FAKE_S5CMD_SLEEP_" + command.upper(), "0")))
data = sys.stdin.buffer.read() if command == "pipe" else b""
with open(log, "a") as out:
    out.write(json.dumps({{"argv": args, "stdin": len(data)}}) + "\\n")
if os.environ.get("FAKE_S5CMD_FAIL") == command:
    sys.stderr.write("denied\\n")
    sys.exit(1)
if command == "presign":
    endpoint = args[1] if args[0] == "--endpoint-url" else "https://s3.amazonaws.com"
    print(os.environ.get("FAKE_S5CMD_URL") or endpoint + "/" + args[-1][5:] + "?X-Amz-Signature=s")
"""

CONFIG = ImageUploadConfig(
    destination="s3://b/actant-images/t1/",
    endpoint_url="http://minio:9000",
)


def _inline(name: str = "image-0", data: bytes = PNG) -> Image:
    source = InlineSource(data_b64=base64.b64encode(data).decode())
    return Image(name=name, media_type="image/png", source=source)


@pytest.fixture
def s5cmd_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake = tmp_path / "bin" / "s5cmd"
    fake.parent.mkdir()
    fake.write_text(FAKE_S5CMD)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    log = tmp_path / "s5cmd.log"
    monkeypatch.setenv("FAKE_S5CMD_LOG", str(log))
    return log


def test_inline_and_asset_sources_round_trip() -> None:
    inline = _inline()
    data = base64.b64encode(PNG).decode()
    assert image_block(inline) == InlineImageBlock(
        source=Base64Source(media_type="image/png", data=data)
    )
    reference = inline.model_copy(update={"source": AssetSource(storage_key="s3://b/x.png")})
    assert image_block(reference) == AssetBlock(storage_key="s3://b/x.png", mime="image/png")
    response = CallResponse(images=[inline, reference])
    assert CallResponse.model_validate_json(response.to_json()) == response


async def test_upload_is_content_addressed_and_never_presigns(s5cmd_log: Path) -> None:
    uploaded, error = await host.upload_images(CallResponse(text="t", images=[_inline()]), CONFIG)
    assert error is None and uploaded.text == "t"
    [image] = uploaded.images
    assert isinstance(image.source, AssetSource)
    assert image.source.storage_key.startswith(CONFIG.destination)
    [command] = [json.loads(line) for line in s5cmd_log.read_text().splitlines()]
    assert command["argv"][:3] == ["--endpoint-url", "http://minio:9000", "pipe"]
    assert command["argv"][-1] == image.source.storage_key
    again, _ = await host.upload_images(CallResponse(images=[_inline()]), CONFIG)
    assert again.images[0].source == image.source


async def test_failed_upload_preserves_bytes_and_reports_reason(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_S5CMD_FAIL", "pipe")
    response = CallResponse(images=[_inline("a.png")])
    kept, error = await host.upload_images(response, CONFIG)
    assert kept == response
    assert error and error.startswith("a.png: upload exited 1") and "denied" in error


async def test_missing_upload_binary_preserves_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    kept, error = await host.upload_images(CallResponse(images=[_inline()]), CONFIG)
    assert isinstance(kept.images[0].source, InlineSource)
    assert error and "could not start" in error


async def test_host_tool_result_uses_durable_reference_and_reports_upload_failure(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploading = host.Host({"counter": Counter}, images=CONFIG)
    request = CallRequest(service="counter", key="k", method="picture", args={"size": 4})
    _, response = await uploading.call(request)
    source = response.images[0].source
    assert isinstance(source, AssetSource)
    result = to_tool_result(response)
    assert result.content_blocks and result.content_blocks[-1] == AssetBlock(
        storage_key=source.storage_key, mime="image/png"
    )
    monkeypatch.setenv("FAKE_S5CMD_FAIL", "pipe")
    _, fallback = await uploading.call(request)
    assert isinstance(fallback.images[0].source, InlineSource)
    assert "upload exited 1" in str(to_tool_result(fallback).metadata[MetadataKey.STORAGE])
    _, plain = await uploading.call(
        CallRequest(service="counter", key="k", method="length", args={"text": "ab"})
    )
    assert plain.storage is not None and plain.storage.image_error is None


def test_upload_configuration_has_no_signing_or_retention_policy() -> None:
    bucket = ImageBucket("b")
    assert host.image_upload_config(bucket, SandboxSpec(upload_images=False), "t1") is None
    assert host.image_upload_config(bucket, SandboxSpec(), "t1") == ImageUploadConfig(
        destination="s3://b/actant-images/t1/", timeout_s=10
    )
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="image_upload_timeout_s"):
            SandboxSpec(image_upload_timeout_s=timeout)


async def test_after_failure_unstarted_images_stay_inline(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host, "UPLOAD_CONCURRENCY", 1)
    monkeypatch.setenv("FAKE_S5CMD_FAIL", "pipe")
    images = [_inline(f"i{n}.png", PNG + bytes([n])) for n in range(3)]
    kept, error = await host.upload_images(CallResponse(images=images), CONFIG)
    assert kept.images == images and error and error.startswith("i0.png: upload")
    assert len(s5cmd_log.read_text().splitlines()) == 1


async def test_default_upload_endpoint_uses_aws(s5cmd_log: Path) -> None:
    uploaded, error = await host.upload_images(
        CallResponse(images=[_inline()]), ImageUploadConfig(destination="s3://b/p/")
    )
    assert error is None and isinstance(uploaded.images[0].source, AssetSource)
    [command] = [json.loads(line)["argv"] for line in s5cmd_log.read_text().splitlines()]
    assert command[0] == "pipe"


async def test_hung_upload_is_bounded_and_preserves_bytes(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_S5CMD_SLEEP_PIPE", "30")
    started = time.monotonic()
    kept, error = await host.upload_images(
        CallResponse(images=[_inline()]), CONFIG.model_copy(update={"timeout_s": 0.5})
    )
    assert time.monotonic() - started < 5
    assert isinstance(kept.images[0].source, InlineSource) and error and "timed out" in error
