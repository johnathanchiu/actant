"""Images a service returns: presigned URLs from a host that uploads, bytes otherwise."""

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
    UrlSource,
)
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
    public_endpoint_url="https://public.example",
    expires_s=3600,
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


def test_image_block_prefers_the_url_and_otherwise_sends_bytes() -> None:
    inline = _inline()
    assert image_block(inline) == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": inline.source.data_b64},  # pyright: ignore[reportAttributeAccessIssue]
    }
    linked = inline.model_copy(update={"source": UrlSource(url="https://u/x.png", expires_at=1)})
    assert image_block(linked) == {
        "type": "image",
        "source": {"type": "url", "url": "https://u/x.png", "expires_at": 1},
    }


def test_the_source_round_trips_by_its_kind() -> None:
    response = CallResponse(
        images=[
            _inline(),
            Image(
                name="a.png",
                media_type="image/png",
                source=UrlSource(url="https://u", expires_at=5.0),
            ),
        ]
    )
    parsed = CallResponse.model_validate_json(response.to_json())
    assert parsed == response
    assert isinstance(parsed.images[0].source, InlineSource)
    assert isinstance(parsed.images[1].source, UrlSource)


async def test_an_upload_is_content_addressed_and_presigned_for_the_public_endpoint(
    s5cmd_log: Path,
) -> None:
    before = time.time()
    uploaded, error = await host.upload_images(CallResponse(text="t", images=[_inline()]), CONFIG)
    assert error is None and uploaded.text == "t"
    [image] = uploaded.images
    assert isinstance(image.source, UrlSource)
    assert image.source.url.startswith("https://public.example/b/actant-images/t1/")
    assert image.source.url.split("?")[0].endswith(".png")
    assert image.source.url.split("?")[0].endswith("/b/" + str(image.source.key))
    assert before + 3600 <= image.source.expires_at <= time.time() + 3600
    pipe, presign = [json.loads(line) for line in s5cmd_log.read_text().splitlines()]
    assert pipe["argv"][:2] == ["--endpoint-url", "http://minio:9000"]
    assert pipe["stdin"] == len(PNG)
    assert pipe["argv"][2:5] == ["pipe", "--content-type", "image/png"]
    assert presign["argv"][:5] == [
        "--endpoint-url",
        "https://public.example",
        "presign",
        "--expire",
        "3600s",
    ]
    assert presign["argv"][-1] == pipe["argv"][-1]


@pytest.mark.parametrize("fail", ["pipe", "presign"])
async def test_a_failed_upload_or_presign_keeps_the_bytes_and_says_why(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    monkeypatch.setenv("FAKE_S5CMD_FAIL", fail)
    response = CallResponse(images=[_inline("a.png")])
    kept, error = await host.upload_images(response, CONFIG)
    assert kept == response
    assert (
        error is not None
        and error.startswith(f"a.png: {'upload' if fail == 'pipe' else 'presign'} exited 1")
        and "denied" in error
    )


async def test_presign_output_that_is_not_a_url_keeps_the_bytes(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_S5CMD_URL", "garbage")
    kept, error = await host.upload_images(CallResponse(images=[_inline()]), CONFIG)
    assert isinstance(kept.images[0].source, InlineSource) and error and "not a URL" in error


async def test_a_missing_s5cmd_never_fails_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    kept, error = await host.upload_images(CallResponse(images=[_inline()]), CONFIG)
    assert isinstance(kept.images[0].source, InlineSource) and error and "could not start" in error


async def test_a_host_that_uploads_reports_image_errors_on_storage_status(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploading = host.Host({"counter": Counter}, images=CONFIG)
    request = CallRequest(service="counter", key="k", method="picture", args={"size": 4})
    _, response = await uploading.call(request)
    assert isinstance(response.images[0].source, UrlSource)
    assert response.storage is not None and response.storage.image_error is None
    result = to_tool_result(response)
    assert result.content_blocks and result.content_blocks[-1]["source"] == {  # pyright: ignore[reportIndexIssue]
        "type": "url",
        "url": response.images[0].source.url,
        "expires_at": response.images[0].source.expires_at,
        "key": response.images[0].source.key,
    }

    monkeypatch.setenv("FAKE_S5CMD_FAIL", "pipe")
    _, fallback = await uploading.call(request)
    assert isinstance(fallback.images[0].source, InlineSource)
    assert fallback.storage is not None and "upload exited 1" in str(fallback.storage.image_error)
    metadata = to_tool_result(fallback).metadata[MetadataKey.STORAGE]
    assert "upload exited 1" in str(metadata["image_error"])  # pyright: ignore[reportIndexIssue]

    # A text-only call on the same host carries status, with nothing to report.
    _, plain = await uploading.call(
        CallRequest(service="counter", key="k", method="length", args={"text": "ab"})
    )
    assert plain.storage is not None and plain.storage.image_error is None


def test_the_spec_bounds_the_url_lifetime_and_buckets_key_images_beside_the_thread() -> None:
    assert SandboxSpec().image_url_ttl_s == 6 * 3600
    for bad in (0, 7 * 24 * 3600 + 1):
        with pytest.raises(ValueError, match="image_url_ttl_s"):
            SandboxSpec(image_url_ttl_s=bad)
    bucket = ImageBucket("b", public_endpoint_url="https://public.example")
    assert host.image_upload_config(bucket, SandboxSpec(image_url_ttl_s=None), "t1") is None
    config = host.image_upload_config(bucket, SandboxSpec(), "t1")
    assert config == ImageUploadConfig(
        destination="s3://b/actant-images/t1/",
        public_endpoint_url="https://public.example",
        expires_s=6 * 3600,
        timeout_s=10,
    )


async def test_after_a_failure_images_not_yet_started_stay_inline_untried(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host, "UPLOAD_CONCURRENCY", 1)
    monkeypatch.setenv("FAKE_S5CMD_FAIL", "pipe")
    images = [_inline(f"i{n}.png", PNG + bytes([n])) for n in range(3)]
    kept, error = await host.upload_images(CallResponse(images=images), CONFIG)
    assert kept.images == images and error and error.startswith("i0.png: upload")
    assert len(s5cmd_log.read_text().splitlines()) == 1


async def test_without_an_upload_endpoint_uploads_go_to_aws(s5cmd_log: Path) -> None:
    aws = ImageUploadConfig(
        destination="s3://b/p/", public_endpoint_url="https://s3.amazonaws.com", expires_s=60
    )
    uploaded, error = await host.upload_images(CallResponse(images=[_inline()]), aws)
    assert error is None and isinstance(uploaded.images[0].source, UrlSource)
    pipe, presign = [json.loads(line)["argv"] for line in s5cmd_log.read_text().splitlines()]
    assert pipe[0] == "pipe" and presign[:2] == ["--endpoint-url", "https://s3.amazonaws.com"]


async def test_a_hung_bucket_costs_one_budget_per_image_then_bytes(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_S5CMD_SLEEP_PIPE", "30")
    started = time.monotonic()
    budget = CONFIG.model_copy(update={"timeout_s": 0.5})
    kept, error = await host.upload_images(CallResponse(images=[_inline()]), budget)
    assert time.monotonic() - started < 5
    assert (
        isinstance(kept.images[0].source, InlineSource)
        and error
        and "timed out after 0.5s" in error
    )


async def test_a_presign_shares_what_the_upload_left_and_does_not_stop_other_images(
    s5cmd_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host, "UPLOAD_CONCURRENCY", 1)
    monkeypatch.setenv("FAKE_S5CMD_SLEEP_PIPE", "0.4")
    monkeypatch.setenv("FAKE_S5CMD_SLEEP_PRESIGN", "5")
    budget = CONFIG.model_copy(update={"timeout_s": 1.0})
    images = [_inline("a.png"), _inline("b.png", PNG + b"b")]
    started = time.monotonic()
    kept, error = await host.upload_images(CallResponse(images=images), budget)
    # Each image costs at most its budget (plus process start and kill), not the presign's 5 s.
    assert time.monotonic() - started < 2 * 1.0 + 2
    assert kept.images == images and error and error.startswith("a.png: presign timed out")
    # The presign timeout did not skip b.png: both uploaded.
    commands = [json.loads(line)["argv"] for line in s5cmd_log.read_text().splitlines()]
    assert sum("pipe" in argv for argv in commands) == 2


def test_a_bucket_needs_a_public_endpoint_and_the_spec_a_positive_upload_timeout() -> None:
    with pytest.raises(TypeError):
        ImageBucket("b")  # pyright: ignore[reportCallIssue]
    for bad_url in ("127.0.0.1:9000", "https://", "http:///x", "https://a b"):
        with pytest.raises(ValueError, match="public_endpoint_url"):
            ImageBucket("b", public_endpoint_url=bad_url)
    for bad_timeout in (0, -1, float("nan"), float("inf"), True, "10"):
        with pytest.raises(ValueError, match="image_upload_timeout_s"):
            SandboxSpec(image_upload_timeout_s=bad_timeout)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="image_url_ttl_s"):
        SandboxSpec(image_url_ttl_s=True)
    config = host.image_upload_config(
        ImageBucket("b", "https://p.example"), SandboxSpec(image_upload_timeout_s=3), "t"
    )
    assert config is not None and config.timeout_s == 3
