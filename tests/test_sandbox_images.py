"""Images a service returns: storage keys from a host that uploads, bytes otherwise."""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import pytest
from botocore.exceptions import ClientError

from actant.sandbox import ImageBucket, SandboxSpec, host, uploads
from actant.sandbox.protocol import (
    CallRequest,
    CallResponse,
    Image,
    ImageUploadConfig,
    InlineSource,
    AssetSource,
)
from actant.sandbox.uploads import ImageUploader
from actant.blocks import AssetBlock, Base64Source, InlineImageBlock
from actant.tools import image_block
from actant.tools.base import MetadataKey
from actant.tools.service import to_tool_result
from service_fixtures import Counter

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 32

CONFIG = ImageUploadConfig(
    destination="s3://b/actant-images/t1/",
    endpoint_url="http://minio:9000",
)


@dataclass
class FakeS3:
    """Records each ``put_object``; ``fail`` refuses it, ``hang`` blocks it until released."""

    puts: list[dict[str, object]] = field(default_factory=list)
    threads: list[str] = field(default_factory=list)
    fail: bool = False
    hang: bool = False
    release: threading.Event = field(default_factory=threading.Event)

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str
    ) -> Mapping[str, object]:
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType})
        self.threads.append(threading.current_thread().name)
        if self.hang:
            self.release.wait(30)
        if self.fail:
            error = {"Error": {"Code": "AccessDenied", "Message": "denied"}}
            raise ClientError(error, "PutObject")
        return {}


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeS3]:
    fake = FakeS3()
    monkeypatch.setattr(uploads, "s3_client", lambda endpoint_url, timeout_s: fake)
    yield fake
    fake.release.set()


def _inline(name: str = "image-0", data: bytes = PNG) -> Image:
    source = InlineSource(data_b64=base64.b64encode(data).decode())
    return Image(name=name, media_type="image/png", source=source)


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


async def test_upload_is_content_addressed_and_never_presigns(s3: FakeS3) -> None:
    uploader = ImageUploader(CONFIG, s3)
    uploaded, error = await host.upload_images(
        CallResponse(text="t", images=[_inline()]), uploader
    )
    assert error is None and uploaded.text == "t"
    [image] = uploaded.images
    assert isinstance(image.source, AssetSource)
    assert image.source.storage_key.startswith(CONFIG.destination)
    [put] = s3.puts
    assert f"s3://{put['Bucket']}/{put['Key']}" == image.source.storage_key
    assert put["Body"] == PNG and put["ContentType"] == "image/png"
    assert image.source.storage_key.endswith(".png")
    again, _ = await host.upload_images(CallResponse(images=[_inline()]), uploader)
    assert again.images[0].source == image.source


async def test_failed_upload_preserves_bytes_and_reports_reason(s3: FakeS3) -> None:
    s3.fail = True
    response = CallResponse(images=[_inline("a.png")])
    kept, error = await host.upload_images(response, ImageUploader(CONFIG, s3))
    assert kept == response
    assert error and error.startswith("a.png: upload failed") and "denied" in error


async def test_host_tool_result_uses_durable_reference_and_reports_upload_failure(
    s3: FakeS3,
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
    s3.fail = True
    _, fallback = await uploading.call(request)
    assert isinstance(fallback.images[0].source, InlineSource)
    assert "upload failed" in str(to_tool_result(fallback).metadata[MetadataKey.STORAGE])
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
    s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host, "UPLOAD_CONCURRENCY", 1)
    s3.fail = True
    images = [_inline(f"i{n}.png", PNG + bytes([n])) for n in range(3)]
    kept, error = await host.upload_images(CallResponse(images=images), ImageUploader(CONFIG, s3))
    assert kept.images == images and error and error.startswith("i0.png: upload")
    assert len(s3.puts) == 1


async def test_hung_upload_is_bounded_and_preserves_bytes(
    s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uploads, "UPLOAD_BACKOFF_S", 0.0)
    s3.hang = True
    uploader = ImageUploader(CONFIG.model_copy(update={"timeout_s": 0.1}), s3)
    started = time.monotonic()
    kept, error = await host.upload_images(CallResponse(images=[_inline()]), uploader)
    assert time.monotonic() - started < 5
    assert isinstance(kept.images[0].source, InlineSource)
    assert error == "image-0: upload timed out after 0.6s"  # 3 attempts' connect and read


async def test_an_upload_is_not_timed_while_it_waits_for_a_thread(
    s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uploads queued behind a busy pool each get their whole budget once started."""
    monkeypatch.setattr(uploads, "UPLOAD_BACKOFF_S", 0.0)
    s3.hang = True
    uploader = ImageUploader(CONFIG.model_copy(update={"timeout_s": 0.1}), s3)
    uploader.executor = ThreadPoolExecutor(1)
    loop = asyncio.get_running_loop()
    loop.call_later(0.8, s3.release.set)  # the first holds the only thread for 0.8 s
    first, second = await asyncio.gather(
        uploader.upload(_inline("a", PNG + b"a")), uploader.upload(_inline("b", PNG + b"b"))
    )
    assert first[1] == "a: upload timed out after 0.6s"
    assert second[1] is None and isinstance(second[0].source, AssetSource)


async def test_uploads_share_one_client_on_their_own_threads(s3: FakeS3) -> None:
    uploading = host.Host({"counter": Counter}, images=CONFIG)
    assert uploading.uploader is not None and uploading.uploader.client is s3
    request = CallRequest(service="counter", key="k", method="picture", args={"size": 4})
    for _ in range(3):
        await uploading.call(request)
    assert len(s3.threads) == 3 and all(name.startswith("actant-upload") for name in s3.threads)


def test_client_reads_the_environment_and_uses_path_style_for_an_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_REGION", "auto")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    meta = uploads.s3_client("http://minio:9000", 7.0).meta  # pyright: ignore[reportAttributeAccessIssue]
    config = meta.config
    assert meta.region_name == "auto" and meta.endpoint_url == "http://minio:9000"
    assert config.s3 == {"addressing_style": "path"}
    assert config.max_pool_connections == uploads.UPLOAD_THREADS
    assert config.read_timeout == 7.0 and config.connect_timeout == 7.0
    assert config.retries == {"mode": "standard", "total_max_attempts": uploads.UPLOAD_ATTEMPTS}
    monkeypatch.delenv("AWS_REGION")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    default = uploads.s3_client(None, 10.0).meta  # pyright: ignore[reportAttributeAccessIssue]
    assert default.region_name == "eu-west-1" and default.config.s3 is None
