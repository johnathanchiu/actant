"""Images a service returns: bytes across the sandbox boundary, stored by the worker."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from actant.blocks import AssetBlock, Base64Source, InlineImageBlock, TextBlock
from actant.sandbox import host
from actant.sandbox.protocol import CallResponse, Image, InlineSource
from actant.sandbox.service import LocalRunner
from actant.tools import tools
from actant.tools.base import CallContext
from actant.storage.s3 import S3AssetStore
from actant.tools.service import to_tool_result

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 32
INLINE = InlineImageBlock(
    source=Base64Source(media_type="image/png", data=base64.b64encode(PNG).decode())
)
DIGEST = hashlib.sha256(PNG).hexdigest()


@dataclass
class FakeStore:
    """An :class:`~actant.assets.AssetStore` that records each ``put``."""

    prefix: str = "worker/"
    puts: list[tuple[str, bytes, str]] = field(default_factory=list)

    async def put_image(self, thread_id: str, data: bytes, mime: str) -> str:
        key = f"{thread_id}/images/{hashlib.sha256(data).hexdigest()}.png"
        self.puts.append((key, data, mime))
        return self.prefix + key


@dataclass
class FakeS3:
    puts: list[dict[str, object]] = field(default_factory=list)

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str
    ) -> dict[str, object]:
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType})
        return {}


async def test_s3_store_derives_a_content_addressed_key_under_its_prefix() -> None:
    s3 = FakeS3()
    store = S3AssetStore(s3, bucket="b", prefix="assets/")
    key = await store.put_image("t1", PNG, "image/png")
    assert key == f"assets/t1/images/{hashlib.sha256(PNG).hexdigest()}.png"
    assert s3.puts == [{"Bucket": "b", "Key": key, "Body": PNG, "ContentType": "image/png"}]
    for thread in ("../t1", "t1/x", ""):
        with pytest.raises(ValueError):
            await store.put_image(thread, PNG, "image/png")


def _inline(name: str = "image-0", data: bytes = PNG) -> Image:
    source = InlineSource(data_b64=base64.b64encode(data).decode())
    return Image(name=name, media_type="image/png", source=source)


def test_inline_source_round_trips() -> None:
    response = CallResponse(images=[_inline()])
    assert CallResponse.model_validate_json(response.to_json()) == response


async def test_images_are_stored_in_the_worker_store_and_referenced_by_its_key() -> None:
    store = FakeStore()
    response = CallResponse(text="t", images=[_inline("a.png")])
    result = await to_tool_result(response, thread_id="t1", store=store)
    assert store.puts == [(f"t1/images/{DIGEST}.png", PNG, "image/png")]
    assert result.content_blocks == [
        TextBlock(text="t"),
        TextBlock(text="Image a.png:"),
        AssetBlock(storage_key=f"worker/t1/images/{DIGEST}.png", mime="image/png"),
    ]


async def test_the_same_bytes_are_stored_once() -> None:
    store = FakeStore()
    other = PNG + b"y"
    response = CallResponse(images=[_inline("a.png"), _inline("b.png"), _inline("c.png", other)])
    result = await to_tool_result(response, thread_id="t1", store=store)
    assert [key for key, _, _ in store.puts] == [
        f"t1/images/{DIGEST}.png",
        f"t1/images/{hashlib.sha256(other).hexdigest()}.png",
    ]
    blocks = [b for b in result.content_blocks or [] if isinstance(b, AssetBlock)]
    assert blocks[0] == blocks[1] != blocks[2]


async def test_without_a_store_images_stay_inline() -> None:
    result = await to_tool_result(CallResponse(images=[_inline()]), thread_id="t1", store=None)
    assert result.content_blocks == [TextBlock(text="Image image-0:"), INLINE]


async def test_a_response_over_the_image_cap_fails_and_stores_nothing() -> None:
    store = FakeStore()
    response = CallResponse(images=[_inline("a.png"), _inline("b.png")])
    result = await to_tool_result(
        response, thread_id="t1", store=store, max_image_bytes=len(PNG) * 2 - 1
    )
    assert result.error and "byte limit for one response" in result.error
    assert store.puts == []
    fits = await to_tool_result(
        response, thread_id="t1", store=store, max_image_bytes=len(PNG) * 2
    )
    assert fits.error is None


async def test_a_service_tool_stores_its_images_in_the_call_context_store() -> None:
    @dataclass
    class Shown:
        text: str
        images: list[object]

    class Camera:
        def snap(self) -> Shown:
            """Take a picture."""
            return Shown("ok", [PNG])

    store = FakeStore()
    [tool] = tools(Camera, LocalRunner(Camera()))
    ctx = CallContext(
        agent_id="a", thread_id="t1", run_id="r", tool_call_id="c", turn_id="u", asset_store=store
    )
    result = await (await tool.build({}, ctx)).execute()
    assert result.content_blocks and result.content_blocks[-1] == AssetBlock(
        storage_key=f"worker/t1/images/{DIGEST}.png", mime="image/png"
    )


def test_caption_round_trips_and_old_messages_have_none() -> None:
    captioned = _inline().model_copy(update={"caption": "Kitchen from the door"})
    response = CallResponse(images=[captioned, _inline("b.png")])
    assert CallResponse.model_validate_json(response.to_json()) == response
    old = '{"images": [{"name": "a.png", "media_type": "image/png", "source": {"kind": "inline", "data_b64": ""}}]}'
    assert CallResponse.model_validate_json(old).images[0].caption is None


async def test_caption_sits_directly_before_its_image_else_the_name_label() -> None:
    captioned = _inline("a.png").model_copy(update={"caption": "Kitchen from the door"})
    response = CallResponse(text="t", images=[captioned, _inline("b.png")])
    result = await to_tool_result(response, thread_id="t1", store=None)
    assert result.content_blocks == [
        TextBlock(text="t"),
        TextBlock(text="Kitchen from the door"),
        INLINE,
        TextBlock(text="Image b.png:"),
        INLINE,
    ]


def test_a_method_captions_an_image_with_a_pair(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(PNG)

    @dataclass
    class Shown:
        text: str
        images: list[object]

    response = host.encode_output(Shown("t", [(str(path), "Kitchen"), PNG, (PNG, "Close-up")]))
    assert [image.caption for image in response.images] == ["Kitchen", None, "Close-up"]
    assert response.images[0].name == str(path)
