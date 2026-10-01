"""`InlineImages`: pictures read from their URLs and sent in the request, fitted, kept."""

from __future__ import annotations

import asyncio
import io
import random
import urllib.error

import pytest
from PIL import Image

from actant import assets
from actant.assets import (
    AssetContext,
    AssetReference,
    InlineImages,
    MissingAsset,
    ResolvedImage,
    fit_image,
)

CONTEXT = AssetContext("agent", "thread", "run", "turn")
SIDE = 1568


class Presigned:
    """A store of one picture, presigned at a `file://` URL; None: the picture is gone.
    Counts the reads each key asks for."""

    def __init__(self, url: str | None) -> None:
        self.url = url
        self.reads: list[str] = []

    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset:
        self.reads.append(asset.storage_key)
        await asyncio.sleep(0)  # a read under way, which others asking at once share
        if self.url is None:
            return MissingAsset()
        return ResolvedImage(asset.mime, url=self.url)


def resolve(url: str | None, **options: int) -> ResolvedImage | MissingAsset:
    reference = AssetReference("s3://bucket/images/scene/abc", "image/png")
    images = InlineImages(Presigned(url), image_side=SIDE, **options)
    return asyncio.run(images.resolve(reference, CONTEXT))


def noise(size: tuple[int, int], mode: str) -> bytes:
    """A PNG no encoder shrinks much: random pixels."""

    image = Image.frombytes(mode, size, random.Random(0).randbytes(size[0] * size[1] * len(mode)))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def picture(size: tuple[int, int], format: str) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (200, 120, 40)).save(out, format=format)
    return out.getvalue()


def test_a_picture_too_big_goes_inline_reencoded_to_fit(tmp_path) -> None:
    photo = tmp_path / "photo.png"
    photo.write_bytes(noise((2600, 1400), "RGB"))
    assert photo.stat().st_size > assets.INLINE_IMAGE_BYTES
    found = resolve(photo.as_uri())
    assert isinstance(found, ResolvedImage) and found.url is None and found.data is not None
    assert found.mime == "image/jpeg" and len(found.data) <= assets.INLINE_IMAGE_BYTES
    with Image.open(io.BytesIO(found.data)) as image:
        assert max(image.size) <= SIDE
    # the same picture always comes out the same: the prompt cache keeps matching
    assert fit_image(photo.read_bytes(), SIDE) == (found.data, found.mime)


def test_a_transparent_picture_stays_png_when_reencoded() -> None:
    data, mime = fit_image(noise((900, 900), "RGBA"))
    assert mime == "image/png" and len(data) <= assets.INLINE_IMAGE_BYTES
    with Image.open(io.BytesIO(data)) as image:
        assert image.mode == "RGBA"


def test_a_picture_goes_inline_small_and_a_full_request_of_them_fits(tmp_path) -> None:
    render = tmp_path / "view.png"  # a render: a large PNG with nothing transparent
    render.write_bytes(picture((2048, 1400), "PNG"))
    found = resolve(render.as_uri())
    assert isinstance(found, ResolvedImage) and found.url is None and found.data is not None
    assert found.mime == "image/jpeg"
    with Image.open(io.BytesIO(found.data)) as image:
        assert max(image.size) == SIDE
    # a JPEG already small goes as it is
    photo = picture((1200, 900), "JPEG")
    assert fit_image(photo) == (photo, "image/jpeg")
    full = 50 * assets.INLINE_IMAGE_BYTES * 4 / 3
    assert full <= 0.9 * assets.READ_IMAGE_BYTES


def test_a_picture_is_never_sent_as_a_url(tmp_path, monkeypatch) -> None:
    """A picture the store will not give fails its turn, after `read_attempts`; one that is
    no picture is missing, said so. Neither goes as a URL."""

    junk = tmp_path / "junk.png"
    junk.write_bytes(b"x" * (assets.INLINE_IMAGE_BYTES + 1))
    found = resolve(junk.as_uri())
    assert isinstance(found, MissingAsset) and "not a picture" in found.reason
    reads: list[str] = []
    real = assets.read_image

    def flaky(url: str, limit: int, timeout_s: float = 30) -> bytes | None:
        reads.append(url)
        if len(reads) == 1:
            raise TimeoutError("the store did not answer")
        return real(url, limit, timeout_s)

    monkeypatch.setattr(assets, "read_image", flaky)
    render = tmp_path / "view.png"
    render.write_bytes(picture((64, 64), "PNG"))
    found = resolve(render.as_uri(), read_attempts=2)
    assert isinstance(found, ResolvedImage) and found.url is None and len(reads) == 2
    with pytest.raises(urllib.error.URLError):
        resolve((tmp_path / "gone.png").as_uri(), read_attempts=2)
    assert isinstance(resolve(None), MissingAsset)


def test_a_picture_larger_than_a_request_is_missing(tmp_path) -> None:
    photo = tmp_path / "photo.png"
    photo.write_bytes(picture((64, 64), "PNG"))
    found = resolve(photo.as_uri(), read_bytes=10)
    assert isinstance(found, MissingAsset) and "larger than" in found.reason


def test_a_threads_next_turn_sends_its_pictures_without_reading_them_again(tmp_path) -> None:
    photo = tmp_path / "photo.png"
    photo.write_bytes(picture((900, 600), "PNG"))
    store = Presigned(photo.as_uri())
    images = InlineImages(store, image_side=SIDE)
    keys = [AssetReference(f"s3://bucket/images/scene/{n}", "image/png") for n in range(4)]

    async def turn() -> list[ResolvedImage | MissingAsset]:
        return await asyncio.gather(*(images.resolve(key, CONTEXT) for key in keys))

    async def turns() -> tuple[list, list, list]:
        # two turns asking at once share each read; the next turn reads nothing
        first, second = await asyncio.gather(turn(), turn())
        return first, second, await turn()

    first, second, third = asyncio.run(turns())
    assert sorted(store.reads) == sorted(k.storage_key for k in keys)
    assert first == second == third  # the same bytes every turn: the prompt cache keeps matching


def test_the_kept_pictures_stay_within_their_bytes_least_recent_dropped(tmp_path) -> None:
    photo = tmp_path / "photo.png"
    photo.write_bytes(picture((900, 600), "PNG"))
    one = resolve(photo.as_uri())
    assert isinstance(one, ResolvedImage) and one.data is not None
    store = Presigned(photo.as_uri())
    images = InlineImages(store, image_side=SIDE, cache_bytes=2 * len(one.data))
    a, b, c = (AssetReference(f"s3://bucket/images/scene/{n}", "image/png") for n in "abc")

    async def send(*keys: AssetReference) -> None:
        for key in keys:
            await images.resolve(key, CONTEXT)

    asyncio.run(send(a, b, a, c))  # c drops b, the least recently sent
    asyncio.run(send(a, c, b))
    assert [r.rpartition("/")[2] for r in store.reads] == ["a", "b", "c", "b"]


def test_a_missing_picture_is_asked_for_again() -> None:
    store = Presigned(None)
    images = InlineImages(store)
    key = AssetReference("s3://bucket/images/scene/gone", "image/png")
    for _ in range(2):
        assert isinstance(asyncio.run(images.resolve(key, CONTEXT)), MissingAsset)
    assert len(store.reads) == 2
