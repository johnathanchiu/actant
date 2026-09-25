"""Stored media remains stable across expiration, retries, and missing objects."""

from __future__ import annotations
import time
from dataclasses import replace
from collections.abc import Mapping
import pytest
from actant.assets import (
    AssetContext,
    AssetReference,
    InMemorySignedUrls,
    MissingAsset,
    ResolvedImage,
    SignedUrl,
    prepare_messages,
)
from actant.llm.messages import Message
from actant.runtime.session import message_to_parts, parts_to_messages
from actant.storage.s3 import S3AssetResolver, S3Client
from typing import cast

CONTEXT = AssetContext("a", "t", "r", "turn")


class Resolver:
    def __init__(self) -> None:
        self.seen: list[str] = []
        self.result: ResolvedImage | MissingAsset = ResolvedImage("image/png", data=b"png")

    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset:
        self.seen.append(asset.storage_key)
        return self.result


async def test_asset_preparation_preserves_stored_history_and_message_metadata() -> None:
    block = AssetReference("images/a.png", "image/png").to_block()
    original = Message(role="user", content=[block], input_tokens=7)
    resolver = Resolver()
    [prepared] = await prepare_messages([original], resolver, CONTEXT)
    assert prepared.content != original.content
    assert original.content == [block] and prepared.input_tokens == 7
    assert parts_to_messages(message_to_parts(original))[0].content == [block]
    assert resolver.seen == ["images/a.png"]


async def test_legacy_expired_url_resolves_only_with_explicit_reference() -> None:
    source = {"type": "url", "url": "https://old", "expires_at": 1, "key": "images/a.png"}
    resolver = Resolver()
    messages = [
        Message(role="tool", tool_call_id="c", content=[{"type": "image", "source": source}])
    ]
    [prepared] = await prepare_messages(messages, resolver, CONTEXT)
    assert prepared.tool_call_id == "c" and "base64" in str(prepared.content)
    del source["key"]
    [missing] = await prepare_messages(messages, resolver, CONTEXT)
    assert "no storage reference" in str(missing.content)
    assert resolver.seen == ["images/a.png"]


async def test_missing_is_visible_but_storage_failure_is_not_missing() -> None:
    resolver = Resolver()
    resolver.result = MissingAsset("deleted")
    messages = [
        Message(role="user", content=[AssetReference("images/a.png", "image/png").to_block()])
    ]
    assert "deleted" in str((await prepare_messages(messages, resolver, CONTEXT))[0].content)

    class Failing(Resolver):
        async def resolve(self, asset: AssetReference, context: AssetContext) -> ResolvedImage:
            raise PermissionError("access denied")

    with pytest.raises(PermissionError):
        await prepare_messages(messages, Failing(), CONTEXT)
    with pytest.raises(ValueError, match="AssetResolver"):
        await prepare_messages(messages, None, CONTEXT)


async def test_legacy_live_url_is_sanitized_without_mutating_history() -> None:
    source = {"type": "url", "url": "https://live", "expires_at": time.time() + 2000, "key": "k"}
    message = Message(role="user", content=[{"type": "image", "source": source}])
    [prepared] = await prepare_messages([message], None, CONTEXT)
    assert prepared.content == [
        {"type": "image", "source": {"type": "url", "url": "https://live"}}
    ]
    assert "expires_at" in source and "key" in source


class Client:
    calls = 0
    error: Exception | None = None

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]:
        if self.error:
            raise self.error
        return {}

    def generate_presigned_url(
        self, ClientMethod: str, *, Params: dict[str, str], ExpiresIn: int
    ) -> str:
        self.calls += 1
        return f"https://bucket/{Params['Key']}?signature={self.calls}"


async def test_sdk_signing_cache_refresh_and_prefix_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    resolver = S3AssetResolver(
        cast(S3Client, client), bucket="b", prefix="images/", urls=InMemorySignedUrls()
    )
    asset = AssetReference("s3://b/images/a.png", "image/png")
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    first = await resolver.resolve(asset, CONTEXT)
    assert first == await resolver.resolve(asset, CONTEXT) and client.calls == 1
    monkeypatch.setattr(time, "time", lambda: now + 3000)
    assert first != await resolver.resolve(asset, CONTEXT) and client.calls == 2
    with pytest.raises(PermissionError):
        await resolver.resolve(replace(asset, storage_key="s3://other/images/a.png"), CONTEXT)
    with pytest.raises(PermissionError):
        await resolver.resolve(replace(asset, storage_key="private/a.png"), CONTEXT)


async def test_processes_and_restarts_share_one_url_until_it_nears_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each resolver stands for a worker process: the SDK signs differently every call, yet
    every process, and one started later, hands out the stored URL until it nears expiry."""
    urls = InMemorySignedUrls()
    client = Client()
    one, two = (
        S3AssetResolver(cast(S3Client, client), bucket="b", prefix="images/", urls=urls)
        for _ in range(2)
    )
    asset = AssetReference("images/a.png", "image/png")
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    first = await one.resolve(asset, CONTEXT)
    assert first == await two.resolve(asset, CONTEXT) and client.calls == 1
    monkeypatch.setattr(time, "time", lambda: now + 3600 - 721)
    restarted = S3AssetResolver(cast(S3Client, client), bucket="b", prefix="images/", urls=urls)
    assert first == await restarted.resolve(asset, CONTEXT) and client.calls == 1
    monkeypatch.setattr(time, "time", lambda: now + 3600 - 720)
    second = await two.resolve(asset, CONTEXT)
    assert second != first and client.calls == 2
    assert second == await one.resolve(asset, CONTEXT) and client.calls == 2


async def test_concurrent_signers_all_use_the_url_that_was_stored_first() -> None:
    urls = InMemorySignedUrls()
    late = time.time() + 3600
    stored = await urls.put("s3://b/k", SignedUrl("https://one", late), replace_before=0)
    assert stored == await urls.put("s3://b/k", SignedUrl("https://two", late), replace_before=0)


@pytest.mark.parametrize(
    "code,missing",
    [("NoSuchKey", True), ("404", True), ("AccessDenied", False), ("SlowDown", False)],
)
async def test_only_explicit_missing_sdk_errors_become_notes(code: str, missing: bool) -> None:
    class SDKError(Exception):
        response = {"Error": {"Code": code}}

    client = Client()
    client.error = SDKError()
    resolver = S3AssetResolver(
        cast(S3Client, client), bucket="b", prefix="images/", urls=InMemorySignedUrls()
    )
    asset = AssetReference("images/a.png", "image/png")
    if missing:
        assert isinstance(await resolver.resolve(asset, CONTEXT), MissingAsset)
    else:
        with pytest.raises(SDKError):
            await resolver.resolve(asset, CONTEXT)


def test_short_resolved_url_cannot_reach_provider() -> None:
    with pytest.raises(ValueError, match="budget"):
        ResolvedImage("image/png", url="https://u", expires_at=time.time() + 10).to_block(600)


async def test_real_boto_sdk_signs_and_cache_keeps_same_request_prefix() -> None:
    import boto3
    from botocore.config import Config
    from botocore.stub import Stubber

    client = boto3.client(
        "s3",
        endpoint_url="https://storage.example",
        region_name="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        config=Config(signature_version="s3v4"),
    )
    try:
        with Stubber(client) as stub:
            stub.add_response(
                "head_object", {"ContentLength": 3}, {"Bucket": "b", "Key": "images/a.png"}
            )
            resolver = S3AssetResolver(
                cast(S3Client, client), bucket="b", prefix="images/", urls=InMemorySignedUrls()
            )
            asset = AssetReference("images/a.png", "image/png")
            first = await resolver.resolve(asset, CONTEXT)
            assert (
                isinstance(first, ResolvedImage) and first.url and "X-Amz-Signature=" in first.url
            )
            assert first == await resolver.resolve(asset, CONTEXT)
            stub.assert_no_pending_responses()
    finally:
        client.close()
