"""Stored media remains stable across expiration, retries, and missing objects."""

from __future__ import annotations
import time
from dataclasses import replace
from collections.abc import Callable, Mapping
import pytest
from actant.assets import (
    AssetContext,
    AssetReference,
    MissingAsset,
    ResolvedImage,
    prepare_messages,
)
from actant.llm.messages import Message
from actant.runtime.session import message_to_parts, parts_to_messages
from actant.storage.s3 import S3AssetResolver, S3Client
from actant.storage.sigv4 import SigningKeys, presign_get
from datetime import datetime, timezone
from typing import Literal, cast
from urllib.parse import parse_qs, quote, urlsplit

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
        self.calls += 1
        if self.error:
            raise self.error
        return {}


KEYS_ = SigningKeys("test-key", "test-secret")
WINDOW, BUFFER = 3600, 1800
WINDOW_START = 1_800_000_000 - 1_800_000_000 % WINDOW


def resolver_for(client: Client, clock: Callable[[], float] = time.time) -> S3AssetResolver:
    return S3AssetResolver(
        cast(S3Client, client),
        endpoint_url="https://account.r2.cloudflarestorage.com",
        region="auto",
        keys=KEYS_,
        bucket="b",
        prefix="images/",
        window_s=WINDOW,
        buffer_s=BUFFER,
        clock=clock,
    )


def test_signer_matches_the_aws_documented_presigned_url() -> None:
    """The worked example in AWS's "Authenticating Requests: Using Query Parameters"."""
    keys = SigningKeys("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    signed_at = int(datetime(2013, 5, 24, tzinfo=timezone.utc).timestamp())
    url = presign_get(
        "https://s3.amazonaws.com",
        "us-east-1",
        keys,
        "examplebucket",
        "test.txt",
        signed_at=signed_at,
        expires_s=86400,
        addressing_style="virtual",
    )
    assert url.endswith(
        "X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"
    )


KEYS = [
    "images/a.png",
    "images/a b+c=d&e.png",
    "images/ümlaut/こんにちは.jpg",
    "images/100%/x?y#z;,:@$!*'()[].webp",
    "images/a//b/~tilde.png",
]


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(
    "endpoint,style",
    [
        ("https://account.r2.cloudflarestorage.com", "path"),
        ("https://account.r2.cloudflarestorage.com", "virtual"),
        ("https://host:443", "virtual"),
        ("http://127.0.0.1:9000", "path"),
    ],
)
@pytest.mark.parametrize("token", [None, "session/token+=="])
def test_crt_signature_matches_botocores_signer_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    style: Literal["path", "virtual"],
    endpoint: str,
    token: str | None,
) -> None:
    """Two independent SigV4 implementations, AWS CRT (ours) and botocore's pure-Python
    signer, agree on every parameter and the signature."""
    import botocore.auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    moment = datetime(2026, 9, 25, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(botocore.auth, "get_current_datetime", lambda: moment)
    origin = urlsplit(endpoint)
    path = "/" + quote(key, safe="/~")
    url = (
        f"{origin.scheme}://bucket.{origin.netloc}{path}"
        if style == "virtual"
        else f"{endpoint}/bucket{path}"
    )
    request = AWSRequest(method="GET", url=url)
    credentials = Credentials("test-key", "test-secret", token)
    botocore.auth.S3SigV4QueryAuth(credentials, "s3", "auto", expires=5400).add_auth(request)
    theirs = urlsplit(request.url)

    ours = urlsplit(
        presign_get(
            endpoint,
            "auto",
            SigningKeys("test-key", "test-secret", session_token=token),
            "bucket",
            key,
            signed_at=int(moment.timestamp()),
            expires_s=5400,
            addressing_style=style,
        )
    )
    assert (ours.scheme, ours.netloc, ours.path) == (theirs.scheme, theirs.netloc, theirs.path)
    assert parse_qs(ours.query) == parse_qs(theirs.query)
    assert ("X-Amz-Security-Token" in parse_qs(ours.query)) == (token is not None)


async def test_one_url_per_window_never_near_expiry() -> None:
    client = Client()
    clock = [0.0]
    resolver = resolver_for(client, lambda: clock[0])
    asset = AssetReference("images/a.png", "image/png")
    for offset in (0, 1800, 3599.999, 3600, 7199.999):
        clock[0] = WINDOW_START + offset
        resolved = await resolver.resolve(asset, CONTEXT)
        assert isinstance(resolved, ResolvedImage) and resolved.expires_at is not None
        assert resolved.expires_at - clock[0] >= BUFFER
        # The fresh resolver another process would use signs the same URL.
        assert resolved == await resolver_for(Client(), lambda: clock[0]).resolve(asset, CONTEXT)
    assert client.calls == 5  # existence is checked on every resolve; nothing is remembered
    with pytest.raises(PermissionError):
        await resolver.resolve(replace(asset, storage_key="s3://other/images/a.png"), CONTEXT)
    with pytest.raises(PermissionError):
        await resolver.resolve(replace(asset, storage_key="private/a.png"), CONTEXT)
    with pytest.raises(ValueError, match="buffer"):
        await resolver.resolve(asset, replace(CONTEXT, minimum_validity_s=1700))


@pytest.mark.parametrize(
    "code,missing",
    [("NoSuchKey", True), ("404", True), ("AccessDenied", False), ("SlowDown", False)],
)
async def test_only_explicit_missing_sdk_errors_become_notes(code: str, missing: bool) -> None:
    class SDKError(Exception):
        response = {"Error": {"Code": code}}

    client = Client()
    client.error = SDKError()
    resolver = resolver_for(client)
    asset = AssetReference("images/a.png", "image/png")
    if missing:
        assert isinstance(await resolver.resolve(asset, CONTEXT), MissingAsset)
    else:
        with pytest.raises(SDKError):
            await resolver.resolve(asset, CONTEXT)


def test_short_resolved_url_cannot_reach_provider() -> None:
    with pytest.raises(ValueError, match="budget"):
        ResolvedImage("image/png", url="https://u", expires_at=time.time() + 10).to_block(600)
