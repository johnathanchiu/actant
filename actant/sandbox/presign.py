"""Sign expired image URLs again on the worker, so a resumed thread keeps its images.

A host stores each returned image's bucket ``key`` beside its presigned URL. Before a
turn, the worker's ``image_signer`` replaces every URL that has expired (or will within
``EXPIRY_MARGIN_S``) with a fresh one. Nothing is written back: :func:`sigv4_signer`
returns the same URL for a key all through a time window, so turns in that window send
identical bytes and the prompt cache holds. An image without a key, older than
``max_age_s`` (its object likely deleted by the lifecycle rule), or with no signer
configured, is left for the adapters to replace with a note.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import quote, urlsplit

from actant.llm.messages import Message
from actant.llm.providers._shared import EXPIRY_MARGIN_S
from actant.sandbox.base import MAX_PRESIGN_S

#: ``(key, ttl_s) -> url``: a URL for the bucket object ``key`` living at least ``ttl_s``.
ImageSigner = Callable[[str, int], str]
#: Matches an eight-day lifecycle rule on the image prefix, less a day of margin.
MAX_IMAGE_AGE_S = 7 * 24 * 3600


def sigv4_signer(
    bucket: str,
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    region: str,
    *,
    window_s: int = 3600,
) -> ImageSigner:
    """An :data:`ImageSigner` presigning path-style S3 GETs against ``endpoint_url`` (the
    host the model provider reaches: ``public_endpoint_url``). The signing time is floored
    to ``window_s`` and the URL lives ``ttl_s + window_s`` (at most seven days), so one key
    signs to one URL per window and every URL lives at least ``ttl_s``."""
    host = urlsplit(endpoint_url).netloc
    base = endpoint_url.rstrip("/")

    def sign(key: str, ttl_s: int) -> str:
        signed_at = int(time.time()) // window_s * window_s
        stamp = datetime.fromtimestamp(signed_at, UTC).strftime("%Y%m%dT%H%M%SZ")
        scope = f"{stamp[:8]}/{region}/s3/aws4_request"
        path = f"/{bucket}/{quote(key, safe='/-_.~')}"
        query = "&".join(
            f"{name}={quote(value, safe='-_.~')}"
            for name, value in sorted(
                {
                    "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                    "X-Amz-Credential": f"{access_key}/{scope}",
                    "X-Amz-Date": stamp,
                    "X-Amz-Expires": str(min(ttl_s + window_s, MAX_PRESIGN_S)),
                    "X-Amz-SignedHeaders": "host",
                }.items()
            )
        )
        request = f"GET\n{path}\n{query}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
        to_sign = "\n".join(
            ["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(request.encode()).hexdigest()]
        )
        signing_key = f"AWS4{secret_key}".encode()
        for part in (stamp[:8], region, "s3", "aws4_request"):
            signing_key = hmac.new(signing_key, part.encode(), hashlib.sha256).digest()
        signature = hmac.new(signing_key, to_sign.encode(), hashlib.sha256).hexdigest()
        return f"{base}{path}?{query}&X-Amz-Signature={signature}"

    return sign


def sign_expired_images(
    messages: Sequence[Message],
    signer: ImageSigner,
    ttl_s: int,
    *,
    now: float,
    max_age_s: float = MAX_IMAGE_AGE_S,
) -> list[Message]:
    """``messages`` with each expiring URL image that has a ``key`` and was presigned (at
    ``expires_at - ttl_s``) less than ``max_age_s`` ago signed again. Other messages and
    blocks are returned as they are."""
    signed: list[Message] = []
    for message in messages:
        if isinstance(message.content, list):
            content = [_signed(block, signer, ttl_s, now, max_age_s) for block in message.content]
            if any(new is not old for new, old in zip(content, message.content)):
                message = replace(message, content=content)
        signed.append(message)
    return signed


def _signed(
    block: dict[str, object], signer: ImageSigner, ttl_s: int, now: float, max_age_s: float
) -> dict[str, object]:
    source = block.get("source")
    if block.get("type") != "image" or not isinstance(source, dict):
        return block
    key, expires_at = source.get("key"), source.get("expires_at")
    if (
        source.get("type") != "url"
        or not isinstance(key, str)
        or not isinstance(expires_at, int | float)
        or expires_at > now + EXPIRY_MARGIN_S
        or now - (expires_at - ttl_s) >= max_age_s
    ):
        return block
    fresh = {"type": "url", "url": signer(key, ttl_s), "expires_at": now + ttl_s, "key": key}
    return {**block, "source": fresh}
