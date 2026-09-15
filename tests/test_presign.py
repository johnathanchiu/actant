"""Expired image URLs signed again on the worker: deterministic per window, a note otherwise."""

from __future__ import annotations

import time
from typing import cast

import pytest

from actant.llm.messages import Message
from actant.llm.providers._shared import EXPIRED_IMAGE, sanitize_tool_messages
from actant.sandbox import presign
from actant.sandbox.presign import sign_expired_images, sigv4_signer

TTL = 6 * 3600
KEY = "actant-images/t1/a b.png"


def _message(expires_at: float, key: str | None = KEY) -> Message:
    source: dict[str, object] = {"type": "url", "url": "https://old", "expires_at": expires_at}
    if key is not None:
        source["key"] = key
    return Message(role="tool", tool_call_id="c", content=[{"type": "image", "source": source}])


def _signer(key: str, ttl_s: int) -> str:
    return f"https://fresh/{key}?ttl={ttl_s}"


def _sent(messages: list[Message]) -> object:
    [sent] = sanitize_tool_messages(messages)
    return cast("list[object]", sent.content)[0]


def test_an_expired_image_with_a_key_is_signed_again() -> None:
    now = time.time()
    [signed] = sign_expired_images([_message(now - 60)], _signer, TTL, now=now)
    assert _sent([signed]) == {
        "type": "image",
        "source": {"type": "url", "url": f"https://fresh/{KEY}?ttl={TTL}"},
    }


def test_an_image_with_time_left_is_the_same_message() -> None:
    now = time.time()
    stored = _message(now + 3600)
    assert sign_expired_images([stored], _signer, TTL, now=now)[0] is stored


@pytest.mark.parametrize(
    ("message", "signer"),
    [
        (_message(time.time() - 60, key=None), _signer),
        # Presigned eight days ago: the lifecycle rule has likely deleted the object.
        (_message(time.time() - 8 * 86400 + TTL), _signer),
    ],
)
def test_no_key_or_an_old_image_stays_expired_and_becomes_a_note(
    message: Message, signer: presign.ImageSigner
) -> None:
    signed = sign_expired_images([message], signer, TTL, now=time.time())
    assert signed[0] is message
    assert _sent(signed) == {"type": "text", "text": EXPIRED_IMAGE}


def test_the_sigv4_url_is_identical_within_a_window_and_matches_s5cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sign = sigv4_signer(
        "bkt",
        "https://acct.r2.example.com",
        "AKIDEXAMPLE",
        "secret/xyz+abc",
        "auto",
        window_s=3600,
    )
    start = 1789464969 // 3600 * 3600  # 2026-09-15T09:00:00Z
    monkeypatch.setattr(presign.time, "time", lambda: start + 5)
    first = sign(KEY, 3600)
    monkeypatch.setattr(presign.time, "time", lambda: start + 3599)
    assert sign(KEY, 3600) == first
    monkeypatch.setattr(presign.time, "time", lambda: start + 3600)
    assert sign(KEY, 3600) != first
    # Checked against `s5cmd presign --expire 25200s` signing at 09:36:09Z.
    monkeypatch.setattr(presign.time, "time", lambda: 1789464969)
    assert sigv4_signer(
        "bkt", "https://acct.r2.example.com", "AKIDEXAMPLE", "secret/xyz+abc", "auto", window_s=1
    )(KEY, 25199) == (
        "https://acct.r2.example.com/bkt/actant-images/t1/a%20b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=AKIDEXAMPLE%2F20260915%2Fauto%2Fs3%2Faws4_request"
        "&X-Amz-Date=20260915T093609Z&X-Amz-Expires=25200&X-Amz-SignedHeaders=host"
        "&X-Amz-Signature=08e5b1d7c222fbf50f5aefd5109b9dc5f6b9c51daf04856b322b047894768f09"
    )
