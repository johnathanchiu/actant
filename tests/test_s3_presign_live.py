"""A URL from ``presign_get`` downloads a real object, as botocore's does.

Opt in with a reachable S3 API (MinIO or R2) and a bucket that already exists::

    ACTANT_TEST_S3_ENDPOINT=http://127.0.0.1:9000 ACTANT_TEST_S3_BUCKET=... \\
        AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \\
        uv run pytest tests/test_s3_presign_live.py

The test writes one object under ``actant-presign-test/`` and deletes it.
"""

from __future__ import annotations

import os
import time
import urllib.request
import uuid

import pytest

from actant.storage.sigv4 import SigningKeys, presign_get

ENDPOINT = os.environ.get("ACTANT_TEST_S3_ENDPOINT", "")
BUCKET = os.environ.get("ACTANT_TEST_S3_BUCKET", "")
KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
REGION = os.environ.get("AWS_REGION", "us-east-1")

if not (ENDPOINT and BUCKET and KEY_ID and SECRET):
    pytest.skip(
        "needs ACTANT_TEST_S3_ENDPOINT, ACTANT_TEST_S3_BUCKET and AWS credentials",
        allow_module_level=True,
    )


def fetch(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=30) as reply:  # noqa: S310 -- the test's own URL
        return reply.status, reply.read()


def test_our_url_and_botocores_both_download_the_object() -> None:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    key = f"actant-presign-test/{uuid.uuid4().hex}/a b+c ü.png"
    body = b"presign " + uuid.uuid4().bytes
    client.put_object(Bucket=BUCKET, Key=key, Body=body)
    try:
        now = int(time.time())
        ours = presign_get(
            ENDPOINT,
            REGION,
            SigningKeys(KEY_ID, SECRET),
            BUCKET,
            key,
            signed_at=now - now % 3600,
            expires_s=5400,
        )
        theirs = client.generate_presigned_url(
            "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=5400
        )
        assert fetch(ours) == (200, body)
        assert fetch(theirs) == (200, body)
    finally:
        client.delete_object(Bucket=BUCKET, Key=key)
        client.close()
