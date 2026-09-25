"""Presigned S3 GET URLs signed at a time the caller chooses, by the AWS Common Runtime.

botocore always signs at the current time and has no way to set it (boto/botocore#2647).
``awscrt``, AWS's own signing library (the one botocore itself uses when installed), takes the
signing date explicitly, so every process signing the same key at the same time gets the same
URL. The algorithm stays AWS's; this module only builds the request and reads the URL back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import quote, urlsplit

from awscrt.auth import (
    AwsCredentialsProvider,
    AwsSignatureType,
    AwsSignedBodyValue,
    AwsSigningAlgorithm,
    AwsSigningConfig,
    aws_sign_request,
)
from awscrt.http import HttpHeaders, HttpRequest

SERVICE = "s3"
#: An object path is URI-encoded once, keeping its `/`, as S3 expects.
PATH_SAFE = "/~"
#: S3 rejects a presigned URL whose X-Amz-Expires exceeds seven days.
MAX_EXPIRES_S = 604800
DEFAULT_PORTS = {"https": 443, "http": 80}

AddressingStyle = Literal["path", "virtual"]


@dataclass(frozen=True)
class SigningKeys:
    """Static credentials (R2 access keys). With a ``session_token`` every URL changes
    whenever the token rotates."""

    access_key_id: str
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)


def check_endpoint(endpoint_url: str) -> None:
    endpoint = urlsplit(endpoint_url)
    if (
        endpoint.scheme not in DEFAULT_PORTS
        or not endpoint.hostname
        or endpoint.path.strip("/")
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("endpoint_url must be a bare http(s) origin")


def presign_get(
    endpoint_url: str,
    region: str,
    keys: SigningKeys,
    bucket: str,
    key: str,
    *,
    signed_at: int,
    expires_s: int,
    addressing_style: AddressingStyle = "path",
) -> str:
    """A presigned GET URL for ``bucket/key``, signed at ``signed_at`` (Unix seconds)."""
    check_endpoint(endpoint_url)
    if not 0 < expires_s <= MAX_EXPIRES_S:
        raise ValueError(f"expires_s must be between 1 and {MAX_EXPIRES_S} seconds")
    endpoint = urlsplit(endpoint_url)
    # The Host header omits a default port; the URL keeps the endpoint as given.
    host = endpoint.hostname or ""
    if endpoint.port is not None and endpoint.port != DEFAULT_PORTS[endpoint.scheme]:
        host = f"{host}:{endpoint.port}"
    netloc, path = endpoint.netloc, "/" + quote(key, safe=PATH_SAFE)
    if addressing_style == "virtual":
        netloc, host = f"{bucket}.{netloc}", f"{bucket}.{host}"
    else:
        path = f"/{bucket}{path}"

    config = AwsSigningConfig(
        algorithm=AwsSigningAlgorithm.V4,
        signature_type=AwsSignatureType.HTTP_REQUEST_QUERY_PARAMS,
        credentials_provider=AwsCredentialsProvider.new_static(
            keys.access_key_id, keys.secret_access_key, keys.session_token
        ),
        region=region,
        service=SERVICE,
        date=datetime.fromtimestamp(signed_at, timezone.utc),
        expiration_in_seconds=expires_s,
        signed_body_value=AwsSignedBodyValue.UNSIGNED_PAYLOAD,
        # S3 signs the path as sent: encoded once, never normalized.
        use_double_uri_encode=False,
        should_normalize_uri_path=False,
    )
    request = HttpRequest("GET", path, HttpHeaders([("host", host)]))
    signed = aws_sign_request(request, config).result()
    return f"{endpoint.scheme}://{netloc}{signed.path}"
