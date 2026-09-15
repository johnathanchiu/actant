"""Services: a plain class whose public methods are callable remotely.

A product writes an ordinary class, with no actant import::

    class Notes:
        @classmethod
        async def open(cls, folder: str) -> "Notes": ...   # optional; else Notes(**init)

        async def add(self, title: str, body: str = "") -> str:
            \"\"\"Save a note.\"\"\"

Its methods are the public functions (``async def`` or plain ``def``) on the class
and its bases, minus ``open`` and ``close`` (:func:`actant.sandbox.host.service_methods`).
A plain ``def`` runs in a worker thread; an ``async def`` runs on the host's event
loop, so blocking or CPU-heavy work in one stalls every concurrent call until it
awaits. Arguments are validated against the method's signature without ``self``
(:func:`actant.sandbox.host.parameters_model`), so methods receive the annotated types.

A method returns ``str``, an object with ``.text`` and ``.images`` (file paths or
bytes), or any JSON-encodable value; every runner encodes it as the same
:class:`~actant.sandbox.protocol.CallResponse`.

A :class:`Runner` says where the methods run: :class:`LocalRunner` in-process,
:class:`RemoteRunner` against a host endpoint, or :class:`SandboxRunner` against a
sandbox's host (a name in the spec's ``services``). Orchestration code calls a
runner directly; :func:`actant.tools.tools` exposes a service to a model through one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from http import HTTPStatus
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError

if TYPE_CHECKING:
    import httpx

import actant.sandbox.host as host
from actant.sandbox.base import Endpoint, Sandbox
from actant.sandbox.protocol import CallRequest, CallResponse, Route

DEFAULT_CALL_TIMEOUT_S = 600.0


class Runner(Protocol):
    """Runs one service method. ``needs_sandbox`` says :meth:`call` needs ``sandbox``;
    ``key`` names the host-side instance where the runner does not fix one."""

    needs_sandbox: bool

    async def call(
        self,
        method: str,
        args: Mapping[str, object],
        *,
        key: str | None = None,
        sandbox: Sandbox | None = None,
    ) -> CallResponse: ...


class LocalRunner:
    """Runs methods on an instance in this process."""

    needs_sandbox = False

    def __init__(self, instance: object) -> None:
        self.instance = instance

    async def call(
        self,
        method: str,
        args: Mapping[str, object],
        *,
        key: str | None = None,
        sandbox: Sandbox | None = None,
    ) -> CallResponse:
        del key, sandbox
        return await host.call_method(self.instance, method, args)


class RemoteRunner:
    """Runs methods of the host's ``service``. ``key`` names the host-side instance,
    created from ``init`` on its first call."""

    needs_sandbox = False

    def __init__(
        self,
        endpoint: Endpoint,
        service: str,
        key: str,
        init: Mapping[str, object] | None = None,
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        import httpx

        self.endpoint = endpoint
        self.service = service
        self.key = key
        self.init = dict(init or {})
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            trust_env=False, limits=httpx.Limits(max_connections=None)
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> RemoteRunner:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def call(
        self,
        method: str,
        args: Mapping[str, object],
        *,
        key: str | None = None,
        sandbox: Sandbox | None = None,
    ) -> CallResponse:
        del key, sandbox
        body = _body(self.service, self.key, self.init, method, args)
        return (await _send(self.endpoint, body, self.timeout, self._client))[1]


class SandboxRunner:
    """Runs methods of ``service`` (a name in ``SandboxSpec.services``) on ``sandbox``'s
    host, one instance per ``key`` (a tool passes its thread id), created from ``init``.
    A connect credential the host rejects is refreshed once through
    :meth:`Sandbox.endpoint`."""

    needs_sandbox = True

    def __init__(
        self,
        service: str,
        init: Mapping[str, object] | None = None,
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        import httpx

        self.service = service
        self.init = dict(init or {})
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            trust_env=False, limits=httpx.Limits(max_connections=None)
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> SandboxRunner:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def call(
        self,
        method: str,
        args: Mapping[str, object],
        *,
        key: str | None = None,
        sandbox: Sandbox | None = None,
    ) -> CallResponse:
        endpoint = await sandbox.endpoint() if sandbox is not None else None
        if sandbox is None or endpoint is None:
            return CallResponse(error="this sandbox serves no services (SandboxSpec.services)")
        if key is None:
            return CallResponse(error="SandboxRunner.call needs a key")
        body = _body(self.service, key, self.init, method, args)
        status, response = await _send(endpoint, body, self.timeout, self._client)
        if status == HTTPStatus.UNAUTHORIZED:  # rejected before running: safe to retry
            endpoint = await sandbox.endpoint(refresh=True)
            assert endpoint is not None
            _, response = await _send(endpoint, body, self.timeout, self._client)
        return response


async def call_host(
    endpoint: Endpoint,
    service: str,
    method: str,
    args: Mapping[str, object],
    *,
    key: str,
    init: Mapping[str, object] | None = None,
    timeout: float = DEFAULT_CALL_TIMEOUT_S,
) -> CallResponse:
    """``POST /v1/call`` to a service host. Transport failures become error responses."""
    body = _body(service, key, init or {}, method, args)
    return (await _send(endpoint, body, timeout))[1]


def _body(
    service: str, key: str, init: Mapping[str, object], method: str, args: Mapping[str, object]
) -> bytes:
    request = CallRequest(
        service=service, key=key, init=dict(init), method=method, args=dict(args)
    )
    return request.model_dump_json().encode()


async def _send(
    endpoint: Endpoint, body: bytes, timeout: float, client: httpx.AsyncClient | None = None
) -> tuple[int | None, CallResponse]:
    import httpx

    if client is None:
        async with httpx.AsyncClient(trust_env=False) as owned:
            return await _send(endpoint, body, timeout, owned)
    try:
        async with asyncio.timeout(timeout):
            response = await client.post(
                endpoint.url.rstrip("/") + Route.CALL,
                content=body,
                headers={"Content-Type": "application/json", **endpoint.headers},
                timeout=timeout,
            )
        status, data = response.status_code, response.content
    except (OSError, httpx.HTTPError) as error:
        return None, CallResponse(
            error=f"service host call to {endpoint.url} failed and may have run: {error}"
        )
    try:
        return status, CallResponse.model_validate_json(data)
    except ValidationError:
        tail = data[-1000:].decode(errors="replace")
        return status, CallResponse(error=f"service host returned HTTP {status}: {tail}")
