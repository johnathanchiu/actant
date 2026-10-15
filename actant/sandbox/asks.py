"""Asks: code in a sandbox asking its caller to run what only the caller may.

Some work must not run where the sandbox's code can watch it: a call to an API with the
caller's keys, which any code in the sandbox could read from its environment. Code in the
sandbox asks for it by name, with JSON arguments; the caller runs the handler it registered
under that name, where its keys are, and returns JSON::

    # in the sandbox: a service method, a thread it started, a subprocess of either
    masks = await ask("segment", {"image_b64": ...})

    # the caller: serves asks while its calls run
    runner = SandboxRunner("tools", asks={"segment": segment})

The host relays them. A caller serves asks while each of its calls runs
(``SandboxRunner(asks=...)``, ``RemoteRunner``, ``call_host``) or while a block runs
(:func:`serving`); callers in one process share one poll per host and handlers. Each ask goes
to one caller serving its name; one no caller answers within its ``timeout_s`` fails, so an
ask made with no call running waits for the next.

The sandbox reaches the host at ``ACTANT_ASK_URL`` with ``ACTANT_ASK_TOKEN``, which the host
sets in its own environment once it listens: its services, their worker processes and their
subprocesses inherit them, and scripts never get them (:func:`actant.sandbox.host.script_env`).
The token only asks: it calls no service and takes no ask. Code that reads it can make a
caller run its handlers, and never sees what the handlers hold.

Arguments and answers are JSON: the caller validates what it is given and never unpickles
what the sandbox sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from pydantic import ValidationError

from actant.sandbox.base import Endpoint
from actant.sandbox.protocol import Ask, AskAnswer, Asks, AsksRequest, Header, Route

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

ASK_URL_ENV = "ACTANT_ASK_URL"
ASK_TOKEN_ENV = "ACTANT_ASK_TOKEN"
#: What scripts the services start never inherit.
ASK_ENV = frozenset({ASK_URL_ENV, ASK_TOKEN_ENV})
#: How long an ask waits for its answer unless it says.
ASK_TIMEOUT_S = 900.0
#: Time for the transport on top of an ask's or a poll's own wait.
SLACK_S = 30.0
#: A poll's longest wait for an ask to arrive.
POLL_WAIT_S = 20.0
#: The longest pause between polls that failed.
POLL_BACKOFF_S = 30.0

#: Runs one ask: its ``args``, and the JSON value to answer with.
AskHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class AskError(RuntimeError):
    """An ask that has no answer: its handler failed, or no caller answered in time."""


def asking() -> bool:
    """Whether this process runs under a host that relays asks to its caller."""
    return bool(os.environ.get(ASK_URL_ENV))


def _target() -> tuple[str, dict[str, str]]:
    url = os.environ.get(ASK_URL_ENV)
    if not url:
        raise AskError(f"{ASK_URL_ENV} is not set: this process runs under no service host")
    token = os.environ.get(ASK_TOKEN_ENV, "")
    headers = {
        str(Header.CONTENT_TYPE): "application/json",
        str(Header.AUTHORIZATION): f"Bearer {token}",
    }
    return url.rstrip("/") + Route.ASK, headers


def _answer(name: str, status: int, data: bytes) -> Any:
    if status != HTTPStatus.OK:
        raise AskError(f"ask {name!r}: the host answered HTTP {status}: {data[-1000:]!r}")
    answer = AskAnswer.model_validate_json(data)
    if answer.error is not None:
        raise AskError(answer.error)
    return answer.value


_clients: WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = WeakKeyDictionary()


def _client() -> httpx.AsyncClient:
    """This event loop's one client for asks and polls."""
    import httpx

    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None:
        limits = httpx.Limits(max_connections=None, keepalive_expiry=POLL_WAIT_S)
        client = _clients[loop] = httpx.AsyncClient(trust_env=False, limits=limits)
    return client


async def ask(
    name: str, args: Mapping[str, Any] | None = None, *, timeout_s: float = ASK_TIMEOUT_S
) -> Any:
    """The caller's answer to ``name`` with ``args``, from inside a sandbox. Raises
    :class:`AskError` when its handler failed or no caller answered within ``timeout_s``."""
    request = Ask(name=name, args=dict(args or {}), timeout_s=timeout_s)
    url, headers = _target()
    response = await _client().post(
        url, content=request.model_dump_json(), headers=headers, timeout=timeout_s + SLACK_S
    )
    return _answer(name, response.status_code, response.content)


def ask_sync(
    name: str, args: Mapping[str, Any] | None = None, *, timeout_s: float = ASK_TIMEOUT_S
) -> Any:
    """:func:`ask` for code with no event loop of its own (a thread, a pool's worker)."""
    import httpx

    request = Ask(name=name, args=dict(args or {}), timeout_s=timeout_s)
    url, headers = _target()
    with httpx.Client(trust_env=False) as client:
        response = client.post(
            url, content=request.model_dump_json(), headers=headers, timeout=timeout_s + SLACK_S
        )
    return _answer(name, response.status_code, response.content)


# --- the caller's side ---------------------------------------------------------------------


@dataclass
class _Poll:
    """One poll of a host for some handlers' asks, shared by the blocks serving them."""

    endpoint: Endpoint
    handlers: Mapping[str, AskHandler]
    users: int = 0
    task: asyncio.Task[None] | None = None


_polls: dict[tuple[asyncio.AbstractEventLoop, str, tuple[tuple[str, str], ...], int], _Poll] = {}
#: Polls finishing and handlers answering: each runs to its end after its block has.
_answering: set[asyncio.Task[None]] = set()


@contextlib.asynccontextmanager
async def serving(endpoint: Endpoint, handlers: Mapping[str, AskHandler]) -> AsyncIterator[None]:
    """Answer the asks of the host at ``endpoint`` named in ``handlers`` while the block
    runs. The poll out when it ends is let finish, never cancelled: the host may already
    have handed it asks, which are answered."""
    if not handlers:
        yield
        return
    slot = (
        asyncio.get_running_loop(),
        endpoint.url,
        tuple(sorted(endpoint.headers.items())),
        id(handlers),
    )
    poll = _polls.get(slot)
    if poll is None:
        poll = _polls[slot] = _Poll(endpoint, handlers)
        poll.task = asyncio.create_task(_poll(poll))
        _answering.add(poll.task)
        poll.task.add_done_callback(_answering.discard)
    poll.users += 1
    try:
        yield
    finally:
        poll.users -= 1
        if poll.users == 0:
            del _polls[slot]


async def _poll(poll: _Poll) -> None:
    """Poll while a block serves; the last poll's asks are answered too."""
    import httpx

    url = poll.endpoint.url.rstrip("/")
    headers = {Header.CONTENT_TYPE: "application/json", **poll.endpoint.headers}
    body = AsksRequest(names=sorted(poll.handlers), wait_s=POLL_WAIT_S).model_dump_json()
    failures = 0
    while poll.users:
        try:
            response = await _client().post(
                url + Route.ASKS, content=body, headers=headers, timeout=POLL_WAIT_S + SLACK_S
            )
            if response.status_code != HTTPStatus.OK:
                raise ValueError(f"HTTP {response.status_code}: {response.content[-500:]!r}")
            taken = Asks.model_validate_json(response.content).asks
        except (httpx.HTTPError, OSError, ValueError, ValidationError) as error:
            failures += 1
            if poll.users:  # else the host closed under the last poll: nothing is lost
                logger.warning("polling %s for asks failed (%d): %s", url, failures, error)
            await asyncio.sleep(min(POLL_BACKOFF_S, 2.0**failures))
            continue
        failures = 0
        for one in taken:
            task = asyncio.create_task(_run(url, headers, poll.handlers, one))
            _answering.add(task)
            task.add_done_callback(_answering.discard)


async def _run(
    url: str, headers: Mapping[str, str], handlers: Mapping[str, AskHandler], one: Ask
) -> None:
    import httpx

    handler = handlers.get(one.name)
    try:
        if handler is None:
            raise LookupError(f"this caller serves no ask {one.name!r}")
        answer = AskAnswer(ask_id=one.ask_id, value=await handler(one.args))
        body = answer.model_dump_json()
    except Exception as error:  # noqa: BLE001 -- the asker gets the failure, the poll goes on
        body = AskAnswer(
            ask_id=one.ask_id, error=f"{type(error).__name__}: {error}"
        ).model_dump_json()
    try:
        await _client().post(
            url + Route.ANSWER, content=body, headers=dict(headers), timeout=SLACK_S
        )
    except (httpx.HTTPError, OSError) as error:  # the asker times out instead
        logger.warning("answering ask %s (%s) failed: %s", one.ask_id, one.name, error)
