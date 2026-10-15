"""Code in a sandbox asks its caller to run what only the caller may, and gets the answer."""

from __future__ import annotations

import json
from http import HTTPStatus
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from actant.sandbox import AskHandler, LocalSandbox, LocalSandboxProvider, SandboxSpec
from actant.sandbox import SandboxRunner, host
from actant.sandbox.protocol import CallResponse, Route

TESTS = str(Path(__file__).parent)


async def double(args: dict[str, Any]) -> dict[str, int]:
    return {"twice": 2 * int(args["n"])}


async def broken(args: dict[str, Any]) -> None:
    raise ValueError(f"no {args['n']}")


ASKS: dict[str, AskHandler] = {"double": double, "broken": broken}


@pytest.fixture
async def sandbox(tmp_path: Path) -> AsyncIterator[LocalSandbox]:
    spec = SandboxSpec(services={"asker": "service_fixtures:Asker"}, env={"PYTHONPATH": TESTS})
    opened = await LocalSandboxProvider(tmp_path).open(spec, sandbox_id="asks")
    assert isinstance(opened, LocalSandbox)
    yield opened
    await opened.close()


async def _call(
    sandbox: LocalSandbox, method: str, asks: dict[str, AskHandler] | None, **args: Any
) -> CallResponse:
    async with SandboxRunner("asker", asks=asks) as runner:
        return await runner.call(method, args, key="k", sandbox=sandbox)


@pytest.mark.parametrize("method", ["in_a_thread", "in_a_subprocess"])
async def test_the_caller_answers_from_wherever_the_sandbox_asks(
    sandbox: LocalSandbox, method: str
) -> None:
    here = await _call(sandbox, "here", ASKS, name="double", n=4)
    assert json.loads(here.text) == {"twice": 8}, here
    there = await _call(sandbox, method, ASKS, n=5)
    assert json.loads(there.text) == {"twice": 10}, there


async def test_a_failed_or_unserved_ask_fails_the_asker(sandbox: LocalSandbox) -> None:
    failed = await _call(sandbox, "here", ASKS, name="broken", n=1)
    assert failed.error and "ValueError: no 1" in failed.error
    unserved = await _call(sandbox, "here", ASKS, name="triple", n=1)
    assert unserved.error and "no caller took it" in unserved.error


async def test_scripts_never_get_the_asks_address(sandbox: LocalSandbox) -> None:
    seen = await _call(sandbox, "scripts_see", None)
    assert json.loads(seen.text) == []


def test_the_ask_token_asks_and_does_nothing_else() -> None:
    served = host.Host({}, token="host")
    asking = {"authorization": f"Bearer {served.ask_token}"}
    assert served.reject("POST", Route.ASK, asking) is None
    for route in (Route.CALL, Route.ASKS, Route.ANSWER, Route.CANCEL, Route.SHUTDOWN):
        refused = served.reject("POST", route, asking)
        assert refused is not None and refused[0] == HTTPStatus.UNAUTHORIZED
