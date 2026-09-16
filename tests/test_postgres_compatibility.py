"""Existing Postgres schema preserves legacy and reference-based transcripts.

Opt in with ACTANT_TEST_POSTGRES_URL pointing to the disposable local
actant_core_test database. Each test uses and removes its own schema.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from actant.assets import AssetReference
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime.stores.postgres import ACTANT_RUNTIME_METADATA, SQLAlchemyRuntimeStores
from actant.runtime.types.threads import RunStatus
from actant.tools.calls import ToolCallRecord, ToolCallStatus


@pytest.fixture
def postgres_schema() -> str:
    return "test_" + uuid4().hex


@pytest.fixture
async def stores(postgres_schema: str) -> AsyncIterator[SQLAlchemyRuntimeStores]:
    raw = os.environ.get("ACTANT_TEST_POSTGRES_URL")
    if not raw:
        pytest.skip("requires disposable ACTANT_TEST_POSTGRES_URL")
    url = make_url(raw)
    if url.host not in {"localhost", "127.0.0.1"} or url.database != "actant_core_test":
        raise ValueError("test requires local actant_core_test database")
    schema = postgres_schema
    engine = create_async_engine(url, execution_options={"schema_translate_map": {None: schema}})
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.run_sync(ACTANT_RUNTIME_METADATA.create_all)
        yield SQLAlchemyRuntimeStores(async_sessionmaker(engine, expire_on_commit=False))
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


async def test_legacy_and_asset_blocks_usage_and_tool_results_round_trip(
    stores: SQLAlchemyRuntimeStores,
) -> None:
    old = {"type": "image", "source": {"type": "url", "url": "https://old", "expires_at": 1}}
    inline = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"},
    }
    asset = AssetReference("images/a.png", "image/png").to_block()
    await stores.threads.get_or_create("a", "t")
    await stores.runs.create("a", "t", run_id="r", max_turns=3)
    await stores.messages.append_user("a", "t", [old, inline, asset])
    call = ToolCallRecord(
        id="c",
        group_id="g",
        run_id="r",
        agent_id="a",
        thread_id="t",
        turn_id="turn",
        turn_index=1,
        name="look",
        args={},
    )
    assistant = Message(
        role="assistant",
        content="looking",
        input_tokens=13,
        output_tokens=7,
        tool_calls=[ToolCall(id="c", function=ToolCallFunction(name="look", arguments="{}"))],
    )
    await stores.messages.append_assistant_with_tool_calls("a", "t", "turn", assistant, [call])
    result = {"result": "picture", "content_blocks": [asset]}
    for _ in range(2):
        await stores.messages.append_tool_result("a", "t", "turn", "c", "look", result)
        await stores.runs.finish("r", RunStatus.IDLE)
    messages = await stores.messages.list_for_thread("a", "t")
    assert len(messages) == 3
    assert messages[0].content == [old, inline, asset]
    assert messages[1].input_tokens == 13 and messages[1].output_tokens == 7
    assert messages[2].content == [asset]
    assert (await stores.tool_calls.get("c")).turn_id == "turn"


async def test_claim_race_and_stale_update_preserve_single_sandbox(
    stores: SQLAlchemyRuntimeStores,
) -> None:
    stale = await stores.threads.get_or_create("a", "t")
    winners = await asyncio.gather(
        *[
            stores.threads.claim_sandbox("a", "t", expected=None, sandbox_id=f"sandbox-{i}")
            for i in range(12)
        ]
    )
    assert len(set(winners)) == 1
    await stores.threads.update(stale)
    assert (await stores.threads.get("a", "t")).sandbox_id == winners[0]


async def test_killed_worker_repairs_one_result_without_repeating_side_effect(
    stores: SQLAlchemyRuntimeStores, postgres_schema: str, tmp_path: Path
) -> None:
    """Kill after the effect but before result persistence; recover on a fresh process."""
    from temporalio.testing import WorkflowEnvironment

    from actant.runtime import AgentRuntime, TemporalRuntimeConfig

    effects = tmp_path / "effects.txt"
    log_path = tmp_path / "worker.log"
    helper = Path(__file__).with_name("crash_worker.py")
    processes: list[asyncio.subprocess.Process] = []
    async with await WorkflowEnvironment.start_local() as env:
        config = TemporalRuntimeConfig(task_queue=uuid4().hex)
        runtime = AgentRuntime(client=env.client, stores=stores, config=config)
        with log_path.open("wb") as log:

            async def launch() -> asyncio.subprocess.Process:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(helper),
                    env.client.service_client.config.target_host,
                    config.task_queue,
                    postgres_schema,
                    str(effects),
                    stdout=log,
                    stderr=log,
                )
                processes.append(process)
                return process

            try:
                first = await launch()
                workflow_id = await runtime.send_message("a", "t", "go")
                async with asyncio.timeout(40):
                    while not effects.exists() or not effects.read_text():
                        assert first.returncode is None, log_path.read_text()
                        await asyncio.sleep(0.05)
                first.kill()  # SIGKILL: no activity exception handler or graceful shutdown.
                await first.wait()
                await launch()
                await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 40)
                [run] = await stores.runs.list_for_thread("a", "t")
                assert run.status is RunStatus.FAILED
                assert (await stores.tool_calls.get("effect")).status is ToolCallStatus.FAILED
                assert not await stores.tool_calls.get_open_for_thread("a", "t")
                messages = await runtime.thread("a", "t").messages()
                results = [m for m in messages if m.role == "tool"]
                assert len(results) == 1 and results[0].tool_call_id == "effect"
                assert effects.read_text().splitlines() == ["effect"]
            finally:
                for process in processes:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
