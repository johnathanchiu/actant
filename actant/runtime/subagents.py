"""A finished sub-thread reports to its parent.

A thread started with ``parent_thread_id`` messages its parent at the end of every run,
on the inbox a person's message arrives on: that is what wakes a parent whether it is
parked or already closed. Both runtimes build the message here and differ only in how it
is delivered, so what a parent hears never depends on which one ran the child.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from actant.blocks import BLOCKS, TextBlock
from actant.llm.messages import Message
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.temporal.types import InboundMessage, ReportToParentInput, RunOutcome
from actant.runtime.types.threads import AgentThread, ThreadStatus
from actant.tools.base import MetadataKey

__all__ = ["SUBAGENT_REPORT_TAG", "SubagentReport", "parent_report"]

#: The tag on a report's message, in the parent's inbox and transcript.
SUBAGENT_REPORT_TAG = "subagent_report"


@dataclass(frozen=True)
class SubagentReport:
    """How one run of a sub-thread ended. ``status`` is a ``RunOutcome`` value; ``report``
    is its final answer when it completed, else why it stopped."""

    agent_id: str
    thread_id: str
    run_id: str
    status: str
    report: str

    def text(self) -> str:
        return (
            f'<subagent_report thread_id="{self.thread_id}" agent_id="{self.agent_id}" '
            f'run_id="{self.run_id}" status="{self.status}">\n'
            f"{self.report}\n"
            "</subagent_report>"
        )

    def inbound(self) -> InboundMessage:
        # The run id makes the message one delivery: the parent workflow drops a second.
        return InboundMessage(
            content=BLOCKS.dump_python([TextBlock(text=self.text())], mode="json"),
            source="subagent",
            correlation_id=self.run_id,
            tag=SUBAGENT_REPORT_TAG,
        )


async def parent_report(
    stores: RuntimeStores, payload: ReportToParentInput
) -> tuple[AgentThread, InboundMessage] | None:
    """The parent to message and what to say, or ``None``: the thread has no parent, the
    parent is gone or cancelled (a cancel that reached the child through it), or the
    parent's transcript already holds this run's report (a delivery attempted again)."""
    child = await stores.threads.get(payload.agent_id, payload.thread_id)
    if child.parent_thread_id is None:
        return None
    parent = await stores.threads.find(child.parent_thread_id)
    if parent is None or parent.status is ThreadStatus.CANCELLED:
        return None
    delivered = f'run_id="{payload.run_id}"'
    for message in await stores.messages.list_for_thread(parent.agent_id, parent.id):
        if message.tag == SUBAGENT_REPORT_TAG and delivered in _text(message):
            return None
    report = SubagentReport(
        agent_id=payload.agent_id,
        thread_id=payload.thread_id,
        run_id=payload.run_id,
        status=payload.outcome,
        report=await _final_text(stores, payload),
    )
    return parent, report.inbound()


async def _final_text(stores: RuntimeStores, payload: ReportToParentInput) -> str:
    """A completed run's answer: its terminal tool's (``finish``'s summary), else the last
    thing the agent said, which a run that completed by replying ended on."""
    if payload.outcome != RunOutcome.COMPLETED.value:
        return payload.stop_reason or ""
    records = await stores.tool_calls.get_by_run(payload.run_id)
    for record in sorted(records, key=lambda r: (r.turn_index, r.id), reverse=True):
        result = record.result if isinstance(record.result, dict) else {}
        metadata = result.get("metadata")
        if not isinstance(metadata, dict) or not metadata.get(MetadataKey.TERMINAL):
            continue
        output = result.get("result")
        if isinstance(output, dict) and isinstance(output.get("summary"), str):
            return output["summary"]
        return output if isinstance(output, str) else json.dumps(output)
    messages = await stores.messages.list_for_thread(payload.agent_id, payload.thread_id)
    return next((_text(m) for m in reversed(messages) if m.role == "assistant"), "")


def _text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(b.text for b in message.content or [] if isinstance(b, TextBlock))
