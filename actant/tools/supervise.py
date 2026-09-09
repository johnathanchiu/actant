"""Working with a subagent that is already running.

``TaskTool`` starts a subagent and hands back its thread id. These are what
the parent does with that id afterwards: look at how it is going, say
something else to it, or stop it.

They exist because delegation stopped being a blocking call. A parent that
waited on its child needed none of this -- it had exactly one child and no
say in it once it started. A parent holding handles can run several at once,
answer a question one of them raises, and abandon one that has gone wrong.

The library defines the shape and nothing else. A host owns what "status"
means for its threads and how a message reaches one, so it implements
``SubagentSupervisor`` and wires these in beside ``TaskTool``.

Note what is deliberately missing: nothing here waits. A parent does not need
to poll for the ending, because a finished sub-thread messages its parent,
which wakes it whether it is parked or already closed. ``check`` is for a
parent that wants to look before then.
"""

from __future__ import annotations

from typing import Protocol

from actant.core import JSONObject
# Unused by name, but FunctionTool resolves these annotations at build time
# and JSONObject expands to a form naming JSONValue, which get_type_hints
# looks up in this module's globals.
from actant.core import JSONValue as JSONValue  # noqa: F401
from actant.tools.base import Tool
from actant.tools.function import FunctionTool

__all__ = ["SubagentSupervisor", "supervision_tools"]


class SubagentSupervisor(Protocol):
    """What a host must provide for a parent to supervise its children."""

    async def status(self, thread_id: str) -> JSONObject:
        """How the sub-thread is going. Free-form, and read by the model."""
        ...

    async def send(self, thread_id: str, message: str) -> None:
        """Say something else to a running sub-thread."""
        ...

    async def stop(self, thread_id: str) -> None:
        """Cancel a sub-thread. Must be safe to call on one already finished."""
        ...


def supervision_tools(supervisor: SubagentSupervisor) -> list[Tool]:
    """The three tools, bound to one host's supervisor."""

    async def check_subagent(thread_id: str) -> JSONObject:
        """Check how a delegated subagent is going.

        Args:
            thread_id: The id returned when the subagent was started.
        """
        return await supervisor.status(thread_id)

    async def message_subagent(thread_id: str, message: str) -> JSONObject:
        """Send a further instruction to a running subagent.

        Args:
            thread_id: The id returned when the subagent was started.
            message: What to tell it.
        """
        await supervisor.send(thread_id, message)
        return {"thread_id": thread_id, "delivered": True}

    async def stop_subagent(thread_id: str) -> JSONObject:
        """Stop a subagent whose work is no longer wanted.

        Args:
            thread_id: The id returned when the subagent was started.
        """
        await supervisor.stop(thread_id)
        return {"thread_id": thread_id, "stopped": True}

    return [
        FunctionTool(check_subagent),
        FunctionTool(message_subagent),
        FunctionTool(stop_subagent),
    ]
