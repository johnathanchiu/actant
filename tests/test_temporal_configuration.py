"""Small tests for configuration values that cross into workflow input.

These checks exist because configuration on the client cannot affect a running
workflow unless it is copied into a serializable workflow payload.
"""

from actant.runtime.temporal.types import ThreadInput
from actant.runtime.temporal.workflow import _history_rotation_threshold


def test_history_rotation_threshold_comes_from_thread_input() -> None:
    payload = ThreadInput(
        agent_id="agent",
        thread_id="thread",
        history_size_threshold=321,
    )

    assert _history_rotation_threshold(payload) == 321


def test_history_rotation_threshold_has_a_safe_minimum() -> None:
    payload = ThreadInput(
        agent_id="agent",
        thread_id="thread",
        history_size_threshold=0,
    )

    assert _history_rotation_threshold(payload) == 1
