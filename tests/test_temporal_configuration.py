"""Small tests for configuration values that cross into workflow input.

These checks exist because configuration on the client cannot affect a running
workflow unless it is copied into a serializable workflow payload.
"""

from actant.runtime.executors.temporal_types import ThreadInput
from actant.runtime.executors.temporal_workflows import _continue_as_new_threshold


def test_continue_as_new_threshold_comes_from_thread_input() -> None:
    payload = ThreadInput(
        agent_id="agent",
        thread_id="thread",
        history_size_threshold=321,
    )

    assert _continue_as_new_threshold(payload) == 321


def test_continue_as_new_threshold_has_a_safe_minimum() -> None:
    payload = ThreadInput(
        agent_id="agent",
        thread_id="thread",
        history_size_threshold=0,
    )

    assert _continue_as_new_threshold(payload) == 1
