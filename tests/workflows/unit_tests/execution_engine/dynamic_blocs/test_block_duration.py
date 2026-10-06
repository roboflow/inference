"""Pins the workflows-local block-duration channel to the server's original.

`inference/usage_tracking/block_execution.py` is the host's channel; workflows
now owns a copy so the engine can publish a measurement without importing the
server. The two must agree exactly on which values are usable and on the four
source names, because the host relays one into the other - a divergence would
silently change what a block is billed for.
"""

import pytest

from inference.core.workflows.execution_engine.v1.dynamic_blocks import block_duration
from inference.usage_tracking import block_execution as server_channel

SOURCE_NAMES = [
    "BLOCK_DURATION_SOURCE_REMOTE_RUNTIME",
    "BLOCK_DURATION_SOURCE_LOCAL_RUNTIME",
    "BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK",
    "BLOCK_DURATION_SOURCE_UNAVAILABLE",
]

# Values the host must drop, and values it must keep. `True` is in the rejected
# list on purpose: `bool` is an `int`, and billing one second for a flag would
# be silent corruption.
UNUSABLE = [None, True, False, "0.25", -1.0, float("nan"), float("inf"), object()]
USABLE = [0, 0.0, 1, 0.25, 1e-6, 3600.0]


@pytest.fixture(autouse=True)
def _cleared():
    block_duration.clear_block_duration()
    server_channel.clear_measured_block_execution()
    yield
    block_duration.clear_block_duration()
    server_channel.clear_measured_block_execution()


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_source_names_match_the_server_channel(name: str) -> None:
    assert getattr(block_duration, name) == getattr(server_channel, name)


@pytest.mark.parametrize("value", UNUSABLE)
def test_unusable_values_are_dropped_by_both_channels(value) -> None:
    # given
    block_duration.record_block_duration(
        duration=value, source=block_duration.BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    )
    server_channel.record_measured_block_execution(
        duration=value, source=server_channel.BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    )

    # then
    assert block_duration.peek_block_duration() is None
    assert server_channel.peek_measured_block_execution() is None


@pytest.mark.parametrize("value", USABLE)
def test_usable_values_are_kept_identically_by_both_channels(value) -> None:
    # given
    block_duration.record_block_duration(
        duration=value, source=block_duration.BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    )
    server_channel.record_measured_block_execution(
        duration=value, source=server_channel.BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    )

    # then
    local = block_duration.peek_block_duration()
    remote = server_channel.peek_measured_block_execution()
    assert local is not None and remote is not None
    assert local.duration == remote.duration == float(value)
    assert local.source == remote.source


def test_consume_clears_so_a_later_invocation_is_not_billed_twice() -> None:
    # given
    block_duration.record_block_duration(
        duration=0.25, source=block_duration.BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    )

    # when
    first = block_duration.consume_block_duration()
    second = block_duration.consume_block_duration()

    # then
    assert first is not None and first.duration == 0.25
    assert second is None


def test_clear_removes_a_pending_measurement() -> None:
    block_duration.record_block_duration(
        duration=0.5, source=block_duration.BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
    )
    block_duration.clear_block_duration()
    assert block_duration.peek_block_duration() is None
