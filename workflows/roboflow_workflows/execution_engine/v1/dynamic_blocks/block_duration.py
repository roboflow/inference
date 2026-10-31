"""Measured execution duration for a single workflow block invocation.

Whoever actually measured the block - the in-process scaffolding, or the Modal
sandbox that reported its own runtime - publishes the duration here. A host
that bills block execution reads it through `ExecutionObserver` and prefers it
over its own wall clock, which for a remote block also covers input
serialization and the network round trip.

The duration travels through a `ContextVar` rather than an attribute on the
block: block instances are shared across the server's worker threads, so an
attribute would let one step overwrite the duration of another step that is
still running.

Values that cannot be summed are dropped rather than stored, so a misreporting
executor degrades to the host's own measurement instead of corrupting a usage
row. `bool` is rejected explicitly - it is an `int`, and a flag billed as a
second is silent corruption. This validation and the four source names are a
verbatim copy of the host channel they are relayed into, and are pinned to it
by `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_block_duration.py`.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Any, NamedTuple, Optional

# Where the recorded duration came from, reported alongside it so a host can
# tell a measured runtime from a fallback estimate.
BLOCK_DURATION_SOURCE_REMOTE_RUNTIME = "remote_runtime"
BLOCK_DURATION_SOURCE_LOCAL_RUNTIME = "local_runtime"
BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK = "client_wall_clock"
BLOCK_DURATION_SOURCE_UNAVAILABLE = "unavailable"


class MeasuredBlockDuration(NamedTuple):
    duration: float
    source: str


_measured_block_duration: ContextVar[Optional[MeasuredBlockDuration]] = ContextVar(
    "workflows_measured_block_duration",
    default=None,
)


def _as_duration(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def clear_block_duration() -> None:
    _measured_block_duration.set(None)


def record_block_duration(*, duration: Any, source: str) -> None:
    """Publish the duration to bill for the block invocation now running."""
    measured_duration = _as_duration(duration)
    if measured_duration is None:
        return
    _measured_block_duration.set(
        MeasuredBlockDuration(duration=measured_duration, source=source)
    )


def peek_block_duration() -> Optional[MeasuredBlockDuration]:
    """Whether this invocation already published a duration, without taking it.

    Lets the remote arm decide it has nothing to add when the sandbox reported
    its own runtime, while leaving the value for the host to consume.
    """
    return _measured_block_duration.get()


def consume_block_duration() -> Optional[MeasuredBlockDuration]:
    """Read and clear the duration published by the current invocation.

    Clearing on read keeps a stale duration from leaking into a later
    invocation that did not publish one, which would bill it twice.
    """
    measured = _measured_block_duration.get()
    if measured is None:
        return None
    _measured_block_duration.set(None)
    return measured
