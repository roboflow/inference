"""Measured execution duration for a single workflow block invocation.

``usage_collector("workflow_block")`` times the call it decorates, but that wall
clock is not always the number that should be billed. A custom Python block
running in a Modal sandbox spends part of the decorated call serializing inputs
and waiting on the network, which is not time the block itself ran. Blocks
publish the duration they want attributed here and the collector prefers it
over its own measurement.

The duration is published through a :class:`~contextvars.ContextVar` rather than
an attribute on the block. Block instances are shared across the server's worker
threads, so an attribute would let one step overwrite the duration of another
step that is still running.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Any, NamedTuple, Optional

# Where the recorded duration came from, reported alongside it so the usage API
# can tell a measured runtime from a fallback estimate.
BLOCK_DURATION_SOURCE_REMOTE_RUNTIME = "remote_runtime"
BLOCK_DURATION_SOURCE_LOCAL_RUNTIME = "local_runtime"
BLOCK_DURATION_SOURCE_CLIENT_WALL_CLOCK = "client_wall_clock"
BLOCK_DURATION_SOURCE_DECORATOR_WALL_CLOCK = "decorator_wall_clock"
BLOCK_DURATION_SOURCE_UNAVAILABLE = "unavailable"

BLOCK_EXECUTION_MODE_LOCAL = "local"
BLOCK_EXECUTION_MODE_REMOTE = "modal"


class MeasuredBlockExecution(NamedTuple):
    duration: float
    source: str
    execution_mode: Optional[str] = None


_measured_block_execution: ContextVar[Optional[MeasuredBlockExecution]] = ContextVar(
    "usage_measured_block_execution",
    default=None,
)


def _as_duration(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def clear_measured_block_execution() -> None:
    _measured_block_execution.set(None)


def record_measured_block_execution(
    *,
    duration: Any,
    source: str,
    execution_mode: Optional[str] = None,
) -> None:
    """Publish the duration to bill for the block invocation now running.

    Values that cannot be summed (non-numeric, negative, NaN, infinity) are
    dropped so a misreporting executor degrades to the decorator's own wall
    clock instead of corrupting the usage row.
    """
    measured_duration = _as_duration(duration)
    if measured_duration is None:
        return
    _measured_block_execution.set(
        MeasuredBlockExecution(
            duration=measured_duration,
            source=source,
            execution_mode=execution_mode,
        )
    )


def consume_measured_block_execution() -> Optional[MeasuredBlockExecution]:
    """Read and clear the duration published by the current invocation.

    Clearing on read keeps a stale duration from leaking into a later
    invocation that did not publish one, which would bill it twice.
    """
    measured = _measured_block_execution.get()
    if measured is None:
        return None
    _measured_block_execution.set(None)
    return measured
