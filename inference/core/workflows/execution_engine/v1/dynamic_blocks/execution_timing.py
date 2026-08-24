"""Runtime a remote executor reported for one dynamic block invocation.

The Modal sandbox measures how long the user's ``run`` function actually ran and
returns that with its response. The executor publishes the number here so
``block_scaffolding`` can attribute the block's execution duration to the remote
runtime rather than to the client's wall clock, which also covers input
serialization and the network round trip.

A :class:`~contextvars.ContextVar` carries it because executors are pooled and
shared between the server's worker threads, so an attribute on the executor
would let one step overwrite the timing of another step that is still running.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Any, Optional

_remote_execution_duration: ContextVar[Optional[float]] = ContextVar(
    "dynamic_block_remote_execution_duration",
    default=None,
)


def clear_remote_execution_duration() -> None:
    _remote_execution_duration.set(None)


def record_remote_execution_duration(duration: Any) -> None:
    """Publish the runtime a remote executor reported for the current call.

    Anything that is not a usable duration is ignored, so a sandbox that
    predates this field (or misreports it) leaves the caller with no
    measurement rather than a bogus one.
    """
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return
    value = float(duration)
    if not math.isfinite(value) or value < 0:
        return
    _remote_execution_duration.set(value)


def consume_remote_execution_duration() -> Optional[float]:
    """Read and clear the runtime published by the current call."""
    duration = _remote_execution_duration.get()
    if duration is None:
        return None
    _remote_execution_duration.set(None)
    return duration
