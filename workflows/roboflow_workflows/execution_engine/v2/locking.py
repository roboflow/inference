"""Bounded waiting for the locks a graph update holds.

A graph update of a session holds, in this order, the session's use guard,
the active run registry, the update candidate, the control panel and the
active run's lock. An update of a running run has a deadline: every one of
those waits gives up at the same monotonic time, so a lock held by someone
else (a control write whose custom kind codec is slow, for example) cannot
hold the update past its timeout. An idle update passes no deadline and
waits.
"""

import contextlib
import threading
import time
from typing import Iterator, Optional, Union

from roboflow_workflows.execution_engine.v2.errors import UpdateTimeoutError

__all__ = ["acquired"]


@contextlib.contextmanager
def acquired(
    lock: Union[threading.Lock, threading.RLock],
    *,
    deadline: Optional[float],
    what: str,
) -> Iterator[None]:
    """Hold ``lock`` for the block; give up when ``deadline`` passes first.

    Args:
        lock: The lock to hold.
        deadline: ``time.monotonic()`` value at which waiting stops; ``None``
            waits until the lock is free.
        what: What the lock guards, named in the timeout error.

    Raises:
        UpdateTimeoutError: When the lock was not free before ``deadline``.
    """
    if deadline is None:
        with lock:
            yield
        return

    if not lock.acquire(timeout=max(deadline - time.monotonic(), 0.0)):
        raise UpdateTimeoutError(
            f"the graph update timed out waiting for {what}; the session keeps "
            "its graph and the candidate stays prepared"
        )
    try:
        yield
    finally:
        lock.release()
