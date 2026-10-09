"""Fixed pool of pipeline workers, and threads a run owns.

The pool never queues work. ``try_submit`` hands work to an idle worker or
refuses it, so accepted work always starts at once and the executing count
never exceeds the pool size::

    pool = WorkerPool(4, name="run-1a2b", owner=owned, on_free=wake_dispatcher)
    pool.try_submit(work)        # True: a worker runs it now; False: all busy
    pool.submit(work, timeout=1) # waits for a free worker, or False
    pool.wait_free(timeout=1)    # True once a worker is idle (single submitter)
    pool.wait_idle()             # quiescence: no work running
    pool.shutdown()              # refuse new work, finish running work, join

Work must handle its own errors; the pool keeps the first unexpected one in
``unexpected`` and the worker continues.

``OwnedThreads`` marks the threads of one run (readers, dispatcher,
workers), so waiting for that run from one of its own threads raises
instead of deadlocking.
"""

import threading
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional

from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.pipelining.stages import PipelineCounters

__all__ = ["OwnedThreads", "WorkerPool"]

Work = Callable[[], None]


class OwnedThreads:
    """Threads owned by one run; a wait from one of them would never end."""

    def __init__(self) -> None:
        self._local = threading.local()

    def mark(self) -> None:
        """Mark the calling thread as owned; call it first in every owned thread."""
        self._local.owned = True

    @contextmanager
    def borrow(self) -> Iterator[None]:
        """Own the calling thread while it runs work of the run, then restore.

        For a host thread that runs handlers inline, e.g. ``signal()``.
        """
        previous = getattr(self._local, "owned", False)
        self._local.owned = True
        try:
            yield
        finally:
            self._local.owned = previous

    def owns_current(self) -> bool:
        """Whether the calling thread is owned."""
        owned = getattr(self._local, "owned", False)

        return owned

    def reject_wait(
        self,
        action: str,
        *,
        remedy: str = "Call stop() there and wait from another thread",
    ) -> None:
        """Raise when the calling thread is owned.

        Args:
            action: What would wait, e.g. ``"ActiveRun.wait()"``.
            remedy: What the caller should do instead; ends the message.

        Raises:
            ContractError: When called on an owned thread (a handler, an
                observer callback, a block call or a source read).
        """
        if self.owns_current():
            raise ContractError(
                f"{action} was called from a thread of the same run (a handler, "
                f"observer, block or source); it would wait for itself. {remedy}"
            )


class WorkerPool:
    """Fixed number of threads running submitted work, without a queue.

    Args:
        size: Number of workers, at least 1.
        name: Thread name prefix.
        owner: Threads of the run; every worker marks itself owned.
        counters: Counters whose ``executing`` gauge tracks busy workers.
        on_free: Called (no pool lock held) each time a worker becomes free,
            e.g. to wake a dispatcher.
    """

    def __init__(
        self,
        size: int,
        *,
        name: str,
        owner: OwnedThreads,
        counters: Optional[PipelineCounters] = None,
        on_free: Optional[Callable[[], None]] = None,
    ):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ContractError(f"WorkerPool size must be an int >= 1, got {size!r}")

        self._owner = owner
        self._counters = counters
        self._on_free = on_free
        self._condition = threading.Condition()
        self._free = size
        self._handoff: List[Work] = []
        self._closed = False
        self.unexpected: Optional[BaseException] = None
        self._threads = [
            threading.Thread(target=self._serve, name=f"{name}-{number}", daemon=True)
            for number in range(size)
        ]
        self._size = size
        self._launch()

    @property
    def size(self) -> int:
        """Number of workers."""
        return self._size

    def try_submit(self, work: Work) -> bool:
        """Start ``work`` on an idle worker now, if there is one.

        Returns:
            ``True`` when a worker accepted the work; ``False`` when every
            worker is busy or the pool is closed. Nothing is queued.
        """
        with self._condition:
            accepted = self._accept(work)

        return accepted

    def submit(self, work: Work, *, timeout: Optional[float] = None) -> bool:
        """Wait for an idle worker and start ``work`` on it.

        Args:
            work: Callable taking no arguments.
            timeout: Seconds to wait at most; ``None`` waits until a worker
                is free or the pool closes.

        Returns:
            ``True`` when accepted; ``False`` on timeout or a closed pool.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._free > 0, timeout=timeout
            )
            accepted = self._accept(work)

        return accepted

    def wait_free(self, timeout: Optional[float] = None) -> bool:
        """Wait until at least one worker is idle.

        With a single submitter, a following ``try_submit`` then succeeds.

        Args:
            timeout: Seconds to wait at most; ``None`` waits until a worker
                is free or the pool closes.

        Returns:
            ``True`` when a worker is free; ``False`` on timeout or a closed
            pool.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._free > 0, timeout=timeout
            )
            free = not self._closed and self._free > 0

        return free

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait until no work is running.

        Returns:
            ``True`` when idle; ``False`` on timeout.

        Raises:
            ContractError: When called from a worker of this pool.
        """
        self._reject_own_worker("wait_idle()")
        with self._condition:
            idle = self._condition.wait_for(
                lambda: self._free == self._size, timeout=timeout
            )

        return idle

    def close(self) -> None:
        """Refuse new work and wake waiting submitters; running work continues."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def shutdown(self) -> None:
        """Close, let running work finish and join every worker.

        Raises:
            ContractError: When called from a worker of this pool.
        """
        self._reject_own_worker("shutdown()")
        self.close()
        for thread in self._threads:
            thread.join()

    def _launch(self) -> None:
        """Start every worker; if one fails to start, stop and join the started ones.

        The constructor then raises the original error with no worker left
        running, so a caller that never received the pool has nothing to clean.
        """
        started = []
        try:
            for thread in self._threads:
                thread.start()
                started.append(thread)
        except BaseException:
            self.close()
            for thread in started:
                thread.join()
            raise

    def _reject_own_worker(self, action: str) -> None:
        if threading.current_thread() in self._threads:
            raise ContractError(
                f"WorkerPool {action} was called from one of its own workers; it "
                "would wait for itself"
            )

    def _accept(self, work: Work) -> bool:
        """Hand work to an idle worker (condition held)."""
        if self._closed or self._free == 0:
            return False

        self._free -= 1
        self._handoff.append(work)
        self._condition.notify_all()

        return True

    def _serve(self) -> None:
        self._owner.mark()
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._handoff or self._closed)
                if not self._handoff:
                    return
                work = self._handoff.pop(0)
            self._guarded(work, executing=True)
            # An idle worker must not keep the last work's closure (and its data) alive.
            work = None
            with self._condition:
                self._free += 1
                self._condition.notify_all()
            if self._on_free is not None:
                self._guarded(self._on_free, executing=False)

    def _guarded(self, work: Work, *, executing: bool) -> None:
        """Run a callable on this worker; keep the first unexpected error."""
        try:
            if executing and self._counters is not None:
                with self._counters.track("executing"):
                    work()
            else:
                work()
        except BaseException as error:
            with self._condition:
                if self.unexpected is None:
                    self.unexpected = error
