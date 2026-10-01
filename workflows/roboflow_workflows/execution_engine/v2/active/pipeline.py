"""Pipelined driver of an active run: bounded workers fed by one dispatcher.

``ExecutionSession.start(..., pipeline=PipelineOptions(...))`` selects it::

    reader[S] --admit--> ingress[S] --dispatcher--> worker: one whole pulse
      block   wait for one of S's admission_bound slots, then queue the
              admitted pulse (FIFO)
      latest  never wait: replace S's one pending emission (the replaced
              one counts dropped); nothing is admitted yet

    dispatcher (one thread), repeatedly:
      wait for work, then for a free worker (it is the pool's only submitter)
      take, in this order:
        1. a domain end (pulses.end_domain), queued when a domain is sealed
           and drained
        2. the next source, round robin:
             block   its oldest queued pulse
             latest  its pending emission, admitted now, if S has a free slot

    worker: the pulse with its deliveries and inline operator pulses
            (pulses.PulseExecutor), then release S's slot

A pulse is admitted at most ``admission_bound`` times per source and runs on
at most ``max_in_flight`` workers; inline operator pulses run on the worker
that emitted them, so live run states stay below ``max_in_flight`` times one
plus the longest operator chain. Nothing queues inside the worker pool.

Completion: every reader finished, nothing queued, pending or running. Then
the workers are joined, operators closed and ``on_run_finished`` called. On a
failure or ``cancel()`` the dispatcher cancels what was admitted but not
started, waits until no work runs, closes the operators (releasing what they
retained) and then waits for the readers, which close their sources.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional, Tuple

from roboflow_workflows.execution_engine.v2.active.pulses import SourcePulse, attributed
from roboflow_workflows.execution_engine.v2.data import Timestamp
from roboflow_workflows.execution_engine.v2.operators import TerminationReason
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.workers import WorkerPool
from roboflow_workflows.execution_engine.v2.sources import Emission

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.active.runtime import (
        ActiveRun,
        _SourceSlot,
    )

__all__ = ["PipelinedDriver"]

Work = Callable[[], None]


@dataclass
class _Ingress:
    """What one source has read but not yet given to a worker.

    Args:
        slot: The source.
        latest: Overload policy ``latest``; otherwise ``block``.
        queued: Admitted pulses not yet dispatched (``block`` only).
        pending: The newest read, unadmitted emission (``latest`` only).
        reading: Whether the reader may still admit.
        sealed: Whether the source's domain was sealed.
    """

    slot: "_SourceSlot"
    latest: bool
    queued: Deque[SourcePulse] = field(default_factory=deque)
    pending: Optional[Tuple[Emission, Timestamp]] = None
    reading: bool = True
    sealed: bool = False


class PipelinedDriver:
    """Admission, dispatch and lifecycle of one pipelined active run.

    Args:
        run: The active run; the members it may use and the obligations of
            a driver are documented on ``runtime._Driver``.
        options: Worker count and overload policies.
        admission_bound: Pulses one source may have admitted at once.
    """

    def __init__(
        self, run: "ActiveRun", *, options: PipelineOptions, admission_bound: int
    ):
        self._run = run
        self._options = options
        self._bound = admission_bound
        self._executor = run._executor
        self._counters = run.pipeline_counters
        # Reentrant: sealing a source under it queues the source's end.
        self._condition = threading.Condition(threading.RLock())
        self._admission_open = True
        self._ingress: Dict[str, _Ingress] = {
            name: _Ingress(slot=slot, latest=options.overload_for(name) == "latest")
            for name, slot in run._slots.items()
        }
        self._rotation: List[str] = list(self._ingress)
        self._next_source = 0
        self._ends: Deque[Tuple[str, TerminationReason]] = deque()
        self._running = 0
        self._operators_released = False
        self._pool: Optional[WorkerPool] = None
        self._dispatcher = threading.Thread(
            target=self._dispatch,
            name=f"workflows-v2-run-{run.run_id[:8]}",
            daemon=True,
        )

    # Driver interface (called by ActiveRun and its readers) -------------------

    def start(self) -> None:
        """Start the workers and the dispatcher; stop the workers if that fails."""
        self._pool = WorkerPool(
            self._options.max_in_flight,
            name=f"workflows-v2-worker-{self._run.run_id[:8]}",
            owner=self._run._owned,
            counters=self._counters,
        )
        try:
            self._dispatcher.start()
        except BaseException:
            self._pool.shutdown()
            raise

    def admit(
        self, slot: "_SourceSlot", emission: Emission, *, observed: Timestamp
    ) -> bool:
        """Admit (``block``) or keep pending (``latest``) one read emission.

        Returns ``False`` once admission is closed; the emission is then
        counted unadmitted and the reader stops.
        """
        ingress = self._ingress[slot.name]
        if not ingress.latest:
            slot.admission.acquire()
        with self._condition:
            if not self._admission_open:
                slot.counters.unadmitted += 1
                return False
            if ingress.latest:
                if ingress.pending is not None:
                    slot.counters.dropped += 1
                ingress.pending = (emission, observed)
            else:
                pulse = self._run._admitted(slot, emission, observed=observed)
                ingress.queued.append(pulse)
                self._counters.add("queued", 1)
            self._condition.notify_all()

        return True

    def reader_finished(self, source: str) -> None:
        with self._condition:
            ingress = self._ingress[source]
            ingress.reading = False
            self._seal_if_exhausted(ingress)
            self._condition.notify_all()

    def close_admission(self) -> bool:
        """Close admission and discard pending emissions; ``True`` the first time."""
        with self._condition:
            if not self._admission_open:
                return False
            self._admission_open = False
            for ingress in self._ingress.values():
                if ingress.pending is not None:
                    ingress.pending = None
                    ingress.slot.counters.unadmitted += 1
                    self._seal_if_exhausted(ingress)
            self._condition.notify_all()

        return True

    def end_domain_later(self, domain: str, reason: TerminationReason) -> None:
        with self._condition:
            self._ends.append((domain, reason))
            self._condition.notify_all()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    # Dispatcher thread ----------------------------------------------------

    def _dispatch(self) -> None:
        self._run._owned.mark()
        try:
            while self._dispatch_next():
                pass
        except Exception as raised:
            self._run._fail(attributed(raised, stage="observer"))
        finally:
            self._finish()

    def _dispatch_next(self) -> bool:
        """Hand one unit of work to a worker; ``False`` once nothing is left."""
        with self._condition:
            while not self._has_work():
                if self._run.aborting:
                    self._cancel_unstarted()
                    if self._running == 0 and not self._operators_released:
                        break
                if self._exhausted():
                    return False
                self._condition.wait()
        if self._run.aborting:
            self._release_operators()
            return True

        # Only this thread submits, so a free worker is still free below.
        self._pool.wait_free()
        with self._condition:
            work = self._take_work()
        if work is not None and not self._pool.try_submit(work):
            raise RuntimeError("the worker pool refused work while a worker was free")

        return True

    def _has_work(self) -> bool:
        """Whether a domain end or an eligible pulse waits (condition held)."""
        if self._run.aborting:
            return False

        eligible = bool(self._ends) or any(
            ingress.queued or self._latest_ready(ingress)
            for ingress in self._ingress.values()
        )

        return eligible

    def _latest_ready(self, ingress: _Ingress) -> bool:
        ready = ingress.pending is not None and ingress.slot.in_flight < self._bound

        return ready

    def _exhausted(self) -> bool:
        """Every reader finished and nothing is queued, pending or running."""
        exhausted = (
            self._running == 0
            and not self._ends
            and all(
                not ingress.reading and not ingress.queued and ingress.pending is None
                for ingress in self._ingress.values()
            )
        )

        return exhausted

    def _take_work(self) -> Optional[Work]:
        """Pop the next unit of work and count it running (condition held)."""
        if not self._has_work():
            return None

        self._running += 1
        if self._ends:
            domain, reason = self._ends.popleft()
            return lambda: self._run_end(domain, reason)

        for offset in range(len(self._rotation)):
            position = (self._next_source + offset) % len(self._rotation)
            ingress = self._ingress[self._rotation[position]]
            pulse = self._next_pulse(ingress)
            if pulse is not None:
                self._next_source = (position + 1) % len(self._rotation)
                return lambda: self._run_pulse(ingress, pulse)

        raise RuntimeError("eligible work disappeared while the dispatcher held it")

    def _next_pulse(self, ingress: _Ingress) -> Optional[SourcePulse]:
        """The source's next pulse to run, admitting a pending one (condition held)."""
        if ingress.queued:
            self._counters.add("queued", -1)
            pulse = ingress.queued.popleft()
            return pulse
        if not self._latest_ready(ingress):
            return None

        emission, observed = ingress.pending
        ingress.pending = None
        pulse = self._run._admitted(ingress.slot, emission, observed=observed)
        self._seal_if_exhausted(ingress)

        return pulse

    def _seal_if_exhausted(self, ingress: _Ingress) -> None:
        """Seal a finished source once its pending emission is gone (condition held)."""
        if ingress.sealed or ingress.reading or ingress.pending is not None:
            return

        ingress.sealed = True
        name = ingress.slot.name
        self._executor.seal(name, self._run._termination(name))

    def _cancel_unstarted(self) -> None:
        """Cancel admitted pulses and domain ends that never started (condition held)."""
        self._ends.clear()
        for ingress in self._ingress.values():
            while ingress.queued:
                ingress.queued.popleft()
                self._counters.add("queued", -1)
                self._executor.count(ingress.slot.counters, "cancelled")
                self._finish_admitted(ingress)

    def _finish_admitted(self, ingress: _Ingress) -> None:
        """An admitted pulse is complete or cancelled; free its slot (condition held)."""
        ingress.slot.in_flight -= 1
        if not ingress.latest:
            ingress.slot.admission.release()

    def _release_operators(self) -> None:
        """After an abort, once no work runs: close operators now, not after readers."""
        self._pool.wait_idle()
        self._operators_released = True
        self._executor.close_operators()

    def _finish(self) -> None:
        """Quiesce the workers, join every thread and conclude the run."""
        if self._pool is not None:
            self._pool.wait_idle()
        with self._condition:
            self._cancel_unstarted()
        self._executor.close_operators()
        if self._pool is not None:
            self._pool.shutdown()
        for reader in self._run._readers.values():
            if reader.ident is not None:
                reader.join()
        self._run._conclude()

    # Worker work ----------------------------------------------------------

    # The executor records every failure of a pulse or domain end itself; an
    # exception reaching here is an engine defect, which must fail the run
    # rather than stay in the pool's ``unexpected``.

    def _run_pulse(self, ingress: _Ingress, pulse: SourcePulse) -> None:
        try:
            self._executor.run_source_pulse(pulse, counters=ingress.slot.counters)
        except Exception as raised:
            self._run._fail(attributed(raised, stage="observer"))
        finally:
            with self._condition:
                self._finish_admitted(ingress)
                self._running -= 1
                self._condition.notify_all()

    def _run_end(self, domain: str, reason: TerminationReason) -> None:
        try:
            self._counters.count("end_tasks")
            self._executor.end_domain(domain, reason)
        except Exception as raised:
            self._run._fail(attributed(raised, stage="observer"))
        finally:
            with self._condition:
                self._running -= 1
                self._condition.notify_all()
