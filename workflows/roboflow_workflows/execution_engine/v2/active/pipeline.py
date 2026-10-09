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

A graph update pauses the driver: ``block`` readers holding a slot wait
before admitting, ``latest`` readers keep replacing their pending emission
(still counted dropped) but nothing pending is promoted, queued pulses and
domain ends still dispatch, and the dispatcher reports the boundary once
nothing is queued or running. A source that ends meanwhile is sealed at
``resume``, so the run cannot finish across the boundary.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional, Tuple

from roboflow_workflows.execution_engine.v2.active.pulses import SourcePulse, attributed
from roboflow_workflows.execution_engine.v2.controls import ControlSnapshot
from roboflow_workflows.execution_engine.v2.data import Timestamp
from roboflow_workflows.execution_engine.v2.operators import TerminationReason
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.workers import WorkerPool
from roboflow_workflows.execution_engine.v2.sources import Emission

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.active.runtime import (
        ActiveRun,
        _SourceSlot,
        _UpdateToken,
    )

__all__ = ["PipelinedDriver"]

Work = Callable[[], None]


@dataclass(frozen=True)
class _End:
    """A domain end waiting for a worker, with the snapshot it was scheduled under."""

    domain: str
    reason: TerminationReason
    controls: ControlSnapshot


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
        self._in_flight = run.session.controls.in_flight
        # Reentrant: sealing a source under it queues the source's end.
        self._condition = threading.Condition(threading.RLock())
        self._admission_open = True
        self._ingress: Dict[str, _Ingress] = {
            name: _Ingress(slot=slot, latest=options.overload_for(name) == "latest")
            for name, slot in run._slots.items()
        }
        self._rotation: List[str] = list(self._ingress)
        self._next_source = 0
        self._ends: Deque[_End] = deque()
        self._running = 0
        self._operators_released = False
        self._paused = False
        self._token: Optional["_UpdateToken"] = None
        self._acknowledged = False
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
            while self._paused and self._admission_open and not ingress.latest:
                self._condition.wait()
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
            if not self._paused:
                # Paused: sealed at resume, under the graph published by then.
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
            # Sequenced like an admission: the snapshot is taken under the
            # driver's lock and the end is dispatched in version order.
            controls = self._run.session.controls.current
            self._in_flight.enter(controls.version)
            self._ends.append(_End(domain, reason, controls))
            self._condition.notify_all()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def pause(self, token: "_UpdateToken") -> None:
        with self._condition:
            self._paused = True
            self._token = token
            self._acknowledged = False
            self._condition.notify_all()

    def resume(self, token: "_UpdateToken") -> None:
        with self._condition:
            if self._token is not token:
                return
            self._paused = False
            self._token = None
            for ingress in self._ingress.values():
                if not ingress.reading:
                    self._seal_if_exhausted(ingress)
            self._condition.notify_all()

    def restart_domains(self) -> None:
        with self._condition:
            for ingress in self._ingress.values():
                ingress.sealed = False

    def settle_workers(self, timeout: float) -> bool:
        """``True`` once every worker handed back its last work (the updater's thread)."""
        idle = self._pool.wait_idle(timeout)

        return idle

    def readers_finished(self) -> bool:
        with self._condition:
            finished = all(not ingress.reading for ingress in self._ingress.values())

        return finished

    def frontiers(self) -> Dict[str, int]:
        with self._condition:
            frontiers = {
                name: ingress.slot.next_sequence
                for name, ingress in self._ingress.items()
            }

        return frontiers

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
        boundary = None
        with self._condition:
            while not self._has_work():
                if self._run.aborting:
                    self._cancel_unstarted()
                    if self._running == 0 and not self._operators_released:
                        break
                if self._exhausted():
                    return False
                boundary = self._boundary()
                if boundary is not None:
                    break
                self._condition.wait()
        if self._run.aborting:
            self._release_operators()
            return True
        if boundary is not None:
            # Reported outside the condition: the run's lock is taken first
            # by an updater that then pauses this driver.
            self._run._boundary_reached(boundary)
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
        """Whether a pending ``latest`` emission may be admitted now (condition held).

        Not while paused: an admitted pulse takes its sequence, control
        snapshot and graph at admission, which the update is about to change.
        """
        ready = (
            not self._paused
            and ingress.pending is not None
            and ingress.slot.in_flight < self._bound
        )

        return ready

    def _boundary(self) -> Optional["_UpdateToken"]:
        """The token to acknowledge once, when paused with nothing left to run."""
        if self._paused and not self._acknowledged and self._running == 0:
            self._acknowledged = True
            return self._token

        return None

    def _exhausted(self) -> bool:
        """Every reader finished and nothing is queued, pending or running.

        Never while paused: a source that ended meanwhile is sealed at
        resume, and the update decides whether the run continues.
        """
        exhausted = (
            not self._paused
            and self._running == 0
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
        # Lowest control version first (controls.py, rule 2), so work admitted
        # under an older version is never left queued behind newer work that
        # may wait for it; ties keep the order ends first, then rotation.
        oldest = self._oldest_queued_version()
        if self._ends and self._ends[0].controls.version == oldest:
            end = self._ends.popleft()
            return lambda: self._run_end(end)

        for offset in range(len(self._rotation)):
            position = (self._next_source + offset) % len(self._rotation)
            ingress = self._ingress[self._rotation[position]]
            if ingress.queued and ingress.queued[0].controls.version != oldest:
                continue
            if not ingress.queued and oldest != self._run.session.controls.version:
                continue  # a pending ``latest`` emission would be newer
            pulse = self._next_pulse(ingress)
            if pulse is not None:
                self._next_source = (position + 1) % len(self._rotation)
                return lambda: self._run_pulse(ingress, pulse)

        raise RuntimeError("eligible work disappeared while the dispatcher held it")

    def _oldest_queued_version(self) -> int:
        """Lowest control version among queued ends and pulses (condition held).

        A pending ``latest`` emission is admitted at dispatch and takes the
        current version, which no queued work exceeds.
        """
        versions = [self._ends[0].controls.version] if self._ends else []
        versions.extend(
            ingress.queued[0].controls.version
            for ingress in self._ingress.values()
            if ingress.queued
        )
        oldest = min(versions) if versions else self._run.session.controls.version

        return oldest

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
        while self._ends:
            self._in_flight.leave(self._ends.popleft().controls.version)
        for ingress in self._ingress.values():
            while ingress.queued:
                self._in_flight.leave(ingress.queued.popleft().controls.version)
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
            self._in_flight.leave(pulse.controls.version)
            with self._condition:
                self._finish_admitted(ingress)
                self._running -= 1
                self._condition.notify_all()

    def _run_end(self, end: "_End") -> None:
        try:
            self._counters.count("end_tasks")
            self._executor.end_domain(end.domain, end.reason, controls=end.controls)
        except Exception as raised:
            self._run._fail(attributed(raised, stage="observer"))
        finally:
            self._in_flight.leave(end.controls.version)
            with self._condition:
                self._running -= 1
                self._condition.notify_all()
