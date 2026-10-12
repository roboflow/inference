"""Teams: the workers of one pipeline worker, executing one pulse branch by branch.

A pipeline has ``max_in_flight`` (P) workers. With ``max_workflow_parallelism``
(W) above 1, each worker leads a team: itself plus W - 1 helper threads, so
the pipeline owns exactly P x W threads. A team executes one pulse (or passive
submission) at a time; its units are the pulse's steps and, for an active
pulse, its group deliveries::

    team = Team(3, name="run-1a2b-0", owner=owned)   # starts 2 helpers
    with bound_team(team):                           # in the leading worker
        team = team_for(coordination)                # idle team, or None
        completed = team.run(graph, execute=run_unit, stop=aborting)
    team.close()

    run(graph):                     the caller is worker 0
        ready: units whose predecessors all settled, lowest priority first
        a worker that settles a unit continues with its best newly ready
        successor (depth first); other ready units wait for an idle worker
        joins are dependency counts: no worker waits for a child branch
        returns once every started unit settled: no helper still runs

The first error halts the job: no further unit starts, running units finish,
and ``run`` raises it (a real failure is preferred over ``RunAborted``).
``stop()`` is asked before each unit; ``True`` halts without an error and
``run`` returns ``False``. Helpers run their units in a copy of the caller's
``contextvars`` taken at ``run``, so the pulse's context reaches every worker.

``team_for(coordination)`` is the one place that decides whether a pulse
runs on a team; ``None`` means plan order on the calling thread.
"""

import contextvars
import heapq
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Protocol, Sequence, Tuple

from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    Coordination,
    PipelineCounters,
    RunAborted,
)

__all__ = ["Team", "UnitGraph", "bound_team", "team_for"]


class _Stopped(Exception):
    """``stop()`` returned ``True`` before a unit started."""


_STOPPED = _Stopped()


class UnitGraph(Protocol):
    """Dependencies of units ``0..n-1``; ``plan.StepSchedule`` is one.

    Attributes:
        predecessors: Units each unit depends on.
        successors: Units depending on each unit.
        priorities: Rank of every unit; among ready units the lowest starts first.
    """

    predecessors: Sequence[Sequence[int]]
    successors: Sequence[Sequence[int]]
    priorities: Sequence[int]


class _Job:
    """Bookkeeping of one ``Team.run``; every field changes under the team lock."""

    def __init__(
        self,
        graph: UnitGraph,
        *,
        execute: Callable[[int], None],
        stop: Callable[[], bool],
    ):
        self.graph = graph
        self.execute = execute
        self.stop = stop
        self.context = contextvars.copy_context()
        self.remaining = [len(before) for before in graph.predecessors]
        self.ready = [
            (graph.priorities[unit], unit)
            for unit, count in enumerate(self.remaining)
            if count == 0
        ]
        heapq.heapify(self.ready)
        self.running = 0
        self.settled = 0
        self.halted = False
        self.stopped = False
        self.error: Optional[BaseException] = None

    @property
    def startable(self) -> bool:
        return bool(self.ready) and not self.halted

    @property
    def finished(self) -> bool:
        """No unit runs and none will start."""
        done = self.halted or self.settled == len(self.remaining)
        finished = self.running == 0 and done

        return finished

    def fail(self, error: BaseException) -> None:
        self.halted = True
        replaces = isinstance(self.error, RunAborted) and not isinstance(
            error, RunAborted
        )
        if self.error is None or replaces:
            self.error = error


class Team:
    """The calling worker plus ``size - 1`` helper threads; one job at a time.

    Args:
        size: Workers of the team, the caller included; at least 1.
        name: Thread name prefix of the helpers.
        mark_owned: Called first in every helper thread, e.g.
            ``OwnedThreads.mark``, so the helper is known as the run's.
        counters: Counters whose ``parallel_units`` gauge counts the units
            running on team workers.
    """

    def __init__(
        self,
        size: int,
        *,
        name: str,
        mark_owned: Callable[[], None],
        counters: Optional[PipelineCounters] = None,
    ):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ContractError(f"Team size must be an int >= 1, got {size!r}")

        self.size = size
        self._mark_owned = mark_owned
        self._counters = counters
        self._condition = threading.Condition()
        self._job: Optional[_Job] = None
        self._closed = False
        self._helpers = [
            threading.Thread(target=self._serve, name=f"{name}.{number}", daemon=True)
            for number in range(1, size)
        ]
        started = []
        try:
            for helper in self._helpers:
                helper.start()
                started.append(helper)
        except BaseException:
            self._helpers = started
            self.close()
            raise

    @property
    def threads(self) -> Tuple[threading.Thread, ...]:
        """The helper threads (the leading worker is not one of them)."""
        return tuple(self._helpers)

    @property
    def idle(self) -> bool:
        """Whether no job runs."""
        with self._condition:
            idle = self._job is None

        return idle

    def run(
        self,
        graph: UnitGraph,
        *,
        execute: Callable[[int], None],
        stop: Callable[[], bool],
    ) -> bool:
        """Execute every unit of ``graph``; the calling thread is worker 0.

        Args:
            graph: Units and dependencies of one pulse.
            execute: Runs one unit; called at most once per unit.
            stop: Asked before each unit; ``True`` halts the job.

        Returns:
            ``True`` when every unit ran; ``False`` when ``stop`` halted it.

        Raises:
            BaseException: The first error a unit raised, after every
                started unit settled.
            RuntimeError: When the team already runs a job, or the graph
                leaves units that can never become ready (a cycle).
        """
        job = _Job(graph, execute=execute, stop=stop)
        with self._condition:
            if self._job is not None:
                raise RuntimeError("a team runs one job at a time")
            self._job = job
            self._condition.notify_all()
        try:
            self._work(job, leader=True)
        finally:
            with self._condition:
                self._job = None
        if job.error is not None:
            error, job.error = job.error, None
            raise error

        completed = not job.stopped

        return completed

    def close(self) -> None:
        """Stop and join the helpers; a running job is not interrupted.

        Raises:
            ContractError: When called from one of the helpers.
        """
        if threading.current_thread() in self._helpers:
            raise ContractError("Team.close() was called from one of its own helpers")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        for helper in self._helpers:
            helper.join()

    def _serve(self) -> None:
        self._mark_owned()
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed
                    or (self._job is not None and self._job.startable)
                )
                if self._closed:
                    return
                job = self._job
            job.context.copy().run(self._work, job, leader=False)
            # An idle helper must not keep the job (and the pulse it reaches) alive.
            job = None

    def _work(self, job: _Job, *, leader: bool) -> None:
        """Run units until none is ready; the leader also waits for the job's end."""
        with self._condition:
            unit = self._claim(job, leader=leader)
        while unit is not None:
            failure = self._execute(job, unit)
            with self._condition:
                unit = self._settle(job, unit, failure=failure)
                if unit is None:
                    unit = self._claim(job, leader=leader)

    def _execute(self, job: _Job, unit: int) -> Optional[BaseException]:
        """Run one unit outside the lock; return what it raised, or ``_STOPPED``."""
        try:
            if job.stop():
                return _STOPPED
            if self._counters is None:
                job.execute(unit)
            else:
                with self._counters.track("parallel_units"):
                    job.execute(unit)
        except BaseException as error:
            return error

        return None

    def _claim(self, job: _Job, *, leader: bool) -> Optional[int]:
        """Take the best ready unit (lock held); ``None`` when this worker is done.

        A helper is done when nothing is ready. The leader waits until a unit
        is ready or the job finished, so ``run`` returns only after every
        started unit settled.
        """
        while True:
            if job.startable:
                _, unit = heapq.heappop(job.ready)
                job.running += 1
                return unit
            if job.finished or not leader:
                return None
            if job.running == 0:
                job.fail(
                    RuntimeError(
                        f"{len(job.remaining) - job.settled} unit(s) can never become "
                        "ready: their dependencies form a cycle"
                    )
                )
                return None
            self._condition.wait()

    def _settle(
        self, job: _Job, unit: int, *, failure: Optional[BaseException]
    ) -> Optional[int]:
        """Record a unit's end (lock held); return the successor to continue with."""
        job.running -= 1
        if failure is _STOPPED:
            job.stopped = job.halted = True
        elif failure is not None:
            job.fail(failure)
        if job.halted:
            self._condition.notify_all()
            return None

        job.settled += 1
        ready = []
        for successor in job.graph.successors[unit]:
            job.remaining[successor] -= 1
            if job.remaining[successor] == 0:
                ready.append((job.graph.priorities[successor], successor))
        if not ready:
            if job.running == 0:
                self._condition.notify_all()  # the leader may wait for the end
            return None

        ready.sort()
        for item in ready[1:]:
            heapq.heappush(job.ready, item)
        if len(ready) > 1:
            self._condition.notify_all()
        job.running += 1
        continued = ready[0][1]

        return continued


_BOUND = threading.local()


@contextmanager
def bound_team(team: Optional[Team]) -> Iterator[None]:
    """Make ``team`` the calling worker's team for the ``with`` block.

    Args:
        team: The worker's team; ``None`` binds nothing.

    Yields:
        Nothing; the previous binding is restored on exit.
    """
    previous = getattr(_BOUND, "team", None)
    _BOUND.team = team
    try:
        yield
    finally:
        _BOUND.team = previous


def team_for(coordination: Coordination) -> Optional[Team]:
    """Return the team that may run one pulse's units in parallel now.

    ``None`` means: run the units in plan order on the calling thread. Each
    condition below is a reason the calling thread may not lead a job:

    - The run is serial. A serial session may run nested inside pipeline
      work: a block, handler or operator starts it on a worker whose team is
      idle at that moment but belongs to the outer run. A serial run keeps
      plan order (it is the reference) and must not occupy the outer run's
      helpers or count in its gauges.
    - The thread leads no team: a helper, a reader or a passive caller.
    - The team has one worker (W = 1).
    - The team already runs a job: the caller is one of that job's units.

    Args:
        coordination: Coordination of the run asking.

    Returns:
        The calling worker's idle team of at least two workers, or ``None``.
    """
    if not coordination.pipelined:
        return None

    team = getattr(_BOUND, "team", None)
    if team is None or team.size == 1 or not team.idle:
        return None

    return team
