"""Stage gates: which pulse may call a stage next.

A pipelined run executes each pulse on one worker, in plan order. Different
pulses may be at different stages at once. A stage is one gated unit of
work: a whole step call, one phase of a step, a group delivery or an
operator push. Its gate admits one call at a time, in ticket order per
domain::

    Ticket(domain, ordinal)   domain: source, operator or "$passive";
                              ordinal: the pulse's sequence, from 0, no gaps

    wait_turn(t)   wait until every earlier ordinal of t.domain is done here
    call(t)        hold the gate's single call slot (shared by all domains)
    done(t)        retire t; the next ordinal of t.domain may take its turn

``StepStages`` drives the gates of one step for one pulse with ``n`` calls
(one per invocation, or one batch call)::

    unit u (a phase, or "call")    first use: wait_turn; every use: call
                                   after the n-th call of u: done
                                   n == 0: done at once, never waits

So a phased step with ``n == 1`` hands phase A to the next pulse as soon
as A returned ready, while this pulse continues with phase B. Nothing waits
for the end of the step. A failed call is never retired. Whoever sees the
failure aborts the coordination with it (``execute_step`` for step
failures, the driver for handler and operator failures); every waiter then
raises ``RunAborted`` and ``abort_cause`` names the failure.

No lock is held while a block, operator or handler runs, and no gate is
held while waiting for another gate. A waiter only waits for an earlier
ordinal of its own domain, or for a call in progress.
"""

import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.execution.steps import RunState
    from roboflow_workflows.execution_engine.v2.plan import (
        ErrorHandler,
        ExecutionObserver,
        ExecutionSession,
        PlannedStep,
    )

__all__ = [
    "GAUGES",
    "PASSIVE_DOMAIN",
    "SERIAL",
    "Coordination",
    "PipelineCounters",
    "PipelinedCoordination",
    "RunAborted",
    "StageCounters",
    "StageGate",
    "StepStages",
    "Ticket",
    "gates_each_phase",
    "step_stage_units",
]

WHOLE_CALL = "call"
"""Unit of a step whose call is gated as a whole (run mode, phase_overlap False)."""

GAUGES: Tuple[str, ...] = (
    "executing",
    "queued",
    "live_states",
    "pending_operator_pulses",
    "overlapping_pulses",
)
"""Current/peak counts a pipelined run tracks; all are counts, never bytes.

executing                 workers running a pulse, submission or end task
queued                    admitted pulses not yet given to a worker
live_states               RunStates alive, inline operator pulses included
pending_operator_pulses   pulses an operator returned, not yet started
overlapping_pulses        gated calls in progress at once; above 1 = overlap
"""

TOTALS: Tuple[str, ...] = ("end_tasks",)
"""Monotonic totals of a pipelined run: end-of-domain tasks run on workers."""

PASSIVE_DOMAIN = "$passive"
"""Ticket domain of passive pipeline submissions."""


class RunAborted(BaseException):
    """The coordination is aborting; the waiting or checking pulse stops.

    Engine-internal: drivers catch it and report the failure that caused
    the abort, or the cancellation. It derives from ``BaseException`` (like
    ``KeyboardInterrupt``), so handlers of ``Exception`` around block calls
    never turn it into a step failure, an ``on_error`` or an error-handler
    call.
    """


@dataclass(frozen=True)
class Ticket:
    """A pulse's place in the per-domain order of every stage.

    Args:
        domain: Source or operator name, or ``"$passive"``.
        ordinal: The pulse's sequence in its domain, from 0, without gaps.
    """

    domain: str
    ordinal: int


@dataclass
class StageCounters:
    """Counts of one stage; written by its gate under the gate's lock.

    Args:
        calls: Calls that entered the stage.
        waited_turn: Turns that waited for an earlier ordinal.
        waited_call: Calls that waited for another call to leave the stage.
        busy_ns: Engine monotonic time spent inside calls.
    """

    calls: int = 0
    waited_turn: int = 0
    waited_call: int = 0
    busy_ns: int = 0


class PipelineCounters:
    """Thread-safe counters of one pipelined run or passive pipeline.

    Gauges (``GAUGES``) keep a current value and its peak; totals
    (``TOTALS``) only grow. ``snapshot()`` returns everything as plain data.
    All values are counts or nanoseconds; none bounds bytes or device memory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = dict.fromkeys(GAUGES, 0)
        self._peak = dict.fromkeys(GAUGES, 0)
        self._totals = dict.fromkeys(TOTALS, 0)
        self._stages: Dict[str, StageCounters] = {}
        self._age_last_ns: Dict[str, int] = {}
        self._age_max_ns: Dict[str, int] = {}

    def add(self, gauge: str, delta: int = 1) -> None:
        """Change a gauge by ``delta`` and update its peak.

        Args:
            gauge: One of ``GAUGES``.
            delta: Signed change.
        """
        with self._lock:
            value = self._current[gauge] + delta
            self._current[gauge] = value
            if value > self._peak[gauge]:
                self._peak[gauge] = value

    @contextmanager
    def track(self, gauge: str) -> Iterator[None]:
        """Count one unit of ``gauge`` for the duration of the ``with`` block.

        Args:
            gauge: One of ``GAUGES``.

        Yields:
            Nothing; the gauge is decremented on exit, also on errors.
        """
        self.add(gauge, 1)
        try:
            yield
        finally:
            self.add(gauge, -1)

    def count(self, total: str, amount: int = 1) -> None:
        """Increase a total.

        Args:
            total: One of ``TOTALS``.
            amount: Non-negative increase.
        """
        with self._lock:
            self._totals[total] += amount

    def record_result_age(self, group: str, age_ns: int) -> None:
        """Record the age of one delivered group result.

        Args:
            group: Output group name.
            age_ns: Delivery time minus the pulse's observation time.
        """
        with self._lock:
            self._age_last_ns[group] = age_ns
            self._age_max_ns[group] = max(age_ns, self._age_max_ns.get(group, 0))

    def current(self, gauge: str) -> int:
        """Return the current value of a gauge."""
        with self._lock:
            value = self._current[gauge]

        return value

    def peak(self, gauge: str) -> int:
        """Return the highest value a gauge reached."""
        with self._lock:
            value = self._peak[gauge]

        return value

    def stage(self, name: str) -> StageCounters:
        """Return the counters of one stage, created on first use."""
        with self._lock:
            counters = self._stages.setdefault(name, StageCounters())

        return counters

    def snapshot(self) -> Dict[str, Any]:
        """Return every counter as JSON-friendly data.

        Returns:
            ``current``, ``peak`` and ``totals`` by name, ``stages`` by stage
            name and ``result_age_ns`` (``last`` and ``max``) by group.
        """
        with self._lock:
            snapshot = {
                "current": dict(self._current),
                "peak": dict(self._peak),
                "totals": dict(self._totals),
                "stages": {name: asdict(stage) for name, stage in self._stages.items()},
                "result_age_ns": {
                    group: {"last": last, "max": self._age_max_ns[group]}
                    for group, last in self._age_last_ns.items()
                },
            }

        return snapshot


class StageGate:
    """One stage: one call at a time, turns in ordinal order per domain.

    Args:
        name: Stage name, e.g. ``"$steps.classify#logits"``.
        aborted: Set when the coordination aborts; waiters then raise.
        counters: Counters of the run; the gate writes its own stage entry.
    """

    def __init__(
        self, name: str, *, aborted: threading.Event, counters: PipelineCounters
    ):
        self.name = name
        self._aborted = aborted
        self._counters = counters
        self._stats = counters.stage(name)
        self._condition = threading.Condition()
        self._next: Dict[str, int] = {}
        self._finished: Dict[str, Set[int]] = {}
        self._busy = False

    def wait_turn(self, ticket: Ticket) -> None:
        """Wait until every earlier ordinal of the ticket's domain is done here.

        Raises:
            RunAborted: When the coordination aborts first.
        """
        with self._condition:
            self._check_not_done(ticket)
            if self._next_of(ticket.domain) != ticket.ordinal:
                self._stats.waited_turn += 1
            self._wait_until(lambda: self._next_of(ticket.domain) == ticket.ordinal)

    @contextmanager
    def call(self, ticket: Ticket) -> Iterator[None]:
        """Hold the stage for one call of a ticket that has its turn.

        Raises:
            RunAborted: When the coordination aborts before the call starts.
            RuntimeError: When the ticket does not have its turn.
        """
        with self._condition:
            if self._next_of(ticket.domain) != ticket.ordinal:
                raise RuntimeError(f"{ticket} calls stage {self.name} without its turn")
        with self.exclusive(), self._counters.track("overlapping_pulses"):
            yield

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the stage's call slot without a turn (operator end and finish).

        Raises:
            RunAborted: When the coordination aborts before the slot is free.
        """
        with self._condition:
            if self._busy:
                self._stats.waited_call += 1
            self._wait_until(lambda: not self._busy)
            self._busy = True
            self._stats.calls += 1
        started = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed = time.monotonic_ns() - started
            with self._condition:
                self._busy = False
                self._stats.busy_ns += elapsed
                self._condition.notify_all()

    def done(self, ticket: Ticket) -> None:
        """Retire a ticket; never blocks.

        Ordinals retire as a contiguous prefix: a later ordinal done first
        is remembered until every earlier one is done.

        Raises:
            RuntimeError: When the ticket was already retired.
        """
        with self._condition:
            self._check_not_done(ticket)
            finished = self._finished.setdefault(ticket.domain, set())
            finished.add(ticket.ordinal)
            next_ordinal = self._next_of(ticket.domain)
            while next_ordinal in finished:
                finished.remove(next_ordinal)
                next_ordinal += 1
            self._next[ticket.domain] = next_ordinal
            if not finished:
                del self._finished[ticket.domain]
            self._condition.notify_all()

    def wake(self) -> None:
        """Wake every waiter so it can observe an abort."""
        with self._condition:
            self._condition.notify_all()

    def _next_of(self, domain: str) -> int:
        next_ordinal = self._next.get(domain, 0)

        return next_ordinal

    def _check_not_done(self, ticket: Ticket) -> None:
        if ticket.ordinal < self._next_of(ticket.domain) or ticket.ordinal in (
            self._finished.get(ticket.domain, ())
        ):
            raise RuntimeError(f"{ticket} is already retired at stage {self.name}")

    def _wait_until(self, ready: Callable[[], bool]) -> None:
        """Wait on the condition (lock held) until ``ready``; abort wins."""
        while True:
            if self._aborted.is_set():
                raise RunAborted(f"stage {self.name}: the run is aborting")
            if ready():
                return
            self._condition.wait()


class StepStages:
    """Turns of one pulse at the stages of one step, group or operator.

    Args:
        gates: Gate of every unit, e.g. ``{"tensor": gate, "logits": gate}``.
        ticket: The pulse's ticket.
        calls: Calls this pulse makes of every unit; ``0`` retires at once.
    """

    def __init__(
        self,
        gates: Mapping[str, StageGate],
        *,
        ticket: Ticket,
        calls: int,
    ):
        self._gates = dict(gates)
        self._ticket = ticket
        self._calls = calls
        self._uses = dict.fromkeys(self._gates, 0)
        if calls == 0:
            for gate in self._gates.values():
                gate.done(ticket)

    @contextmanager
    def call(self, unit: str) -> Iterator[None]:
        """Run one call of ``unit`` in this pulse's turn.

        Waits for the turn on the unit's first call; retires the ticket right
        after the unit's last (``calls``-th) successful call. A raising call
        is not retired; the caller aborts the coordination with its error.

        Raises:
            RunAborted: When the coordination aborts while waiting.
            RuntimeError: For an unknown unit or a call beyond ``calls``.
        """
        gate = self._gates.get(unit)
        if gate is None:
            raise RuntimeError(
                f"unknown stage unit {unit!r}; units: {list(self._gates)}"
            )
        uses = self._uses[unit]
        if uses >= self._calls:
            raise RuntimeError(
                f"stage {gate.name} called {uses + 1} times; this pulse declared "
                f"{self._calls}"
            )

        if uses == 0:
            gate.wait_turn(self._ticket)
        with gate.call(self._ticket):
            yield

        self._uses[unit] = uses + 1
        if uses + 1 == self._calls:
            gate.done(self._ticket)

    def finish(self) -> None:
        """Retire every unit this pulse never called.

        Raises:
            RuntimeError: When a unit was called, but fewer than ``calls``
                times; its turn is not silently given away.
        """
        partial = [unit for unit, uses in self._uses.items() if 0 < uses < self._calls]
        if partial:
            raise RuntimeError(
                f"stage unit(s) {partial} finished after fewer than {self._calls} "
                "calls"
            )

        for unit, uses in self._uses.items():
            if self._calls and not uses:
                self._uses[unit] = self._calls
                self._gates[unit].done(self._ticket)


class _NoStages:
    """Stages of a serial run: no waiting, nothing to retire."""

    def call(self, unit: str) -> ContextManager[None]:
        return nullcontext()

    def finish(self) -> None:
        pass


class _NoGate:
    """Gate of a serial run: the same calls as ``StageGate``, none waits."""

    name = "serial"

    def wait_turn(self, ticket: Optional[Ticket]) -> None:
        pass

    def call(self, ticket: Optional[Ticket]) -> ContextManager[None]:
        return nullcontext()

    def exclusive(self) -> ContextManager[None]:
        return nullcontext()

    def done(self, ticket: Optional[Ticket]) -> None:
        pass

    def wake(self) -> None:
        pass


_NO_STAGES = _NoStages()
_NO_GATE = _NoGate()


def step_stage_units(step: "PlannedStep") -> Tuple[str, ...]:
    """Units a pipelined run gates for one step.

    One unit per phase when the step executes phases and its implementation
    allows ``phase_overlap``; otherwise one unit, ``"call"``, covering the
    whole call including readiness.

    Args:
        step: A planned step.

    Returns:
        Unit names; a stage is named ``<step selector>#<unit>``.
    """
    if gates_each_phase(step):
        units = tuple(spec.name for spec in step.selected.phases.phases)
        return units

    return (WHOLE_CALL,)


def gates_each_phase(step: "PlannedStep") -> bool:
    """Whether each phase of the step is its own stage (see ``step_stage_units``)."""
    each_phase = step.execution == "phases" and step.selected.phase_overlap

    return each_phase


class Coordination:
    """How the pulses of one run share stages and callbacks: serially.

    The serial coordination (``SERIAL``) gates nothing, never aborts and
    passes the session's observer and error handler through unchanged.
    ``PipelinedCoordination`` overrides every method.
    """

    pipelined: bool = False

    def gate(self, name: str) -> StageGate:
        """Return the gate of a named stage; serially a gate that never waits.

        Args:
            name: Stage name, e.g. ``"$operators.pair#push"``.

        Returns:
            The stage's gate.
        """
        return _NO_GATE

    def step_stages(
        self, run: "RunState", step: "PlannedStep", *, calls: int
    ) -> StepStages:
        """Return the turns of ``run``'s pulse at the stages of ``step``.

        Args:
            run: State of the pulse.
            step: Step about to make ``calls`` calls.
            calls: Number of block calls of the step in this pulse.

        Returns:
            The step's stages; serially, a no-op.
        """
        return _NO_STAGES

    def stages(
        self, run: "RunState", gates: Mapping[str, str], *, calls: int
    ) -> StepStages:
        """Return the turns of ``run``'s pulse at named stages.

        Args:
            run: State of the pulse.
            gates: Unit name to stage name, e.g. ``{"deliver": "$groups.g#deliver"}``.
            calls: Calls of every unit in this pulse.

        Returns:
            The stages; serially, a no-op.
        """
        return _NO_STAGES

    def observer(self, session: "ExecutionSession") -> "ExecutionObserver":
        """Return the observer the run's callbacks go to."""
        return session.observer

    def error_handler(self, session: "ExecutionSession") -> Optional["ErrorHandler"]:
        """Return the error handler the run's step failures go to."""
        return session.error_handler

    def callbacks(self) -> ContextManager[Any]:
        """Return the lock serializing user callbacks (handlers); serially none."""
        return nullcontext()

    def checkpoint(self) -> None:
        """Raise ``RunAborted`` once the coordination aborts; serially never."""

    def abort(self, cause: Optional[BaseException] = None) -> None:
        """Abort the run's gates; serially nothing waits, so nothing happens.

        Args:
            cause: The failure that aborts the run; ``None`` for a cancel.
        """

    @property
    def aborted(self) -> bool:
        """Whether ``abort`` was called."""
        return False

    @property
    def abort_cause(self) -> Optional[BaseException]:
        """The first abort's cause; ``None`` when not aborted or cancelled."""
        return None


SERIAL = Coordination()
"""The coordination of every serial run."""


class PipelinedCoordination(Coordination):
    """Stage gates and serialized callbacks of one pipelined run.

    Every observer, error-handler and handler callback of the run runs under
    one reentrant lock (``callbacks()``), so user callbacks never run
    concurrently with each other. A user observer is therefore coupled to the
    handlers: while a handler runs, other workers wait at their next step
    callback. The built-in no-op observer (``ExecutionObserver`` itself) has
    nothing to serialize and is passed through, so without an observer a
    slow handler holds back only deliveries, not other pulses' compute.
    Gates are created on first use by name.

    Args:
        session: Session whose block instances the run shares.
        options: The run's pipeline options.
    """

    pipelined = True

    def __init__(self, session: "ExecutionSession", *, options: PipelineOptions):
        self.session = session
        self.options = options
        self.counters = PipelineCounters()
        self._aborted = threading.Event()
        self._abort_cause: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._gates: Dict[str, StageGate] = {}
        self._callbacks = threading.RLock()
        self._observer = _serialized_observer(session.observer, lock=self._callbacks)
        self._error_handler = (
            _serialized(session.error_handler, lock=self._callbacks)
            if session.error_handler is not None
            else None
        )

    def gate(self, name: str) -> StageGate:
        """Return the gate of a stage, created once per name."""
        with self._lock:
            gate = self._gates.get(name)
            if gate is None:
                gate = StageGate(name, aborted=self._aborted, counters=self.counters)
                self._gates[name] = gate

        return gate

    def step_stages(
        self, run: "RunState", step: "PlannedStep", *, calls: int
    ) -> StepStages:
        selector = format_step_path(step.path)
        gates = {unit: f"{selector}#{unit}" for unit in step_stage_units(step)}
        stages = self.stages(run, gates, calls=calls)

        return stages

    def stages(
        self, run: "RunState", gates: Mapping[str, str], *, calls: int
    ) -> StepStages:
        if run.ticket is None:
            raise ContractError(
                f"run {run.run_id} executes in a pipeline without a ticket"
            )

        stages = StepStages(
            {unit: self.gate(name) for unit, name in gates.items()},
            ticket=run.ticket,
            calls=calls,
        )

        return stages

    def observer(self, session: "ExecutionSession") -> "ExecutionObserver":
        return self._observer

    def error_handler(self, session: "ExecutionSession") -> Optional["ErrorHandler"]:
        return self._error_handler

    def callbacks(self) -> ContextManager[Any]:
        return self._callbacks

    def checkpoint(self) -> None:
        if self._aborted.is_set():
            raise RunAborted("the run is aborting")

    def abort(self, cause: Optional[BaseException] = None) -> None:
        """Abort: every current and later wait raises ``RunAborted``.

        Calls already running finish; they cannot be interrupted. The first
        call's cause is kept; it is recorded before any waiter wakes.

        Args:
            cause: The failure that aborts the run; ``None`` for a cancel.
                A ``RunAborted`` (a waiter that was itself aborted) is not
                recorded as the cause.
        """
        with self._lock:
            if not self._aborted.is_set():
                if not isinstance(cause, RunAborted):
                    self._abort_cause = cause
                self._aborted.set()
            gates = list(self._gates.values())
        for gate in gates:
            gate.wake()

    @property
    def aborted(self) -> bool:
        return self._aborted.is_set()

    @property
    def abort_cause(self) -> Optional[BaseException]:
        with self._lock:
            cause = self._abort_cause

        return cause


def _serialized_observer(
    observer: "ExecutionObserver", *, lock: threading.RLock
) -> "ExecutionObserver":
    """Wrap a user observer; pass the built-in no-op observer through."""
    # Local import: plan imports this package (pipelining.options).
    from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver

    if type(observer) is ExecutionObserver:
        return observer

    serialized = _SerializedObserver(observer, lock=lock)

    return serialized


class _SerializedObserver:
    """Observer proxy calling every ``on_*`` callback under one lock."""

    def __init__(self, observer: "ExecutionObserver", *, lock: threading.RLock):
        self._observer = observer
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._observer, name)
        if name.startswith("on_") and callable(attribute):
            serialized = _serialized(attribute, lock=self._lock)
            return serialized

        return attribute


def _serialized(callback: Callable[..., Any], *, lock: threading.RLock) -> Callable:
    def call_serialized(*args: Any, **kwargs: Any) -> Any:
        with lock:
            result = callback(*args, **kwargs)

        return result

    return call_serialized
