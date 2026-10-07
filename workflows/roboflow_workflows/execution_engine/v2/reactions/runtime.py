"""Reaction runtime: handlers, signals, system and state machine events.

One runtime serves one active run (``ActiveRun``) or, for synchronous
handlers and fixed transitions only, one passive session. Each handler runs its
own compiled passive plan in its own persistent session
(``session.handler_sessions``),
serially: one event at a time, on a coordination of its own, never under
the parent's stage tickets or callback lock::

    sync handler     emit() -> wait for the handler's lock -> run -> report
                     -> return (raise ReactionError on failure); the emitting
                     step keeps its turn meanwhile (accepted backpressure)
    async handler    emit() -> admit to the handler's line -> return
                     one worker thread, started at the first admission,
                     runs the line in arrival order

The line of an asynchronous handler holds queued events (at most
``max_depth``) and, under ``synchronous`` overflow, placeholders of emitters
waiting to run their event themselves. Example with ``max_depth=1``::

    E0 running on the worker, E1 queued, E2 arrives      line = [E1, E2*]
    E0 finishes -> worker runs E1                       line = [E2*]
    E1 finishes -> E2's emitter runs E2 on its thread   line = []
    E3 arriving meanwhile queues behind E2*, so order stays E0 E1 E2 E3

``leaky`` overflow drops the oldest queued event instead (counted, reported,
released at once) and queues the newest. Only one execution of a handler
runs at a time, worker or emitter.

Every event, whatever its origin, goes through ``dispatch.publish``: fixed
state machine transitions first (``reactions.machines``; an applied
transition's machine event publishes inline, recursively), then handlers.
Handler runs receive this runtime, so their ``state_machine_set`` steps apply
handler-selected transitions, whose events publish the same way. The
compiler rejects cyclic automatic cascades, so every cascade is finite.

Lifecycle of an active run::

    open        pulses, signals, system and machine events are accepted
    stop        stop_ingress(): signal() rejects; everything else continues
    drain       (after the pulses and ``ended``) ingress closed, then wait
                until the in-flight count is 0, then seal (closed)
    closed      lines finish, workers exit; any emission is late ingress
    cancelled   queued events discarded, waiting emitters and drain woken
    settle      (after drain or cancel) wait until the in-flight count is 0

The in-flight count covers every admitted asynchronous event (queued,
running or run by its waiting emitter) and every running ``signal()``. New
work comes only from pulses (finished before the drain), signals (rejected
by then) and in-flight work, which counts itself before it finishes, so 0
is final. The count's condition never covers user code or callbacks.

Lifecycle writes and admission decisions hold the count's condition, so no
work is admitted once ``drain()`` sealed or ``cancel()`` returned. Checks on
the dispatch path read the state without it: cancellation is cooperative.
``cancel()`` discards every event that has not run (``discarded``: queued
ones and those of waiting emitters) and wakes waiting emitters with
``RunAborted``, the engine's cancellation signal, so their
pulse is cancelled rather than failed. Work already accepted is never
interrupted: a running handler, a running ``signal()`` on its caller's
thread, an emitter running its own event. It finishes and leaves the count;
``settle()`` waits for that, so the owner releases nothing it still uses,
and ``join(timeout)`` can still report a worker running. Emissions
after the seal raise ``EventEmissionError`` (late ingress), after
``cancel()`` ``RunAborted``. Async handler failures are counted and reported;
the run continues (the M5.1 policy).

Reports (``ReactionOutcome``, observer, handler output groups) are made
outside every queue lock and serialized by the runtime's own callback lock.
The outcome log is a bounded ring of metadata; no payload, result or
exception is retained after its report.
"""

import dataclasses
import importlib
import itertools
import threading
import time
import traceback
import weakref
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    EventEmissionError,
    ReactionCycleError,
    ReactionError,
    StepExecutionError,
    StepPath,
)
from roboflow_workflows.execution_engine.v2.observer import (
    NULL_REACTION_OBSERVER,
    ReactionObserver,
)
from roboflow_workflows.execution_engine.v2.pipelining.stages import RunAborted
from roboflow_workflows.execution_engine.v2.reactions.dispatch import (
    EventCause,
    publish,
)
from roboflow_workflows.execution_engine.v2.reactions.plan import (
    SYSTEM_EVENTS,
    EventOrigin,
    scoped_name,
)
from roboflow_workflows.execution_engine.v2.reactions.snapshots import Snapshot

__all__ = [
    "DEFAULT_OUTCOME_LIMIT",
    "HandlerCounters",
    "ReactionOutcome",
    "ReactionRuntime",
    "handler_selector",
    "session_reactions",
]

EXECUTION_MODULE = "roboflow_workflows.execution_engine.v2.execution"
"""Runs handler plans; imported on first use to keep imports acyclic."""

MACHINES_MODULE = "roboflow_workflows.execution_engine.v2.reactions.machines"
"""State machine records; imported only by plans with machines (it loads state)."""

DEFAULT_OUTCOME_LIMIT = 256
"""Outcomes one runtime keeps for ``outcomes()``; older ones are forgotten."""

OutcomeStatus = Literal["completed", "failed", "dropped", "discarded"]

_OPEN, _DRAINING, _CLOSED, _CANCELLED = "open", "draining", "closed", "cancelled"


def handler_selector(path: StepPath) -> str:
    """Selector naming a handler in counters, outcomes and errors.

    Args:
        path: Handler path, e.g. ``("child", "notify")``.

    Returns:
        ``"$handlers.child/notify"``.
    """
    selector = "$handlers." + "/".join(path)

    return selector


@dataclass
class HandlerCounters:
    """What happened to the events of one handler.

    Totals: ``emitted`` events reached the handler; each then ends
    ``completed``, ``failed``, ``dropped`` (leaky overflow) or ``discarded``
    (cancelled before it ran: queued, or its emitter still waiting for the
    turn). ``overflow_sync`` counts events their emitter
    ran itself under ``synchronous`` overflow. Gauges: ``pending`` queued now
    (payloads retained, at most ``max_depth``), ``running`` executing now (0
    or 1), ``blocked`` emitters waiting under ``synchronous`` overflow. Peak:
    ``max_pending``. Results are never retained: they are delivered or
    dropped when reported.

    Cost on emitting threads: ``inline`` runs there (every sync run plus
    ``overflow_sync``), taking ``inline_seconds`` in total and at most
    ``inline_max_seconds`` once. A run's time is the emitter's wait for its
    turn (the sync handler's lock, or the earlier events of the line under
    ``synchronous`` overflow) plus the handler's execution. It excludes fixed
    state machine transitions, payload snapshots and reports (observer,
    handler output groups). An emitter cancelled while it waits adds no
    ``inline`` run and no time; its event is ``discarded``.

    Async payload copies: ``snapshots`` taken, ``snapshot_bytes`` copied into
    arrays/tensors/byte arrays, and ``snapshot_unknown_sizes`` copies whose
    size is unknown (not in the bytes).
    """

    emitted: int = 0
    completed: int = 0
    failed: int = 0
    dropped: int = 0
    discarded: int = 0
    overflow_sync: int = 0
    pending: int = 0
    running: int = 0
    blocked: int = 0
    max_pending: int = 0
    inline: int = 0
    inline_seconds: float = 0.0
    inline_max_seconds: float = 0.0
    snapshots: int = 0
    snapshot_bytes: int = 0
    snapshot_unknown_sizes: int = 0


@dataclass(frozen=True)
class ReactionOutcome:
    """How one event ended for one handler (metadata only).

    Args:
        handler: Handler selector, e.g. ``$handlers.notify``.
        cause: The event.
        status: ``completed``, ``failed``, ``dropped`` or ``discarded``.
        inline: Whether it ran on the emitting thread (a sync handler or
            ``synchronous`` overflow).
        error: ``"TypeName: message"`` for ``failed``.
        step: Failing step inside the handler workflow, when known.
    """

    handler: str
    cause: EventCause
    status: OutcomeStatus
    inline: bool = False
    error: Optional[str] = None
    step: Optional[StepPath] = None


def _reject(state: str, *, cause: EventCause, selector: str) -> None:
    """Refuse an emission: cancellation aborts the pulse, closing is an error."""
    if state == _CANCELLED:
        raise RunAborted(f"{cause.origin_selector} -> {selector}: reactions cancelled")

    raise EventEmissionError(
        f"{cause.origin_selector} cannot reach {selector}: the run's reactions "
        "closed (an emission after the run finished)"
    )


def _clear_frames(error: BaseException) -> None:
    """Release locals (payloads) held by a reported error's traceback chain.

    Tracebacks form reference cycles with their frames, which would keep a
    failed event's payload alive until the cyclic collector runs. The text
    of the traceback stays printable.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        error = error.__cause__ or error.__context__


class _Entry:
    """One event in a handler's line; ``inline`` marks a waiting emitter."""

    __slots__ = ("cause", "fields", "inline", "cancelled")

    def __init__(self, cause: EventCause, fields: Mapping[str, Any], *, inline: bool):
        self.cause = cause
        self.fields: Optional[Mapping[str, Any]] = fields
        self.inline = inline
        self.cancelled = False


class _Handler:
    """Per-handler state: counters, exclusivity and, if async, the line."""

    def __init__(self, runtime: "ReactionRuntime", handler: Any):
        self.runtime = runtime
        self.handler = handler
        self.selector = handler_selector(handler.path)
        self.counters = HandlerCounters()
        self.condition = threading.Condition()
        self.runs = itertools.count()
        # Synchronous handlers: one execution at a time; the owner detects reentry.
        self.lock = threading.Lock()
        self.owner: Optional[int] = None
        # Asynchronous handlers only.
        self.line: Deque[_Entry] = deque()
        self.running = False
        self.state = _OPEN
        self.thread: Optional[threading.Thread] = None

    # Synchronous ----------------------------------------------------------

    def run_sync(self, cause: EventCause, fields: Mapping[str, Any]) -> None:
        if self.owner == threading.get_ident():
            raise ReactionCycleError(
                "the handler is already running on this thread; an event chain "
                "re-entered it",
                emitter=cause.emitter,
                event=cause.event,
                handler=self.selector,
            )

        # Timed from before the lock: waiting for another emitter's run counts.
        started = time.perf_counter()
        with self.condition:
            self.counters.emitted += 1
        with self.lock:
            self.owner = threading.get_ident()
            try:
                with self.condition:
                    self.counters.running = 1
                result, error = self.runtime.execute(self.handler, cause, fields)
                with self.condition:
                    self.counters.running = 0
                    self._count_inline(time.perf_counter() - started)
                    self._count_finished(error)
            finally:
                self.owner = None
        self.runtime.report(self, cause, result=result, error=error, inline=True)
        if error is not None:
            raise ReactionError(
                f"handler failed with {type(error).__name__}: {error}",
                emitter=cause.emitter,
                event=cause.event,
                handler=self.selector,
                step_path=getattr(error, "step_path", None),
            ) from error

    # Asynchronous ---------------------------------------------------------

    def admit(self, cause: EventCause, snapshot: Snapshot) -> None:
        """Queue the event, drop the oldest (leaky) or run it here in turn."""
        policy = self.handler.queue
        fields = snapshot.fields
        dropped: Optional[_Entry] = None
        # Counted in flight until the event ran, was dropped or discarded.
        self.runtime.enter(cause, selector=self.selector)
        try:
            with self.condition:
                self._check_open(cause)
                self.counters.emitted += 1
                self.counters.snapshots += 1
                self.counters.snapshot_bytes += snapshot.known_bytes
                self.counters.snapshot_unknown_sizes += snapshot.unknown_sizes
                if self.counters.pending < policy.max_depth:
                    self._queue(_Entry(cause, fields, inline=False))
                    return
                if policy.overflow == "leaky":
                    dropped = self._drop_oldest()
                    self._queue(_Entry(cause, fields, inline=False))
                else:
                    waited_since = time.perf_counter()
                    entry = self._wait_turn(cause, fields)
        except BaseException:
            self.runtime.leave()
            raise
        if dropped is not None:
            self._release(dropped, status="dropped")
            return

        self._run_entry(entry, inline_since=waited_since)

    def close(self) -> None:
        with self.condition:
            if self.state == _OPEN:
                self.state = _CLOSED
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            if self.state == _CANCELLED:
                return
            self.state = _CANCELLED
            # Every entry still in the line never ran: queued events and the
            # events of emitters waiting under synchronous overflow alike.
            discarded = list(self.line)
            for entry in discarded:
                entry.cancelled = True
            self.line.clear()
            self.counters.pending = 0
            self.counters.discarded += len(discarded)
            self.condition.notify_all()
        for entry in discarded:
            self._release(entry, status="discarded")

    def join(self, timeout: Optional[float]) -> bool:
        thread = self.thread
        if thread is None or thread is threading.current_thread():
            return True

        thread.join(timeout)
        joined = not thread.is_alive()

        return joined

    def _check_open(self, cause: EventCause) -> None:
        if self.state == _OPEN:
            return

        _reject(self.state, cause=cause, selector=self.selector)

    def _queue(self, entry: _Entry) -> None:
        """Append a queued event and make sure the worker runs (condition held)."""
        self.line.append(entry)
        self.counters.pending += 1
        self.counters.max_pending = max(
            self.counters.max_pending, self.counters.pending
        )
        self._ensure_worker(entry)
        self.condition.notify_all()

    def _drop_oldest(self) -> _Entry:
        """Remove the oldest queued event; leaky lines hold no waiting emitters."""
        dropped = self.line.popleft()
        self.counters.pending -= 1
        self.counters.dropped += 1

        return dropped

    def _wait_turn(self, cause: EventCause, fields: Mapping[str, Any]) -> _Entry:
        """Line up behind every earlier event, then take the turn (condition held)."""
        entry = _Entry(cause, fields, inline=True)
        self.line.append(entry)
        self.counters.overflow_sync += 1
        self.counters.blocked += 1
        self.condition.notify_all()
        try:
            while not entry.cancelled and not (
                not self.running and self.line and self.line[0] is entry
            ):
                self.condition.wait()
        finally:
            self.counters.blocked -= 1
        if entry.cancelled:
            entry.fields = None
            _reject(_CANCELLED, cause=cause, selector=self.selector)

        self.line.popleft()
        self.running = True
        self.counters.running = 1

        return entry

    def _ensure_worker(self, entry: _Entry) -> None:
        if self.thread is not None:
            return

        thread = threading.Thread(
            target=self._work,
            name=f"workflows-v2-reaction-{self.selector}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as error:
            self.line.remove(entry)
            self.counters.pending -= 1
            raise EventEmissionError(
                f"{self.selector}: its worker thread could not start: {error}"
            ) from error
        self.thread = thread

    def _work(self) -> None:
        if self.runtime.owned is not None:
            self.runtime.owned.mark()
        while True:
            with self.condition:
                while not (self.line and not self.running and not self.line[0].inline):
                    if self.state != _OPEN and not self.line:
                        return
                    self.condition.wait()
                entry = self.line.popleft()
                self.counters.pending -= 1
                self.running = True
                self.counters.running = 1
            self._run_entry(entry, inline_since=None)
            entry = None

    def _run_entry(self, entry: _Entry, *, inline_since: Optional[float]) -> None:
        """Execute, report while still holding the turn, then pass it on.

        ``inline_since`` is when a waiting emitter started waiting for its
        turn; ``None`` on the worker.
        """
        inline = inline_since is not None
        fields, entry.fields = entry.fields, None
        try:
            result, error = self.runtime.execute(self.handler, entry.cause, fields)
            fields = None
            with self.condition:
                if inline:
                    self._count_inline(time.perf_counter() - inline_since)
                self._count_finished(error)
            self.runtime.report(
                self, entry.cause, result=result, error=error, inline=inline
            )
            if error is not None:
                _clear_frames(error)
        finally:
            with self.condition:
                self.running = False
                self.counters.running = 0
                self.condition.notify_all()
            self.runtime.leave()

    def _count_inline(self, seconds: float) -> None:
        self.counters.inline += 1
        self.counters.inline_seconds += seconds
        self.counters.inline_max_seconds = max(
            self.counters.inline_max_seconds, seconds
        )

    def _count_finished(self, error: Optional[BaseException]) -> None:
        if error is None:
            self.counters.completed += 1
        else:
            self.counters.failed += 1

    def _release(self, entry: _Entry, *, status: OutcomeStatus) -> None:
        """Forget a dropped or discarded event's payload, then report it.

        A waiting emitter's entry stays in flight until that emitter wakes
        and raises ``RunAborted``; it leaves the count there.
        """
        entry.fields = None
        try:
            self.runtime.report(
                self, entry.cause, result=None, error=None, status=status
            )
        finally:
            if not entry.inline:
                self.runtime.leave()


class ReactionRuntime:
    """Reactions of one active run, or the synchronous ones of a session.

    Args:
        plan: The emitting plan; its ``reactions`` lists handlers, signals
            and state machines.
        sessions: Persistent session per handler path.
        session_id: Session the events belong to.
        managed_state: State service holding machine records; required when
            the plan has state machines.
        observer: Receives one ``on_reaction_finished`` per outcome.
        owned: Threads of the active run; handler workers mark themselves
            owned, so ``ActiveRun.wait()`` from a handler raises. ``None``
            for a passive session.
        active_run_id: Identity of the active run; ``None`` when passive.
        deliver: Delivers a handler output group's result; ``None`` when no
            group has a callback.
        fail: Records a failed callback as the run's failure; ``None`` raises
            it instead (passive).
        outcome_limit: Outcomes kept for ``outcomes()``.

    Raises:
        ContractError: When ``active_run_id`` is ``None`` and the plan has
            an asynchronous handler, which needs an active run's lifecycle,
            or the plan has state machines but no ``managed_state``.
    """

    def __init__(
        self,
        plan: Any,
        *,
        sessions: Mapping[StepPath, Any],
        session_id: str = "",
        managed_state: Optional[Any] = None,
        observer: Optional[ReactionObserver] = None,
        owned: Optional[Any] = None,
        active_run_id: Optional[str] = None,
        deliver: Optional[Callable[[str, Any], None]] = None,
        fail: Optional[Callable[[BaseException, Optional[str]], None]] = None,
        outcome_limit: int = DEFAULT_OUTCOME_LIMIT,
    ):
        reactions = plan.reactions
        asynchronous = [h for h in reactions.handlers if h.mode == "async"]
        if active_run_id is None and asynchronous:
            raise ContractError(
                f"{handler_selector(asynchronous[0].path)} is asynchronous; "
                "asynchronous handlers need an active run (session.start)"
            )

        if reactions.machines and managed_state is None:
            raise ContractError(
                "The workflow declares state machines, but its session has no "
                "managed_state resource to hold their records"
            )

        self.plan = plan
        self.owned = owned
        self._reactions = reactions
        self._sessions = sessions
        self._session_id = session_id
        self._observer = observer if observer is not None else NULL_REACTION_OBSERVER
        self._active_run_id = active_run_id
        self._deliver = deliver
        self._fail = fail
        self._sequence = itertools.count()
        self._callbacks = threading.Lock()
        self._reporter: Optional[int] = None
        self._deferred: List[tuple] = []
        self._history = threading.Lock()
        self._outcomes: Deque[ReactionOutcome] = deque(maxlen=outcome_limit)
        # Lifecycle and in-flight count; never held while user code runs.
        # Every write of _state/_ingress/_in_flight, and every admission or
        # seal decision, holds _admission. dispatch() reads _state without it:
        # a cooperative cancellation check, not a guarantee. Calls already
        # accepted may still run after cancel(); settle() waits for them.
        self._admission = threading.Condition()
        self._state = _OPEN
        self._ingress = True
        self._in_flight = 0
        self._handlers: Dict[StepPath, _Handler] = {
            handler.path: _Handler(self, handler) for handler in reactions.handlers
        }
        self._machines = (
            importlib.import_module(MACHINES_MODULE).MachineRuntime(
                reactions, managed_state, emit=self._publish_machine_event
            )
            if reactions.machines
            else None
        )
        self._execution = importlib.import_module(EXECUTION_MODULE)

    @property
    def counters(self) -> Mapping[str, HandlerCounters]:
        """A consistent copy of every handler's counters, by selector."""
        copies = {}
        for state in self._handlers.values():
            with state.condition:
                copies[state.selector] = dataclasses.replace(state.counters)

        return copies

    def outcomes(self, limit: Optional[int] = None) -> Tuple[ReactionOutcome, ...]:
        """Return the most recent outcomes, oldest first.

        Args:
            limit: At most this many; all kept ones when ``None``.

        Returns:
            Recent outcomes, at most ``outcome_limit`` in total.
        """
        with self._history:
            kept = tuple(self._outcomes)
        recent = kept if limit is None else kept[max(len(kept) - limit, 0) :]

        return recent

    def machine_state(
        self, machine: str, *, source_id: Optional[str] = None
    ) -> Tuple[str, int]:
        """Read one state machine instance.

        Args:
            machine: Scoped machine name, e.g. ``"gate"`` or ``"child/gate"``.
            source_id: Source of a per-source machine; ``None`` for a global one.

        Returns:
            ``(state, version)``; version 0 before the first transition.

        Raises:
            ContractError: For an unknown machine or a plan without machines.
            StateScopeError: For a missing or unexpected ``source_id``.
        """
        if not isinstance(machine, str) or not machine or "" in machine.split("/"):
            raise ContractError(
                f"State machine name must be a scoped name like 'gate' or "
                f"'child/gate', got {machine!r}"
            )
        if self._machines is None:
            raise ContractError(
                f"Unknown state machine {machine!r}: the workflow declares none"
            )

        current = self._machines.current(tuple(machine.split("/")), source_id=source_id)

        return current

    def machine_counters(self) -> Mapping[str, Any]:
        """Applied, ignored and stale attempts per ``<machine>.<transition>``.

        Returns:
            An independent snapshot; empty without state machines.
        """
        if self._machines is None:
            return {}

        counters = self._machines.counters()

        return counters

    # Emit path (reactions.dispatch.Reactions) ------------------------------

    def subscribed(self, origin: EventOrigin) -> bool:
        """Whether a handler or a fixed transition listens to ``origin``."""
        return self._reactions.subscribed(origin)

    def bound_fields(self, origin: EventOrigin) -> FrozenSet[str]:
        """Fields handlers bind and fixed transitions copy, for ``origin``."""
        return self._reactions.bound_fields(origin)

    def handlers_for(self, origin: EventOrigin) -> Sequence[Any]:
        """Handlers subscribed to ``origin``, in declaration order."""
        return self._reactions.handlers_for(origin)

    def next_sequence(self) -> int:
        """Next emission number of this runtime."""
        sequence = next(self._sequence)

        return sequence

    def dispatch(
        self,
        handlers: Sequence[Any],
        cause: EventCause,
        *,
        fields: Mapping[str, Any],
        snapshots: Mapping[Any, Snapshot],
    ) -> None:
        """Apply fixed transitions, then run or admit handlers, in order.

        Each machine applies at most one fixed transition per event; its
        machine event is published before the next machine and before the
        handlers of this event run. Synchronous handlers run now, on this
        thread; asynchronous ones are admitted.

        Args:
            handlers: Subscribed handlers.
            cause: The emitted event.
            fields: Ready demanded field values (handlers and transitions).
            snapshots: Owned field snapshot per async handler path.

        Raises:
            EventEmissionError: After the seal (late ingress).
            RunAborted: After ``cancel()``, also for an emitter waiting under
                ``synchronous`` overflow when it happens.
            ReactionError: When a synchronous handler failed; later handlers
                of this emission do not run. Applied transitions stay.
            StateError: When a machine record cannot be read or updated.
        """
        # Unlocked reads: cooperative checks between subscribers (see _admission).
        if self._state in (_CLOSED, _CANCELLED):
            _reject(self._state, cause=cause, selector="its subscribers")
        if self._machines is not None:
            self._machines.on_event(cause.origin, cause, fields)
        for handler in handlers:
            state = self._handlers[handler.path]
            if handler.mode == "async":
                state.admit(cause, snapshots[handler.path])
                continue
            if self._state in (_CLOSED, _CANCELLED):
                _reject(self._state, cause=cause, selector=state.selector)
            state.run_sync(cause, {name: fields[name] for name in handler.bound_fields})

    # Ingress -------------------------------------------------------------------

    def signal(
        self,
        name: str,
        fields: Mapping[str, Any],
        *,
        sample: Optional[SampleContext],
        temporal: Optional[TemporalContext],
    ) -> None:
        """Publish one external signal on the calling thread.

        Accepted only while ingress is open; an accepted signal counts as in
        flight until its synchronous work returned, so a drain waits for it.

        Args:
            name: Declared signal name.
            fields: Every declared field of the signal.
            sample: Source context of the signal, or ``None``.
            temporal: Temporal context of the signal, or ``None``.

        Raises:
            EventEmissionError: For an undeclared signal, a source machine
                transition without a source, or closed ingress.
            EventPayloadError: For missing, unknown or ill-kinded fields.
            ReactionError: When a synchronous handler failed.
        """
        planned = self._reactions.signals.get(name)
        if planned is None:
            declared = sorted(self._reactions.signals) or "no signals"
            raise EventEmissionError(
                f"Signal {name!r} is not declared; the workflow declares {declared}"
            )
        planned.event.check_payload(name, fields)
        origin = planned.origin
        if sample is None:
            self._require_source(origin)
        with self._admission:
            if not self._ingress or self._state != _OPEN:
                raise EventEmissionError(
                    f"{origin.selector} rejected: the run is stopping, cancelled "
                    "or finished"
                )
            self._in_flight += 1
        try:
            if self._reactions.subscribed(origin):
                publish(
                    self,
                    origin,
                    declared=planned.event,
                    fields=fields,
                    where=origin.selector,
                    session_id=self._session_id,
                    run_id=self._active_run_id,
                    pulse=None,
                    index=(),
                    sample=sample,
                    temporal=temporal,
                    parent=None,
                )
        finally:
            self.leave()

    def system(self, event: str) -> None:
        """Publish a run lifecycle event (``started`` or ``ended``) here.

        System events carry no fields and no source.

        Args:
            event: ``"started"`` or ``"ended"``.

        Raises:
            ReactionError: When a synchronous handler failed.
            StateError: When a machine record cannot be read or updated.
        """
        origin = EventOrigin(kind="system", event=event)
        if not self._reactions.subscribed(origin):
            return

        publish(
            self,
            origin,
            declared=SYSTEM_EVENTS[event],
            fields={},
            where=origin.selector,
            session_id=self._session_id,
            run_id=self._active_run_id,
            pulse=None,
            index=(),
            sample=None,
            temporal=None,
            parent=None,
        )

    def set_machine_state(
        self,
        *,
        handler: StepPath,
        cause: EventCause,
        machine: str,
        transition: str,
        next_state: str,
    ) -> Any:
        """Apply a handler-selected transition for a running handler.

        Args:
            handler: Path of the handler whose step calls the setter.
            cause: The event that handler run handles.
            machine: Machine name relative to the handler's scope.
            transition: Transition name.
            next_state: Requested target state.

        Returns:
            The ``TransitionResult``; an applied transition's machine event
            is published before this returns.

        Raises:
            ContractError: Without machines, or an unauthorized transition
                or target.
            StateError: When the machine record cannot be read or updated.
        """
        if self._machines is None:
            raise ContractError(
                f"Handler '{scoped_name(handler)}' sets state machine "
                f"{machine!r}, but the workflow declares no state machines"
            )

        result = self._machines.set_state(
            handler=handler,
            cause=cause,
            machine_ref=machine,
            transition=transition,
            next_state=next_state,
        )

        return result

    def _publish_machine_event(
        self,
        origin: EventOrigin,
        fields: Mapping[str, Any],
        *,
        parent: EventCause,
        stamp: Any,
    ) -> None:
        """``MachineRuntime`` emit callback: the event inherits its parent's context."""
        if not self._reactions.subscribed(origin):
            return

        publish(
            self,
            origin,
            declared=self._reactions.declared_event(origin),
            fields=fields,
            where=f"{origin.selector} (transition '{stamp.transition}')",
            session_id=parent.session_id,
            run_id=parent.run_id,
            pulse=parent.pulse,
            index=parent.index,
            sample=parent.sample,
            temporal=parent.temporal,
            parent=parent,
            stamp=stamp,
        )

    def _require_source(self, origin: EventOrigin) -> None:
        for transition in self._reactions.transitions_for(origin):
            if self._reactions.machine(transition.machine).scope == "source":
                raise EventEmissionError(
                    f"{origin.selector} has no source, but transition "
                    f"'{transition.label}' belongs to a per-source state machine; "
                    "pass source_id= or sample="
                )

    # In-flight accounting --------------------------------------------------

    def enter(self, cause: EventCause, *, selector: str) -> None:
        """Count one admitted event in flight, unless sealed or cancelled."""
        with self._admission:
            if self._state in (_CLOSED, _CANCELLED):
                _reject(self._state, cause=cause, selector=selector)
            self._in_flight += 1

    def leave(self) -> None:
        """One in-flight unit finished; wake the drain at zero."""
        with self._admission:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._admission.notify_all()

    # Lifecycle -------------------------------------------------------------

    def stop_ingress(self) -> None:
        """Reject new ``signal()`` calls; admitted ones and cascades continue."""
        with self._admission:
            self._ingress = False

    def drain(self) -> None:
        """Close ingress, wait for every in-flight event, then seal.

        Call once the main flow emits nothing more. Cascades of in-flight
        work are accepted until the count reaches 0. A ``cancel()`` meanwhile
        ends the wait; ``settle()`` then waits for the accepted work. The
        workers then finish and exit; ``join()`` them.
        """
        with self._admission:
            self._ingress = False
            if self._state == _OPEN:
                self._state = _DRAINING
            while self._in_flight and self._state == _DRAINING:
                self._admission.wait()
            if self._state == _DRAINING:
                self._state = _CLOSED
        for state in self._handlers.values():
            state.close()

    def close(self) -> None:
        """Seal now: reject new emissions; queued events still run. Never blocks."""
        with self._admission:
            self._ingress = False
            if self._state in (_OPEN, _DRAINING):
                self._state = _CLOSED
            self._admission.notify_all()
        for state in self._handlers.values():
            state.close()

    def cancel(self) -> None:
        """Reject new emissions, discard queued events, wake waiting emitters."""
        with self._admission:
            self._ingress = False
            self._state = _CANCELLED
            self._admission.notify_all()
        for state in self._handlers.values():
            state.cancel()

    def settle(self) -> None:
        """Wait until every admitted unit finished, also after ``cancel()``.

        Call after ``drain()`` or ``cancel()``: nothing is admitted then, so
        the count only falls. Unlike ``drain()``, cancellation does not end
        this wait; accepted work, e.g. a ``signal()`` on its caller's thread,
        keeps running until it returns.
        """
        with self._admission:
            while self._in_flight:
                self._admission.wait()

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait for every handler worker to finish its line.

        Call after ``close()`` or ``cancel()``.

        Args:
            timeout: Seconds per worker; forever when ``None``.

        Returns:
            ``True`` when every worker ended; ``False`` when one still runs
            (a running handler cannot be interrupted).
        """
        joined = all([state.join(timeout) for state in self._handlers.values()])

        return joined

    # Execution and reports ---------------------------------------------------

    def execute(
        self, handler: Any, cause: EventCause, fields: Mapping[str, Any]
    ) -> Tuple[Optional[Any], Optional[BaseException]]:
        """Run one event through the handler's plan; never raises."""
        metadata = EntryMetadata(
            sample={(): cause.sample}, temporal={(): cause.temporal}
        )
        inputs = {
            name: InputValue(value, metadata=metadata)
            for name, value in handler.inputs_for(fields).items()
        }
        try:
            result = self._execution.run_handler(
                self._sessions[handler.path],
                inputs=inputs,
                cause=cause,
                handler=handler.path,
                reactions=self,
            )
        except Exception as error:
            return None, error

        return result, None

    def report(
        self,
        state: _Handler,
        cause: EventCause,
        *,
        result: Optional[Any],
        error: Optional[BaseException],
        inline: bool = False,
        status: Optional[OutcomeStatus] = None,
    ) -> None:
        """Log the outcome, notify the observer and deliver handler groups.

        Callbacks are serialized by the callback lock. A report made from a
        callback on the reporting thread (one that cancels the run, which
        discards queued events) is delivered by the outer report right after
        that callback, so the lock is never re-entered. Failed callbacks reach
        the failure hook after the lock is released.
        """
        if status is None:
            status = "failed" if error is not None else "completed"
        outcome = ReactionOutcome(
            handler=state.selector,
            cause=cause,
            status=status,
            inline=inline,
            error=None if error is None else f"{type(error).__name__}: {error}",
            step=error.step_path if isinstance(error, StepExecutionError) else None,
        )
        groups = (
            self._reactions.groups_of(state.handler.path)
            if status == "completed" and self._deliver is not None
            else ()
        )
        with self._history:
            self._outcomes.append(outcome)
        pending = [(state, outcome, error, result, groups)]
        if self._reporter == threading.get_ident():
            # A callback of this report reported (e.g. it cancelled the run):
            # the outer report delivers it once the callback returned.
            self._deferred.extend(pending)
            return

        failures: List[Tuple[BaseException, Optional[str]]] = []
        with self._callbacks:
            self._reporter = threading.get_ident()
            try:
                while pending:
                    self._notify(*pending.pop(0), failures=failures)
                    pending.extend(self._deferred)
                    self._deferred.clear()
            finally:
                self._reporter = None
        for raised, group in failures:
            self._callback_failed(raised, group=group)

    def _notify(
        self,
        state: _Handler,
        outcome: ReactionOutcome,
        error: Optional[BaseException],
        result: Optional[Any],
        groups: Sequence[Any],
        *,
        failures: List[Tuple[BaseException, Optional[str]]],
    ) -> None:
        """Call the observer and group callbacks (callback lock held)."""
        try:
            self._observer.on_reaction_finished(
                outcome=outcome, error=error, result=result
            )
        except Exception as raised:
            failures.append((raised, None))
        for group in groups:
            delivered = self._group_result(
                state, group, cause=outcome.cause, result=result
            )
            try:
                self._deliver(group.name, delivered)
            except Exception as raised:
                failures.append((raised, group.name))

    def _callback_failed(self, raised: BaseException, *, group: Optional[str]) -> None:
        if self._fail is None:
            raise raised

        self._fail(raised, group)

    def _group_result(
        self, state: _Handler, group: Any, *, cause: EventCause, result: Any
    ) -> Any:
        """One handler run's outputs as a ``GroupResult`` of ``group``.

        Group fields rename handler outputs; entry keys follow output names,
        so every selected entry is re-keyed for its group field.
        """
        outputs = importlib.import_module(
            "roboflow_workflows.execution_engine.v2.execution.outputs"
        )
        data = importlib.import_module("roboflow_workflows.execution_engine.v2.data")
        plan_module = importlib.import_module(
            "roboflow_workflows.execution_engine.v2.plan"
        )
        plan = result.plan
        declared = {output.name: output for output in plan.outputs}
        originals = [declared[output] for output in group.fields.values()]
        fields = tuple(
            dataclasses.replace(original, name=name)
            for name, original in zip(group.fields, originals)
        )
        keys = {
            old.key: new.key
            for old, new in zip(
                outputs.selected_ports(plan, originals),
                outputs.selected_ports(plan, fields),
            )
        }
        buffer = result.outputs
        delivered = outputs.GroupResult(
            group=group.name,
            source=state.selector,
            pulse=plan_module.PulseKey(
                active_run_id=self._active_run_id,
                source=state.selector,
                sequence=next(state.runs),
            ),
            run_id=result.run_id,
            session_id=result.session_id,
            fields=fields,
            outputs=data.WorkflowsBuffer(
                lineage_id=buffer.lineage_id,
                pulse_id=buffer.pulse_id,
                data={
                    new: buffer.data[old]
                    for old, new in keys.items()
                    if old in buffer.data
                },
                layout={
                    new: buffer.layout[old]
                    for old, new in keys.items()
                    if old in buffer.layout
                },
                metadata={
                    new: buffer.metadata[old]
                    for old, new in keys.items()
                    if old in buffer.metadata
                },
            ),
            selections={
                name: {
                    selector: keys[key]
                    for selector, key in result.selections[original.name].items()
                }
                for name, original in zip(group.fields, originals)
            },
            statuses={new: result.statuses[old] for old, new in keys.items()},
            filtered_paths={
                new: result.filtered_paths[old] for old, new in keys.items()
            },
            plan=plan,
            input_row_count=result.input_row_count,
            causes=(cause.pulse,) if cause.pulse is not None else (),
        )

        return delivered


_SESSION_RUNTIMES: "weakref.WeakKeyDictionary[Any, ReactionRuntime]" = (
    weakref.WeakKeyDictionary()
)
_SESSION_RUNTIMES_LOCK = threading.Lock()


def session_reactions(session: Any) -> Optional[ReactionRuntime]:
    """The synchronous reaction runtime of a passive session, created once.

    Shared by every direct run and pipeline submission of the session, so a
    handler runs one event at a time across them.

    Args:
        session: A passive session.

    Returns:
        The runtime, or ``None`` when the plan has no handlers or machines.

    Raises:
        ContractError: When the plan has an asynchronous handler.
    """
    reactions = session.plan.reactions
    if not reactions.handlers and not reactions.machines:
        return None

    with _SESSION_RUNTIMES_LOCK:
        runtime = _SESSION_RUNTIMES.get(session)
        if runtime is None:
            runtime = ReactionRuntime(
                session.plan,
                sessions=session.handler_sessions,
                session_id=session.session_id,
                managed_state=getattr(session, "managed_state", None),
                observer=getattr(session, "reaction_observer", None),
            )
            _SESSION_RUNTIMES[session] = runtime

    return runtime
