"""One pulse of an active run, and the end of a domain; serial and pipelined alike.

The serial processor and the pipeline workers run the same code. Only the
run's ``Coordination`` differs: serially it gates nothing; pipelined it gives
each pulse its turn at every stage it visits (steps, group deliveries,
operator pushes), in sequence order per domain::

    run_pulse(key)                                 one thread, start to end
        state = begin_pulse(..., ticket=(domain, sequence))
        unactivated groups of the domain: retire their turn now
        for step in route(domain):                 execute_step takes its turns
            execute_step; deliver every group now ready: in its turn,
                record it if recorded, then its handler (callbacks lock)
        for operator in consumers_of(domain):      after the whole route
            turn at "$operators.<o>#push": push, number the returned pulses,
                count them outstanding             (one operator call at a time)
            run the returned pulses, depth first   (turn already released)
        abandon state; processed or cancelled; on_pulse_finished
        outstanding(domain) -= 1                   may end the domain

    end_domain(D)                                  once D is sealed and drained
        for operator in consumers_of(D):
            end_input(each input of D)             operator exclusive; run pulses
            every upstream ended -> finish once    operator exclusive; sealed
        an operator sealed with nothing outstanding ends in turn

A domain is *sealed* when it can produce no further pulse (its reader is
done, or its operator's ``finish`` returned) and *drained* when none of its
pulses is outstanding. It ends exactly once, when both hold, through
``schedule_end``: inline on the serial processor, as worker work in a
pipeline. Nothing here holds a gate or lock while running a pulse.
"""

import inspect
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.active.execution import (
    abandon_pulse,
    begin_operator_pulse,
    begin_pulse,
    group_result,
    operator_arrivals,
)
from roboflow_workflows.execution_engine.v2.context import use_pulse_run_id
from roboflow_workflows.execution_engine.v2.controls import ControlSnapshot
from roboflow_workflows.execution_engine.v2.data import Timestamp
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ActiveRunStage,
    ContractError,
    StepExecutionError,
    StepPath,
)
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.outputs import GroupResult
from roboflow_workflows.execution_engine.v2.execution.steps import (
    RunState,
    execute_step,
)
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorCounters,
    OperatorPulse,
    TerminationReason,
)
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    Coordination,
    PipelineCounters,
    PipelinedCoordination,
    RunAborted,
    StepStages,
)
from roboflow_workflows.execution_engine.v2.plan import (
    ExecutionSession,
    PlannedOperator,
    PlannedOutputGroup,
    PulseKey,
)
from roboflow_workflows.execution_engine.v2.sources import Emission

__all__ = [
    "DomainProgress",
    "OperatorSlot",
    "PulseExecutor",
    "Registered",
    "SourcePulse",
    "attributed",
    "close_operator",
]

GroupHandler = Callable[[GroupResult], None]
Counters = Any
"""``SourceCounters`` or ``OperatorCounters``: both have the pulse fields."""


@dataclass(frozen=True)
class SourcePulse:
    """An admitted emission of a source: its key, when it was read and its controls.

    Args:
        key: Identity of the pulse.
        emission: What the source emitted.
        observed: When the runtime read the emission.
        controls: Control snapshot taken at admission; the whole pulse, the
            operator pulses it feeds and its results use exactly this one.
    """

    key: PulseKey
    emission: Emission
    observed: Timestamp
    controls: ControlSnapshot


@dataclass(frozen=True)
class Registered:
    """An output group with its consumers and the steps its fields wait for.

    Args:
        group: The output group.
        handler: The host's callback; ``None`` for a group only recorded.
        prerequisites: Steps the group's fields need.
        recorder: Engine-owned capture of the group; it receives each result
            in the group's delivery turn, before the handler and outside the
            run-wide callbacks lock. ``None`` when the group is not recorded.
    """

    group: PlannedOutputGroup
    handler: Optional[GroupHandler]
    prerequisites: FrozenSet[StepPath]
    recorder: Optional[GroupHandler] = None

    @property
    def stage(self) -> str:
        """Name of the group's delivery stage."""
        return f"$groups.{self.group.name}#deliver"


@dataclass
class OperatorSlot:
    """One operator of the run: its instance and its lifecycle bookkeeping.

    Every field below ``instance`` is changed only while the operator's
    stage is held (its push turn, or exclusively for end and finish).
    """

    planned: PlannedOperator
    instance: Operator
    next_sequence: int = 0
    ended: Set[str] = field(default_factory=set)
    stopped: bool = False
    finish_called: bool = False
    close_attempted: bool = False
    error: Optional[ActiveRunError] = None

    @property
    def name(self) -> str:
        return self.planned.name

    @property
    def counters(self) -> OperatorCounters:
        return self.instance.counters

    @property
    def stage(self) -> str:
        """Name of the operator's stage; push, end_input and finish share it."""
        return f"$operators.{self.name}#push"


@dataclass(frozen=True)
class _Outcome:
    """How a pulse, or the operator work it fed, ended.

    Args:
        completed: Whether everything ran and every handler returned.
        error: The failure raised by this work, already recorded as the run's
            failure (or suppressed); ``None`` when it completed or a failure
            elsewhere cut it short.
    """

    completed: bool
    error: Optional[ActiveRunError] = None


_COMPLETED = _Outcome(completed=True)
_CANCELLED = _Outcome(completed=False)

_Emitted = List[Tuple[PulseKey, OperatorPulse]]
"""Pulses an operator call returned, numbered in its domain."""


class DomainProgress:
    """When each domain has ended: sealed and nothing outstanding, exactly once.

    ``outstanding`` counts admitted (sources) or returned (operators) pulses
    that have not completed, inline operator work included. It grows before
    the producer releases its own lock (a reader's admission, an operator's
    stage), so a domain is never seen drained while a pulse is on its way.
    """

    def __init__(self, domains: Sequence[str]):
        self._lock = threading.Lock()
        self._outstanding = dict.fromkeys(domains, 0)
        self._sealed: Dict[str, TerminationReason] = {}
        self._ended: Set[str] = set()

    def add(self, domain: str, count: int = 1) -> None:
        """Count ``count`` new pulses of ``domain`` outstanding."""
        with self._lock:
            self._outstanding[domain] += count

    def complete(self, domain: str) -> Optional[TerminationReason]:
        """One pulse completed; returns the reason when this ended the domain."""
        with self._lock:
            self._outstanding[domain] -= 1
            reason = self._end_if_drained(domain)

        return reason

    def seal(
        self, domain: str, reason: TerminationReason
    ) -> Optional[TerminationReason]:
        """No further pulse of ``domain``; returns the reason when it ended now."""
        with self._lock:
            self._sealed.setdefault(domain, reason)
            ended = self._end_if_drained(domain)

        return ended

    def _end_if_drained(self, domain: str) -> Optional[TerminationReason]:
        if (
            domain in self._ended
            or domain not in self._sealed
            or self._outstanding[domain]
        ):
            return None

        self._ended.add(domain)

        return self._sealed[domain]


def attributed(
    raised: Exception,
    *,
    stage: ActiveRunStage,
    source: Optional[str] = None,
    operator: Optional[str] = None,
    pulse: Optional[int] = None,
) -> ActiveRunError:
    """Wrap an exception of the run into its attributed terminal error."""
    if isinstance(raised, ActiveRunError):
        return raised
    if isinstance(raised, StepExecutionError):
        error = ActiveRunError(
            str(raised),
            stage="step",
            source=source,
            operator=operator,
            pulse=pulse,
            step_path=raised.step_path,
            phase=raised.phase,
        )
    else:
        error = ActiveRunError(
            f"{type(raised).__name__}: {raised}",
            stage=stage,
            source=source,
            operator=operator,
            pulse=pulse,
        )
    error.__cause__ = raised

    return error


def close_operator(slot: OperatorSlot) -> Optional[ActiveRunError]:
    """Close an operator exactly once; return what ``close`` raised."""
    if slot.close_attempted:
        return None

    slot.close_attempted = True
    try:
        slot.instance.close()
    except Exception as error:
        failure = ActiveRunError(
            f"close raised {type(error).__name__}: {error}",
            stage="operator",
            operator=slot.name,
        )
        failure.__cause__ = error
        return failure
    slot.counters.closed = True

    return None


class PulseExecutor:
    """Runs pulses and domain ends of one active run over the session's blocks.

    Args:
        session: Session whose block instances process the pulses.
        run_id: Identity of the active run.
        inputs: Static input entries shared by every pulse.
        registered: Output groups with handlers, in plan order.
        operators: Constructed operators by name, in plan order.
        sources: Declared source names (for attribution).
        coordination: ``SERIAL``, or the run's ``PipelinedCoordination``.
        fail: Records a failure as the run's (first) failure and aborts it.
        aborting: Whether the run failed or was cancelled.
        schedule_end: Ends a sealed, drained domain: inline when serial,
            as worker work when pipelined.
        reactions: The run's reaction runtime; ``None`` without handlers.
    """

    def __init__(
        self,
        session: ExecutionSession,
        *,
        run_id: str,
        inputs: Mapping[str, Entry],
        registered: Sequence[Registered],
        operators: Mapping[str, OperatorSlot],
        sources: FrozenSet[str],
        coordination: Coordination,
        fail: Callable[[ActiveRunError], None],
        aborting: Callable[[], bool],
        schedule_end: Callable[[str, TerminationReason], None],
        reactions: Optional[Any] = None,
    ):
        self.session = session
        self.reactions = reactions
        self.run_id = run_id
        self.coordination = coordination
        self.observer = coordination.observer(session)
        self.progress = DomainProgress([*sources, *operators])
        self._inputs = inputs
        self._registered = registered
        self.operators = operators
        self._sources = sources
        self._fail = fail
        self._aborting = aborting
        self._schedule_end = schedule_end
        self._counting = threading.Lock()
        self._gauges: Optional[PipelineCounters] = (
            coordination.counters
            if isinstance(coordination, PipelinedCoordination)
            else None
        )

    # Counters ---------------------------------------------------------------

    def count(self, counters: Counters, name: str, amount: int = 1) -> None:
        """Add to one counter field; workers and readers share the counters."""
        with self._counting:
            setattr(counters, name, getattr(counters, name) + amount)

    def _gauge(self, name: str, delta: int) -> None:
        if self._gauges is not None:
            self._gauges.add(name, delta)

    # Domains ----------------------------------------------------------------

    def seal(self, domain: str, reason: TerminationReason) -> None:
        """No further pulse of ``domain`` will start; end it once drained."""
        ended = self.progress.seal(domain, reason)
        if ended is not None:
            self._end_later(domain, ended)

    def _end_later(self, domain: str, reason: TerminationReason) -> None:
        if not self._aborting():
            self._schedule_end(domain, reason)

    def end_domain(
        self,
        domain: str,
        reason: TerminationReason,
        *,
        controls: ControlSnapshot,
    ) -> None:
        """Tell the operators of ``domain`` that its inputs ended; finish the done ones.

        An operator finishes once, by whichever upstream domain ends last;
        its domain ends after its final pulses and any pulse of it still
        running elsewhere. Stops at the first failure, which is recorded.
        ``controls`` is the snapshot the driver took when it sequenced this
        end; every pulse the end emits runs under it.
        """
        try:
            for planned in self.session.plan.consumers_of(domain):
                if self._aborting():
                    return
                slot = self.operators[planned.name]
                if not self._end_inputs(
                    slot, planned, domain, reason=reason, controls=controls
                ):
                    return
                if not self._finish_if_ended(slot, planned, controls=controls):
                    return
        except RunAborted:
            return

    def _end_inputs(
        self,
        slot: OperatorSlot,
        planned: PlannedOperator,
        domain: str,
        *,
        reason: TerminationReason,
        controls: ControlSnapshot,
    ) -> bool:
        inputs = planned.inputs_from(domain)
        for position, item in enumerate(inputs):
            self._drain_before_operator(controls)
            with self._exclusive(slot):
                slot.stopped = slot.stopped or reason == "stop"
                emitted = self._call_operator(slot, "end_input", item.name)
                if position == len(inputs) - 1:
                    slot.ended.add(domain)
            if not self._run_emissions(slot, emitted, controls=controls).completed:
                return False

        return True

    def _drain_before_operator(
        self, controls: ControlSnapshot, run: Optional[RunState] = None
    ) -> None:
        """Rule 4 of ``controls``: let older in-flight work enter the operator first.

        Work carrying reset epoch ``e`` waits until no work admitted under a
        version below ``e`` is in flight, so every operator pulse an older
        pulse emits is sequenced before this work's emissions and reaches a
        reset member before the reset. Holds no turn; raises ``RunAborted``
        when the run aborts meanwhile.
        """
        epoch = controls.reset_epoch
        in_flight = self.session.controls.in_flight
        if not epoch or in_flight.drained_below(epoch):
            return

        def aborted() -> Optional[BaseException]:
            if self._aborting():
                return RunAborted("the run is aborting while older work drains")
            return None

        if run is not None:
            run.record("reset_drain", epoch=epoch, in_flight=in_flight.describe())
        in_flight.wait_drained_below(epoch, aborted)

    def _finish_if_ended(
        self,
        slot: OperatorSlot,
        planned: PlannedOperator,
        *,
        controls: ControlSnapshot,
    ) -> bool:
        ended: Optional[TerminationReason] = None
        self._drain_before_operator(controls)
        with self._exclusive(slot):
            if slot.finish_called or not slot.ended.issuperset(
                planned.upstream_domains
            ):
                return True
            slot.finish_called = True
            final: TerminationReason = "stop" if slot.stopped else "eof"
            emitted = self._call_operator(slot, "finish", final)
            if emitted is not None:
                slot.counters.finished = True
                ended = self.progress.seal(slot.name, final)
        completed = self._run_emissions(slot, emitted, controls=controls).completed
        if ended is not None:
            self._end_later(slot.name, ended)

        return completed

    # Pulses -----------------------------------------------------------------

    def run_source_pulse(self, pulse: SourcePulse, *, counters: Counters) -> None:
        """Run one admitted pulse of a source, its deliveries and operator work."""
        self._run_pulse(
            pulse.key,
            counters=counters,
            begin=lambda: begin_pulse(
                self.session,
                pulse=pulse.key,
                emission=pulse.emission,
                inputs=self._inputs,
                observed=pulse.observed,
                coordination=self.coordination,
                reactions=self.reactions,
                controls=pulse.controls,
            ),
            begin_stage="emission",
            present=frozenset(pulse.emission.data),
            filtered=pulse.emission.is_filtered,
            observed=pulse.observed,
        )

    def _run_operator_pulse(
        self,
        slot: OperatorSlot,
        key: PulseKey,
        emission: OperatorPulse,
        *,
        controls: ControlSnapshot,
    ) -> _Outcome:
        outcome = self._run_pulse(
            key,
            counters=slot.counters,
            begin=lambda: begin_operator_pulse(
                self.session,
                pulse=key,
                emission=emission,
                inputs=self._inputs,
                coordination=self.coordination,
                reactions=self.reactions,
                controls=controls,
            ),
            begin_stage="operator",
            present=frozenset(emission.ports),
            filtered=False,
            observed=None,
        )

        return outcome

    def _run_pulse(
        self,
        key: PulseKey,
        *,
        counters: Counters,
        begin: Callable[[], RunState],
        begin_stage: ActiveRunStage,
        present: FrozenSet[str],
        filtered: bool,
        observed: Optional[Timestamp],
    ) -> _Outcome:
        """Run one pulse of a source or an operator, then the operators it feeds.

        A failure raised here is attributed to this pulse, recorded as the
        run's failure once and returned; a failure of work the pulse fed is
        returned as it was recorded there. Either way ``on_pulse_finished``
        of this pulse receives it. A pulse cut short by a failure elsewhere
        (or a cancellation) is cancelled with no error.
        """
        run: Optional[RunState] = None
        stage: ActiveRunStage = "observer"
        with use_pulse_run_id(key.run_id):
            try:
                self.observer.on_pulse_started(
                    run_id=key.run_id, source=key.source, pulse=key
                )
                stage = begin_stage
                run = begin()
                self._gauge("live_states", 1)
                stage = "step"
                if self._run_route(
                    run,
                    counters=counters,
                    present=present,
                    filtered=filtered,
                    observed=observed,
                ):
                    outcome = self._feed_operators(run)
                else:
                    outcome = _CANCELLED
            except RunAborted:
                outcome = _CANCELLED
            except Exception as raised:
                error = attributed(raised, stage=stage, **self._where(key))
                self._fail(error)
                outcome = _Outcome(completed=False, error=error)
            finally:
                if run is not None:
                    abandon_pulse(run)
                    self._gauge("live_states", -1)
            self.count(counters, "processed" if outcome.completed else "cancelled")
            self._pulse_finished(key, error=outcome.error)
        ended = self.progress.complete(key.source)
        if ended is not None:
            self._end_later(key.source, ended)

        return outcome

    def _pulse_finished(
        self, key: PulseKey, *, error: Optional[ActiveRunError]
    ) -> None:
        try:
            self.observer.on_pulse_finished(
                run_id=key.run_id, source=key.source, pulse=key, error=error
            )
        except Exception as raised:
            self._fail(attributed(raised, stage="observer", **self._where(key)))

    def _feed_operators(self, run: RunState) -> _Outcome:
        """Push the pulse's values to each consuming operator; run what they emit."""
        for planned in self.session.plan.consumers_of(run.pulse.source):
            if self._aborting():
                return _CANCELLED
            slot = self.operators[planned.name]
            arrivals = operator_arrivals(run, planned)
            self._drain_before_operator(run.controls, run)
            turn = self.coordination.stages(run, {"push": slot.stage}, calls=1)
            with turn.call("push"):
                self.count(slot.counters, "arrivals", len(arrivals))
                emitted = self._call_operator(slot, "push", arrivals, fed_by=run.pulse)
            # An operator pulse inherits the snapshot of the pulse that fed it.
            outcome = self._run_emissions(slot, emitted, controls=run.controls)
            if not outcome.completed:
                return outcome

        return _COMPLETED

    def _exclusive(self, slot: OperatorSlot) -> ContextManager[Any]:
        """The operator's stage without a turn, for ``end_input`` and ``finish``."""
        exclusive = self.coordination.gate(slot.stage).exclusive()

        return exclusive

    def _call_operator(
        self,
        slot: OperatorSlot,
        method: str,
        argument: Any,
        *,
        fed_by: Optional[PulseKey] = None,
    ) -> Optional[_Emitted]:
        """Call ``push``, ``end_input`` or ``finish``; number what it returned.

        The caller holds the operator's stage, so returned pulses get their
        sequences and are counted outstanding before another call of this
        operator, or a domain end, can observe them. Returns ``None`` when
        the call raised or returned something other than a list of
        ``OperatorPulse``; that failure is attributed to the operator, kept
        as ``slot.error`` and recorded as the run's failure. Pulses an
        operator built but did not return were never emitted.
        """
        try:
            returned = getattr(slot.instance, method)(argument)
            if not isinstance(returned, (list, tuple)) or not all(
                isinstance(pulse, OperatorPulse) for pulse in returned
            ):
                raise ContractError(
                    f"{method}() must return a list of OperatorPulse, got {returned!r}"
                )
        except Exception as error:
            fed = ""
            where: Dict[str, Any] = {}
            if fed_by is not None and fed_by.source in self._sources:
                where = {"source": fed_by.source, "pulse": fed_by.sequence}
            elif fed_by is not None:
                fed = f" (fed by pulse {fed_by.source}#{fed_by.sequence})"
            failure = ActiveRunError(
                f"{method}{fed} raised {type(error).__name__}: {error}",
                stage="operator",
                operator=slot.name,
                **where,
            )
            failure.__cause__ = error
            slot.error = failure
            self._fail(failure)
            return None

        emitted = [
            (
                PulseKey(
                    active_run_id=self.run_id,
                    source=slot.name,
                    sequence=slot.next_sequence + position,
                ),
                pulse,
            )
            for position, pulse in enumerate(returned)
        ]
        slot.next_sequence += len(emitted)
        self.progress.add(slot.name, len(emitted))
        self.count(slot.counters, "emitted", len(emitted))
        self._gauge("pending_operator_pulses", len(emitted))

        return emitted

    def _run_emissions(
        self,
        slot: OperatorSlot,
        emitted: Optional[_Emitted],
        *,
        controls: ControlSnapshot,
    ) -> _Outcome:
        """Run an operator's returned pulses in order until one does not complete.

        ``None`` is a failed call (see ``_call_operator``). A started pulse
        counts itself as processed or cancelled; returned pulses that never
        started because of a failure are counted cancelled here. ``controls``
        is the snapshot the emitted pulses run under: the feeding pulse's for
        a ``push``; the one the driver took when it scheduled the domain end
        for an end-of-input or finish emission, which no pulse fed.
        """
        if emitted is None:
            return _Outcome(completed=False, error=slot.error)

        # A pulse that does not complete always leaves a recorded run failure,
        # or the run was cancelled.
        outcome = _COMPLETED
        started = 0
        for key, emission in emitted:
            if self._aborting():
                break
            self._gauge("pending_operator_pulses", -1)
            outcome = self._run_operator_pulse(slot, key, emission, controls=controls)
            started += 1
        unstarted = len(emitted) - started
        self.count(slot.counters, "cancelled", unstarted)
        self._gauge("pending_operator_pulses", -unstarted)
        if unstarted and outcome.completed:
            outcome = _CANCELLED  # the run failed elsewhere before they started

        return outcome

    def close_operators(self) -> None:
        """Close every operator once, releasing what it retained; notify the observer."""
        for slot in self.operators.values():
            if slot.close_attempted:
                continue
            close_error = close_operator(slot)
            if close_error is not None:
                self._fail(close_error)
                slot.error = slot.error or close_error
            try:
                self.observer.on_operator_finished(operator=slot.name, error=slot.error)
            except Exception as raised:
                self._fail(attributed(raised, stage="observer", operator=slot.name))

    # Route and delivery -----------------------------------------------------

    def _run_route(
        self,
        run: RunState,
        *,
        counters: Counters,
        present: FrozenSet[str],
        filtered: bool,
        observed: Optional[Timestamp],
    ) -> bool:
        """Run the pulse's route, delivering each activated group once it is ready.

        A group is activated when its anchor port is ``present``, or for
        every group of the domain when the pulse is explicitly ``filtered``;
        an unactivated group retires this pulse's delivery turn at once.
        Returns ``False`` when a failure elsewhere cancelled the rest of the
        pulse at a step or delivery boundary.
        """
        pending: List[Tuple[Registered, StepStages]] = []
        for item in self._registered:
            if item.group.source != run.pulse.source:
                continue
            activated = filtered or item.group.anchor.output in present
            omitted = activated and run.group_omitted(item.group)
            turn = self.coordination.stages(
                run,
                {"deliver": item.stage},
                calls=1 if activated and not omitted else 0,
            )
            if omitted:
                # Every field reads a disabled control's steps: nothing to
                # deliver, nothing to wait for; the turn retired above.
                self._omit_group(run, item, counters=counters)
            elif activated:
                pending.append((item, turn))
        executed: Set[StepPath] = set()
        delivery = _Delivery(
            counters=counters, executed=executed, filtered=filtered, observed=observed
        )
        self._deliver_ready(run, pending, delivery=delivery)
        for step in run.plan.route(run.pulse.source):
            if self._aborting():
                return False
            execute_step(run, step)
            executed.add(step.path)
            self._deliver_ready(run, pending, delivery=delivery)
        completed = not pending

        return completed

    def _omit_group(
        self, run: RunState, item: Registered, *, counters: Counters
    ) -> None:
        reason = (
            run.controls.omitted_reason(next(iter(item.group.dependencies), ()))
            or "every field reads steps of a disabled control"
        )
        run.record("group_omitted", group=item.group.name, reason=reason)
        self.count(counters, "omitted")
        try:
            self.observer.on_group_omitted(
                run_id=run.run_id,
                group=item.group.name,
                source=run.pulse.source,
                pulse=run.pulse,
                reason=reason,
            )
        except Exception as error:
            raise ActiveRunError(
                f"on_group_omitted raised {type(error).__name__}: {error}",
                stage="observer",
                group=item.group.name,
                **self._where(run.pulse),
            ) from error

    def _deliver_ready(
        self,
        run: RunState,
        pending: List[Tuple[Registered, StepStages]],
        *,
        delivery: "_Delivery",
    ) -> None:
        """Deliver, and drop from ``pending``, every group whose steps have run."""
        for item, turn in list(pending):
            if not delivery.filtered and not item.prerequisites <= delivery.executed:
                continue
            if self._aborting():
                return
            self._deliver(run, item, turn=turn, delivery=delivery)
            pending.remove((item, turn))

    def _deliver(
        self,
        run: RunState,
        item: Registered,
        *,
        turn: StepStages,
        delivery: "_Delivery",
    ) -> None:
        result = group_result(run, item.group, filtered=delivery.filtered)
        where = self._where(run.pulse)
        with turn.call("deliver"):
            if item.recorder is not None:
                self._record(item, result, where=where)
            if item.handler is not None:
                called_ns = self._call_handler(item, result, where=where)
            else:
                called_ns = time.monotonic_ns()
            self.count(delivery.counters, "delivered")
            self._record_age(item, observed=delivery.observed, called_ns=called_ns)
            try:
                self.observer.on_group_delivered(
                    run_id=run.run_id,
                    group=item.group.name,
                    source=run.pulse.source,
                    pulse=run.pulse,
                )
            except Exception as error:
                raise ActiveRunError(
                    f"on_group_delivered raised {type(error).__name__}: {error}",
                    stage="observer",
                    group=item.group.name,
                    **where,
                ) from error

    def _record(
        self, item: Registered, result: GroupResult, *, where: Dict[str, Any]
    ) -> None:
        """Persist the result before any handler or later step can change it.

        Runs in the group's delivery turn, so a group's records keep pulse
        order, but outside the callbacks lock: other groups record, and
        handlers run, meanwhile.
        """
        self.coordination.checkpoint()
        try:
            item.recorder(result)
        except Exception as error:
            raise ActiveRunError(
                f"recording raised {type(error).__name__}: {error}",
                stage="recording",
                group=item.group.name,
                **where,
            ) from error

    def _call_handler(
        self, item: Registered, result: GroupResult, *, where: Dict[str, Any]
    ) -> int:
        """Call the host's handler under the callbacks lock; return the call time."""
        with self.coordination.callbacks():
            self.coordination.checkpoint()
            called_ns = time.monotonic_ns()
            try:
                returned = item.handler(result)
            except Exception as error:
                raise ActiveRunError(
                    f"handler raised {type(error).__name__}: {error}",
                    stage="handler",
                    group=item.group.name,
                    **where,
                ) from error
        if _closed_awaitable(returned):
            raise ActiveRunError(
                "handler returned an awaitable; active runs call synchronous "
                "handlers only and never await their results",
                stage="handler",
                group=item.group.name,
                **where,
            )

        return called_ns

    def _record_age(
        self, item: Registered, *, observed: Optional[Timestamp], called_ns: int
    ) -> None:
        """Handler call time minus observation time, for groups of source pulses."""
        if self._gauges is None or observed is None:
            return

        age_ns = called_ns - observed.ticks
        self._gauges.record_result_age(item.group.name, age_ns)

    def _where(self, pulse: PulseKey) -> Dict[str, Any]:
        """Attribution of a pulse: its source or operator, and its sequence."""
        domain = "operator" if pulse.source in self.operators else "source"
        where = {domain: pulse.source, "pulse": pulse.sequence}

        return where


@dataclass(frozen=True)
class _Delivery:
    """What every delivery of one pulse shares."""

    counters: Counters
    executed: Set[StepPath]
    filtered: bool
    observed: Optional[Timestamp]


def _closed_awaitable(returned: Any) -> bool:
    """Whether a handler returned an awaitable; a coroutine is closed unrun."""
    if not inspect.isawaitable(returned):
        return False
    if inspect.iscoroutine(returned):
        returned.close()

    return True
