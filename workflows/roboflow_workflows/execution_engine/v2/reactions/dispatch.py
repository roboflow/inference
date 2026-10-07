"""From ``self.emit(...)`` (or a signal, system or machine event) to subscribers.

One step emission, on the emitting thread, inside the block call::

    declared?     the step's BlockSpec.events has the name, else EventEmissionError
    fields        exactly the declared names, else EventPayloadError
    index         the call's one index, or ``at`` (required by every
                  batch-delivering call, even of one member)
    subscribed?   ReactionPlan.subscribed(origin): a handler or a fixed
                  transition listens; none -> return here
    publish       shared by every origin kind (step, signal, system, machine):
      ready         futures of the demanded fields resolved (readiness
                    boundary), their kinds checked; the demand is the union of
                    handler bindings and fixed-transition ``$event`` fields;
                    other fields are dropped
      snapshot      one owned snapshot per asynchronous handler, all taken
                    before any transition or handler runs
      cause         EventCause: origin kind, event, emitter, run, pulse,
                    index, source, time, parent cause, machine stamp
      dispatch      fixed transitions first (their machine events publish
                    inline, recursively), then handlers in declaration order:
                    sync handlers run now, on this thread; async handlers are
                    admitted to their queues

Nothing here retains a payload: the cause is metadata only, and the fields
live only in the dispatched handler calls and queue items.
"""

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

from roboflow_workflows.execution_engine.v2.context import ExecutionContext
from roboflow_workflows.execution_engine.v2.data import (
    Index,
    SampleContext,
    TemporalContext,
)
from roboflow_workflows.execution_engine.v2.errors import (
    EventEmissionError,
    StepPath,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.events import Event, EventPayloadError
from roboflow_workflows.execution_engine.v2.kinds import WILDCARD_KIND_NAME
from roboflow_workflows.execution_engine.v2.reactions.plan import EventOrigin
from roboflow_workflows.execution_engine.v2.reactions.snapshots import (
    Snapshot,
    snapshot_fields,
)
from roboflow_workflows.execution_engine.v2.readiness import resolve_futures

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.reactions.machines import MachineStamp

__all__ = ["EventCause", "Reactions", "emit_event", "has_subscribers", "publish"]


@dataclass(frozen=True)
class EventCause:
    """Why a handler run happened: one emitted event, without its payload.

    Args:
        event: Declared event name.
        emitter: Step path of the emitting step in its plan, machine path of
            a machine event, ``()`` for a signal or system event.
        session_id: Session of the emitting run.
        run_id: Emitting run: ``PulseKey.run_id`` of a pulse, or the passive
            run id.
        pulse: Emitting pulse of an active run; ``None`` for a passive run.
        index: Full logical index of the emitting invocation.
        sample: Source context at ``index``; ``None`` when it has no single
            source.
        temporal: Temporal context at ``index``; ``None`` when it has no
            single time.
        sequence: Emission number within the reaction runtime, from 0.
        parent: Cause of the handler run that emitted this event, or of the
            event whose transition emitted this machine event; ``None`` for
            main-flow, signal and system events.
        kind: Origin kind: ``step``, ``signal``, ``system`` or ``machine``.
        stamp: The applied transition of a machine event; ``None`` otherwise.
    """

    event: str
    emitter: StepPath
    session_id: str
    run_id: str
    pulse: Optional[Any]
    index: Index
    sample: Optional[SampleContext]
    temporal: Optional[TemporalContext]
    sequence: int
    parent: Optional["EventCause"] = None
    kind: str = "step"
    stamp: Optional["MachineStamp"] = None

    @property
    def origin(self) -> EventOrigin:
        """Origin of the event in the reaction plan."""
        return EventOrigin(kind=self.kind, event=self.event, path=self.emitter)

    @property
    def origin_selector(self) -> str:
        """Selector of the emitted event, e.g. ``$steps.count.events.crossed``."""
        return self.origin.selector

    @property
    def source_id(self) -> Optional[str]:
        """Source id of the emitting index, or ``None``."""
        return self.sample.source_id if self.sample is not None else None

    def describe(self) -> Dict[str, Any]:
        """Describe the cause as JSON-friendly data.

        Returns:
            Origin, event, emitter, run, pulse, index, source and sequence.
        """
        pulse = self.pulse
        description = {
            "origin": self.origin_selector,
            "event": self.event,
            "emitter": format_step_path(self.emitter) if self.emitter else None,
            "run_id": self.run_id,
            "pulse": (
                None
                if pulse is None
                else {"source": pulse.source, "sequence": pulse.sequence}
            ),
            "index": list(self.index),
            "source_id": self.source_id,
            "sequence": self.sequence,
        }

        return description


class Reactions(Protocol):
    """What the emit path needs from a reaction runtime."""

    def subscribed(self, origin: EventOrigin) -> bool:
        """Whether a handler or a fixed transition listens to ``origin``."""

    def bound_fields(self, origin: EventOrigin) -> FrozenSet[str]:
        """Fields any subscriber of ``origin`` reads; the others are dropped."""

    def handlers_for(self, origin: EventOrigin) -> Sequence[Any]:
        """Handlers subscribed to ``origin``, in declaration order."""

    def next_sequence(self) -> int:
        """Next emission number."""

    def dispatch(
        self,
        handlers: Sequence[Any],
        cause: EventCause,
        *,
        fields: Mapping[str, Any],
        snapshots: Mapping[Any, Snapshot],
    ) -> None:
        """Apply fixed transitions, then run or admit handlers, in order."""


def has_subscribers(
    reactions: Optional[Reactions],
    *,
    events: Mapping[str, Event],
    path: StepPath,
    event: str,
) -> bool:
    """Whether emitting ``event`` from step ``path`` would reach a handler.

    Args:
        reactions: Reaction runtime of the run; ``None`` when it has none.
        events: Events the step's block declares.
        path: Emitting step path.
        event: Event name.

    Returns:
        ``True`` when at least one handler or fixed transition subscribes.

    Raises:
        EventEmissionError: For an undeclared event.
    """
    _declared(events, event=event, path=path)
    subscribed = reactions is not None and reactions.subscribed(
        EventOrigin(kind="step", event=event, path=path)
    )

    return subscribed


def emit_event(
    reactions: Optional[Reactions],
    *,
    context: ExecutionContext,
    events: Mapping[str, Event],
    event: str,
    fields: Mapping[str, Any],
    at: Optional[Index],
    pulse: Optional[Any],
    parent: Optional[EventCause],
) -> None:
    """Validate one emission and hand it to its subscribers.

    Args:
        reactions: Reaction runtime of the emitting run; ``None`` when the
            run has none (the emission is validated, then dropped).
        context: Context of the emitting call.
        events: Events the step's block declares.
        event: Emitted event name.
        fields: Emitted field values.
        at: Index the event belongs to, when given.
        pulse: Emitting pulse, when the run is a pulse.
        parent: Cause of the handler run that emits, when it is one.

    Raises:
        EventEmissionError: For an undeclared event, a bad ``at``, a failed
            future, a field without a snapshot or closed reactions.
        EventPayloadError: For missing, unknown or ill-kinded fields.
        ReactionError: When a synchronous handler failed.
    """
    path = context.step_path
    declared = _declared(events, event=event, path=path)
    _check_names(declared, event=event, fields=fields)
    index = _event_index(context, event=event, at=at)
    if reactions is None:
        return
    origin = EventOrigin(kind="step", event=event, path=path)
    if not reactions.subscribed(origin):
        return

    publish(
        reactions,
        origin,
        declared=declared,
        fields=fields,
        where=f"{context.step_selector} event '{event}'",
        session_id=context.session_id,
        run_id=context.run_id,
        pulse=pulse,
        index=index,
        sample=context.sample_at(index),
        temporal=context.temporal_at(index),
        parent=parent,
    )


def publish(
    reactions: Reactions,
    origin: EventOrigin,
    *,
    declared: Event,
    fields: Mapping[str, Any],
    where: str,
    session_id: str,
    run_id: Optional[str],
    pulse: Optional[Any],
    index: Index,
    sample: Optional[SampleContext],
    temporal: Optional[TemporalContext],
    parent: Optional[EventCause],
    stamp: Optional["MachineStamp"] = None,
) -> None:
    """Hand one validated, subscribed event to its transitions and handlers.

    Args:
        reactions: Reaction runtime receiving the event.
        origin: Origin of the event.
        declared: Its payload declaration (kinds of the retained fields).
        fields: Every field of the event; only demanded ones are kept.
        where: Location text for errors.
        session_id: Session of the run.
        run_id: Run the event belongs to.
        pulse: Emitting pulse, if any.
        index: Full logical index of the event (``()`` outside a step).
        sample: Source context of the event, or ``None``.
        temporal: Temporal context of the event, or ``None``.
        parent: Cause this event descends from, if any.
        stamp: Applied transition of a machine event.

    Raises:
        EventEmissionError: For a failed future, a field without a snapshot
            or closed reactions.
        EventPayloadError: For an ill-kinded retained field.
        ReactionError: When a synchronous handler failed.
    """
    handlers = reactions.handlers_for(origin)
    ready = _ready(fields, bound=reactions.bound_fields(origin), where=where)
    _check_kinds(declared, event=origin.event, values=ready)
    snapshots = {
        handler.path: snapshot_fields(
            {name: ready[name] for name in handler.bound_fields}, where=where
        )
        for handler in handlers
        if handler.mode == "async"
    }
    cause = EventCause(
        event=origin.event,
        emitter=origin.path,
        session_id=session_id,
        run_id=run_id,
        pulse=pulse,
        index=index,
        sample=sample,
        temporal=temporal,
        sequence=reactions.next_sequence(),
        parent=parent,
        kind=origin.kind,
        stamp=stamp,
    )
    reactions.dispatch(handlers, cause, fields=ready, snapshots=snapshots)


def _declared(events: Mapping[str, Event], *, event: str, path: StepPath) -> Event:
    declared = events.get(event)
    if declared is None:
        names = sorted(events) if events else "no events"
        raise EventEmissionError(
            f"{format_step_path(path)} emitted undeclared event {event!r}; its "
            f"block declares {names}"
        )

    return declared


def _check_names(declared: Event, *, event: str, fields: Mapping[str, Any]) -> None:
    if len(fields) == len(declared.fields) and all(
        name in declared.fields for name in fields
    ):
        return

    missing = sorted(name for name in declared.fields if name not in fields)
    unknown = sorted(name for name in fields if name not in declared.fields)
    raise EventPayloadError(
        f"Event '{event}' declares fields {sorted(declared.fields)}; missing "
        f"{missing}, unknown {unknown}",
        event=event,
    )


def _check_kinds(declared: Event, *, event: str, values: Mapping[str, Any]) -> None:
    """Kind check of the fields handlers receive (``Event.check_payload`` rule)."""
    for name, value in values.items():
        kinds = declared.fields[name]
        if any(
            kind.name == WILDCARD_KIND_NAME
            or kind.validate is None
            or kind.validate(value)
            for kind in kinds
        ):
            continue
        raise EventPayloadError(
            f"Event '{event}' field '{name}' got {type(value).__name__}, which "
            f"none of the kinds {list(declared.kind_names(name))} accepts",
            event=event,
        )


def _event_index(
    context: ExecutionContext, *, event: str, at: Optional[Index]
) -> Index:
    """The emitting index: the call's one index, or ``at`` for a batch call."""
    if at is None:
        if not context.batched and len(context.indices) == 1:
            return context.indices[0]
        raise EventEmissionError(
            f"{context.step_selector} emitted '{event}' from a batch-delivering "
            f"call ({len(context.indices)} members); pass at=<member index>, e.g. "
            "at=batch.indices[k]"
        )

    index = tuple(at)
    if index not in context.indices:
        raise EventEmissionError(
            f"{context.step_selector} emitted '{event}' at {list(index)}, which "
            f"is not an index of this call ({[list(item) for item in context.indices]})"
        )

    return index


def _ready(
    fields: Mapping[str, Any], *, bound: FrozenSet[str], where: str
) -> Dict[str, Any]:
    """Bound fields with every future resolved; the others are dropped here."""
    retained = {name: fields[name] for name in fields if name in bound}
    try:
        ready = resolve_futures(retained)
    except Exception as error:
        raise EventEmissionError(
            f"{where}: a future in a bound field failed with "
            f"{type(error).__name__}: {error}"
        ) from error

    return ready
