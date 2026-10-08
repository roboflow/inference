"""Unit-test a block's events and source-scoped state without the engine.

``block_call`` makes one call context current, as the engine would, and
captures what the block emits::

    block = CountThreshold(managed_state=ManagedState())
    with block_call(block, source_id="cam_a") as call:
        block.run(count=5, threshold=3)
    assert call.events == [
        CapturedEvent("crossed", {"total": 5}, at=(), source_id="cam_a")
    ]

Inside the ``with``, ``self.emit`` validates exactly as in a run (declared
name and fields, kinds, ``at`` for several indices), ``has_subscribers``
answers per ``subscribed``, ``execution_context.sample_at`` returns the
given sources, and ``ManagedState.source`` resolves to them. No handler
runs; nothing is snapshotted.

``self.wants(name)`` answers from ``wanted`` (every output when ``None``)
exactly as the engine does, and ``call.queried`` lists what the block asked::

    with block_call(painter, wanted=("count",)) as call:
        result = painter.run(value=1.0)
    assert "overlay" not in result and call.queried == {"overlay"}
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    answer_wants,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import (
    Index,
    SampleContext,
    TemporalContext,
)
from roboflow_workflows.execution_engine.v2.declaration import spec_of
from roboflow_workflows.execution_engine.v2.errors import ContractError, StepPath
from roboflow_workflows.execution_engine.v2.reactions.dispatch import (
    EventCause,
    emit_event,
    has_subscribers,
)

__all__ = ["CapturedCall", "CapturedEvent", "block_call"]


@dataclass(frozen=True)
class CapturedEvent:
    """One event a block emitted inside ``block_call``.

    Args:
        event: Event name.
        fields: Emitted field values, as passed (not copied).
        at: Index the event belongs to.
        source_id: Source of that index, or ``None``.
    """

    event: str
    fields: Mapping[str, Any]
    at: Index = ()
    source_id: Optional[str] = None


@dataclass
class CapturedCall:
    """What one ``block_call`` captured.

    Args:
        context: The call's execution context.
        events: Emitted events, in order.
    """

    context: ExecutionContext
    events: List[CapturedEvent] = field(default_factory=list)

    @property
    def queried(self) -> FrozenSet[str]:
        """Outputs the block asked ``wants`` about during the call."""
        return frozenset(self.context.queried_outputs)


class _Capture:
    """``reactions.dispatch.Reactions`` that records instead of dispatching."""

    def __init__(
        self,
        call: CapturedCall,
        *,
        events: Mapping[str, Any],
        subscribed: Optional[Iterable[str]],
    ):
        self._call = call
        self._events = events
        self._subscribed = None if subscribed is None else frozenset(subscribed)
        self._sequence = 0

    def subscribed(self, origin: Any) -> bool:
        return self._subscribed is None or origin.event in self._subscribed

    def bound_fields(self, origin: Any) -> FrozenSet[str]:
        return frozenset(self._events[origin.event].fields)

    def handlers_for(self, origin: Any) -> Sequence[Any]:
        return ()

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence - 1

    def dispatch(
        self,
        handlers: Sequence[Any],
        cause: EventCause,
        *,
        fields: Mapping[str, Any],
        snapshots: Mapping[Any, Any],
    ) -> None:
        captured = CapturedEvent(
            event=cause.event,
            fields=dict(fields),
            at=cause.index,
            source_id=cause.source_id,
        )
        self._call.events.append(captured)


class _Scope:
    """Call scope with fixed contexts per index."""

    def __init__(
        self,
        events: Mapping[str, Any],
        reactions: _Capture,
        *,
        samples: Mapping[Index, Optional[SampleContext]],
        temporals: Mapping[Index, Optional[TemporalContext]],
        outputs: Iterable[str] = (),
    ):
        self._events = events
        self._reactions = reactions
        self._samples = samples
        self._temporals = temporals
        self._outputs = frozenset(outputs)

    def wants(self, context: ExecutionContext, output: str) -> bool:
        wanted = answer_wants(context, output, declared=self._outputs)

        return wanted

    @property
    def cause(self) -> Optional[EventCause]:
        return None

    def sample_at(self, index: Index) -> Optional[SampleContext]:
        return self._samples.get(index)

    def temporal_at(self, index: Index) -> Optional[TemporalContext]:
        return self._temporals.get(index)

    def has_subscribers(self, event: str) -> bool:
        subscribed = has_subscribers(
            self._reactions, events=self._events, path=("$capture",), event=event
        )
        return subscribed

    def set_machine_state(self, machine: str, transition: str, next_state: str) -> Any:
        raise ContractError(
            "set_machine_state needs a handler run of a compiled workflow; "
            "block_call() has no state machines"
        )

    def emit(
        self,
        context: ExecutionContext,
        event: str,
        fields: Mapping[str, Any],
        *,
        at: Optional[Index],
    ) -> None:
        emit_event(
            self._reactions,
            context=context,
            events=self._events,
            event=event,
            fields=fields,
            at=at,
            pulse=None,
            parent=None,
        )


@contextmanager
def block_call(
    block: Any,
    *,
    source_id: Optional[str] = "test",
    indices: Sequence[Index] = ((),),
    batched: Optional[bool] = None,
    sources: Optional[Mapping[Index, Optional[str]]] = None,
    temporal: Optional[Mapping[Index, Optional[TemporalContext]]] = None,
    subscribed: Optional[Iterable[str]] = None,
    wanted: Optional[Iterable[str]] = None,
    step_path: StepPath = ("block",),
) -> Iterator[CapturedCall]:
    """Run block code as one engine call and capture its events.

    Args:
        block: The block instance or class (its declared ``events`` apply).
        source_id: Source of every index unless ``sources`` names them.
        indices: Logical indices of the call; several for a batch call.
        batched: Whether the call is batch-delivering; ``True`` when
            ``indices`` has several entries unless given.
        sources: Source id per index, overriding ``source_id``; ``None``
            means the index has no source.
        temporal: Temporal context per index; none by default.
        subscribed: Events reported as having subscribers; every declared
            event when ``None``. Unsubscribed events are validated, not
            captured, as in a run.
        wanted: Outputs ``self.wants`` answers ``True`` for; every declared
            output when ``None``. Use it to test the branch that skips an
            unwanted output.
        step_path: Step path shown in messages.

    Yields:
        The captured call; ``events`` fills while the block runs and
        ``queried`` lists the outputs the block asked about.

    Raises:
        ContractError: When ``wanted`` names an output the block does not
            declare.
    """
    block_class = block if isinstance(block, type) else type(block)
    spec = spec_of(block_class)
    if isinstance(wanted, str):
        raise ContractError(
            f"block_call wanted must be a collection of output names, got {wanted!r}"
        )
    wanted_outputs = None if wanted is None else frozenset(wanted)
    if wanted_outputs is not None and not wanted_outputs <= set(spec.outputs):
        raise ContractError(
            f"block_call wanted {sorted(wanted_outputs - set(spec.outputs))}, which "
            f"{spec.type} does not declare; its outputs are {sorted(spec.outputs)}"
        )
    call_indices = tuple(tuple(index) for index in indices)
    named: Dict[Index, Optional[str]] = {index: source_id for index in call_indices}
    named.update({tuple(index): value for index, value in (sources or {}).items()})
    samples = {
        index: SampleContext(source_id=name) if name is not None else None
        for index, name in named.items()
    }
    temporals = {tuple(index): value for index, value in (temporal or {}).items()}
    call = CapturedCall(context=None)  # type: ignore[arg-type]
    scope = _Scope(
        spec.events,
        _Capture(call, events=spec.events, subscribed=subscribed),
        samples=samples,
        temporals=temporals,
        outputs=spec.outputs,
    )
    context = ExecutionContext(
        step_path=step_path,
        block_type=spec.type,
        session_id="block-call",
        run_id="block-call",
        indices=call_indices,
        batched=len(call_indices) > 1 if batched is None else batched,
        call_scope=scope,
        wanted_outputs=wanted_outputs,
    )
    call.context = context
    with use_execution_context(context):
        yield call
