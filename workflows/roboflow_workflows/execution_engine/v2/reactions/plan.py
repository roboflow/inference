"""Compiled reaction contract: handlers, signals, state machines and state.

A workflow reacts to events through handlers. Each handler is an independent
passive ``CompiledWorkflow`` compiled once with the normal compiler; the
runtime runs it with its own persistent session at event rate::

    origin (step / signal / system / machine event)
        │ bound fields only (handler bindings ∪ transition $event fields)
        ├──▶ PlannedTransition (fixed, per machine) ──▶ machine event origin
        └──▶ PlannedHandler ──▶ handler plan outputs ──▶ PlannedHandlerGroup
                  │ may set handler-selected transitions (state_machine_set)
                  ▼
             machine event origin

Every automatic cascade edge (fixed transition, handler-selected transition)
is known at compile time, so ``ReactionPlan`` rejects cascades that can reach
their own origin again. State graphs themselves may cycle.

``ReactionPlan.EMPTY`` is the plan of every workflow without reactions; its
lookups return empty tuples so the main flow pays one cheap branch.

The module imports no execution or plan modules, so it stays importable from
the declaration, compiler, plan and runtime layers alike.
"""

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
)

from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    FieldPath,
    KindMismatchError,
    SelectorError,
    StepPath,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import kinds_compatible

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow

HandlerMode = Literal["sync", "async"]
OverflowPolicy = Literal["leaky", "synchronous"]
OriginKind = Literal["step", "signal", "system", "machine"]
MachineScope = Literal["source", "global"]
FieldSource = Literal["event", "transition", "literal"]

HANDLER_MODES: Tuple[str, ...] = ("sync", "async")
OVERFLOW_POLICIES: Tuple[str, ...] = ("leaky", "synchronous")
ORIGIN_KINDS: Tuple[str, ...] = ("step", "signal", "system", "machine")
MACHINE_SCOPES: Tuple[str, ...] = ("source", "global")
TRANSITION_VALUES: Tuple[str, ...] = ("from", "to", "name")
DEFAULT_QUEUE_DEPTH = 16

STATE_MACHINE_SET_TYPE = "v2/state_machine_set"
"""Block type of the handler-only setter of handler-selected transitions."""

SYSTEM_EVENTS: Mapping[str, Event] = MappingProxyType(
    {
        "started": Event(
            description=(
                "The active run started, before any source. Sent once per run, "
                "also when a processing reset replaces the handlers."
            )
        ),
        "ended": Event(
            description="The main pulses finished on end of input or graceful stop."
        ),
    }
)
"""Run lifecycle events, subscribed as ``$system.events.<name>``."""


def scoped_name(path: StepPath) -> str:
    """Render a scoped path as written in selectors, e.g. ``child/notify``.

    Args:
        path: Step or handler path.

    Returns:
        Path segments joined with ``/``.
    """
    return "/".join(path)


class ReactionContractError(ContractError):
    """A plan contract violation that a workflow definition can cause.

    The plan types check these rules once; the compiler reports the same
    error as ``compile_error`` at the same location (see
    ``as_compile_error``).

    Args:
        message: Problem description.
        compile_error: ``WorkflowCompileError`` type for a definition.
        step_path: Handler, step or machine path the problem belongs to.
        field_path: Definition field inside it, e.g. ``("bindings", "zone")``.
    """

    def __init__(
        self,
        message: str,
        *,
        compile_error: Type[WorkflowCompileError] = WorkflowCompileError,
        step_path: StepPath = (),
        field_path: FieldPath = (),
    ):
        super().__init__(message)
        self.compile_error = compile_error
        self.step_path = tuple(step_path)
        self.field_path = tuple(field_path)


def as_compile_error(
    error: ContractError, *, where: str, step_path: StepPath = ()
) -> WorkflowCompileError:
    """Report a plan constructor's error as a compile error.

    Args:
        error: Error raised while building a plan type.
        where: Definition location prefixed to the message.
        step_path: Path for an error that does not carry its own.

    Returns:
        A ``ReactionContractError`` as its ``compile_error`` type and location;
        any other contract error as a ``WorkflowCompileError`` at ``step_path``.
    """
    if isinstance(error, ReactionContractError):
        return error.compile_error(
            f"{where}: {error}",
            step_path=error.step_path,
            field_path=error.field_path,
        )

    return WorkflowCompileError(f"{where}: {error}", step_path=step_path)


@dataclass(frozen=True)
class EventOrigin:
    """Where an event comes from.

    Args:
        kind: ``"step"`` for a block event, ``"signal"`` for external ingress,
            ``"system"`` for a run lifecycle event, ``"machine"`` for an event
            a state machine transition emits.
        event: Event name, as declared by the block, signal or transition.
        path: Emitting step path or machine path; ``()`` for a signal or a
            system event.

    Raises:
        ContractError: On an unknown kind, a step or machine origin without a
            path, a signal or system origin with one, or an unknown system
            event.
    """

    kind: OriginKind
    event: str
    path: StepPath = ()

    def __post_init__(self) -> None:
        if self.kind not in ORIGIN_KINDS:
            raise ContractError(
                f"Event origin kind must be one of {list(ORIGIN_KINDS)}, got "
                f"{self.kind!r}"
            )
        object.__setattr__(self, "path", tuple(self.path))
        has_path = self.kind in ("step", "machine")
        if has_path and not self.path:
            raise ContractError(
                f"{self.kind.title()} event '{self.event}' needs a path"
            )
        if not has_path and self.path:
            raise ContractError(f"{self.kind.title()} event '{self.event}' has no path")
        if self.kind == "system" and self.event not in SYSTEM_EVENTS:
            raise ContractError(
                f"System event must be one of {sorted(SYSTEM_EVENTS)}, got "
                f"{self.event!r}"
            )

    @property
    def selector(self) -> str:
        """Selector text naming this origin, e.g. ``$steps.a/b.events.hit``."""
        if self.kind == "signal":
            return f"$signals.{self.event}"
        if self.kind == "system":
            return f"$system.events.{self.event}"
        if self.kind == "machine":
            return f"$state_machines.{scoped_name(self.path)}.events.{self.event}"

        return f"$steps.{scoped_name(self.path)}.events.{self.event}"


@dataclass(frozen=True)
class QueuePolicy:
    """Bounded queue of one async handler.

    Capacity counts queued events, not bytes. The running payload and payloads
    held by emitters waiting on synchronous overflow are additional.

    Args:
        max_depth: Queued events, not counting the running one; at least 1.
        overflow: ``"leaky"`` drops the oldest queued event and keeps the
            newest. ``"synchronous"`` keeps FIFO order: the emitter waits for
            older events and then runs the overflowing event itself.

    Raises:
        ContractError: On a non-positive depth or an unknown policy.
    """

    max_depth: int = DEFAULT_QUEUE_DEPTH
    overflow: OverflowPolicy = "synchronous"

    def __post_init__(self) -> None:
        depth = self.max_depth
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ContractError(
                f"Handler queue max_depth must be a positive integer, got {depth!r}"
            )
        if self.overflow not in OVERFLOW_POLICIES:
            raise ContractError(
                f"Handler queue overflow must be one of {list(OVERFLOW_POLICIES)}, "
                f"got {self.overflow!r}"
            )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {"max_depth": self.max_depth, "overflow": self.overflow}

        return description


@dataclass(frozen=True)
class PlannedSignal:
    """A declared external signal of the root workflow.

    Args:
        name: Signal name, used as ``$signals.<name>``.
        event: Declared payload of the signal.
    """

    name: str
    event: Event

    def __post_init__(self) -> None:
        if not isinstance(self.event, Event):
            raise ContractError(f"Signal '{self.name}' must declare an Event")

    @property
    def origin(self) -> EventOrigin:
        """Origin handlers use to subscribe to this signal."""
        return EventOrigin(kind="signal", event=self.name)


@dataclass(frozen=True)
class PlannedHandler:
    """One compiled handler: subscription, bindings and passive plan.

    Args:
        path: Handler identity: declaring scope plus handler name, e.g.
            ``("child", "notify")``. Used for counters, groups and errors.
        origin: Subscribed event origin.
        event: Declaration of the subscribed event.
        plan: Passive handler plan, compiled with the parent's catalogue and
            options. It has flat outputs and no reactions of its own.
        bindings: Handler input name to the event field it receives.
        constants: Handler input name to a literal value.
        mode: ``"sync"`` runs before ``emit`` returns; ``"async"`` uses
            ``queue``.
        queue: Queue policy; required for async handlers, ``None`` for sync.

    Raises:
        ContractError: On an empty path, an active or reacting handler plan, an
            input bound twice or a queue that does not match the mode.
        ReactionContractError: On a binding to an unknown input or field, an
            unbound required input, an input with axes or incompatible kinds.
    """

    path: StepPath
    origin: EventOrigin
    event: Event
    plan: "CompiledWorkflow"
    bindings: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    constants: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    mode: HandlerMode = "sync"
    queue: Optional[QueuePolicy] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "constants", MappingProxyType(dict(self.constants)))
        where = f"Handler '{scoped_name(self.path)}'"
        if not self.path:
            raise ContractError("Handler path must not be empty")
        if not isinstance(self.origin, EventOrigin):
            raise ContractError(f"{where} origin must be an EventOrigin")
        if not isinstance(self.event, Event):
            raise ContractError(f"{where} event must be an Event")
        if self.mode not in HANDLER_MODES:
            raise ContractError(
                f"{where} mode must be one of {list(HANDLER_MODES)}, got {self.mode!r}"
            )
        if self.mode == "async" and not isinstance(self.queue, QueuePolicy):
            raise ContractError(f"{where} is async and needs a QueuePolicy")
        if self.mode == "sync" and self.queue is not None:
            raise ContractError(f"{where} is sync and must not declare a queue")
        _check_handler_plan(self, where=where)

    @property
    def name(self) -> str:
        """Handler name inside its declaring scope."""
        return self.path[-1]

    @property
    def scope(self) -> StepPath:
        """Path of the workflow scope that declares the handler."""
        return self.path[:-1]

    @property
    def bound_fields(self) -> FrozenSet[str]:
        """Event fields this handler reads; other fields are not retained."""
        return frozenset(self.bindings.values())

    @property
    def outputs(self) -> Tuple[str, ...]:
        """Flat output names of the handler plan."""
        return tuple(output.name for output in self.plan.outputs)

    def inputs_for(self, fields: Mapping[str, Any]) -> Dict[str, Any]:
        """Build handler-plan run inputs from bound event field values.

        Args:
            fields: Event field values; must contain every bound field.

        Returns:
            Handler input name to value, constants included.
        """
        inputs = dict(self.constants)
        for input_name, field_name in self.bindings.items():
            inputs[input_name] = fields[field_name]

        return inputs

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description, including the handler plan."""
        description = {
            "on": self.origin.selector,
            "mode": self.mode,
            "queue": self.queue.describe() if self.queue is not None else None,
            "bindings": {
                name: f"$event.{item}" for name, item in self.bindings.items()
            },
            "constants": dict(self.constants),
            "workflow": self.plan.describe(),
        }

        return description


def _check_handler_plan(handler: PlannedHandler, *, where: str) -> None:
    plan = handler.plan
    if plan.is_active:
        raise ContractError(
            f"{where} workflow declares sources; handler workflows are passive"
        )
    if not plan.reactions.is_empty:
        raise ContractError(
            f"{where} workflow declares reactions (handlers, signals, machines or "
            "state); handler "
            "workflows cannot declare reactions of their own"
        )

    bound_twice = sorted(set(handler.bindings) & set(handler.constants))
    if bound_twice:
        raise ContractError(f"{where} binds inputs {bound_twice} twice")
    for input_name in [*handler.bindings, *handler.constants]:
        if input_name not in plan.inputs:
            raise _binding_error(
                handler,
                input_name,
                f"{where} binds '{input_name}', which is not an input of its "
                f"workflow; inputs are {sorted(plan.inputs)}",
            )
    for input_name, planned in plan.inputs.items():
        if planned.layout.axis_ids:
            raise _binding_error(
                handler,
                input_name,
                f"{where} input '{input_name}' declares axes "
                f"{list(planned.layout.axis_ids)}; a handler input receives one "
                "event value, so declare it without axes: WorkflowParameter or "
                '{"name": ..., "kind": [...]}',
            )
        bound = input_name in handler.bindings or input_name in handler.constants
        if planned.required and not bound:
            raise _binding_error(
                handler,
                input_name,
                f"{where} leaves required input '{input_name}' unbound; bind it "
                "to '$event.<field>' or a literal",
            )

    for input_name, field_name in handler.bindings.items():
        if field_name not in handler.event.fields:
            raise _binding_error(
                handler,
                input_name,
                f"{where} binds '$event.{field_name}', but "
                f"{handler.origin.selector} declares fields "
                f"{sorted(handler.event.fields)}",
                compile_error=SelectorError,
            )
        produced = handler.event.kind_names(field_name)
        accepted = plan.inputs[input_name].kinds
        if not kinds_compatible(produced, accepted):
            raise _binding_error(
                handler,
                input_name,
                f"{where} binds '$event.{field_name}' of kinds {list(produced)} to "
                f"input '{input_name}' accepting {list(accepted)}",
                compile_error=KindMismatchError,
            )


def _binding_error(
    handler: PlannedHandler,
    input_name: str,
    message: str,
    *,
    compile_error: Type[WorkflowCompileError] = WorkflowCompileError,
) -> ReactionContractError:
    error = ReactionContractError(
        message,
        compile_error=compile_error,
        step_path=handler.path,
        field_path=("bindings", input_name),
    )

    return error


@dataclass(frozen=True)
class PlannedHandlerGroup:
    """An output group fed by one handler's results (``$handlers.<h>.<out>``).

    One handler run that completes delivers one group result.

    Args:
        name: Group name, unique among all output groups of the workflow.
        handler: Path of the anchoring handler.
        fields: Group field name to the handler output it carries.

    Raises:
        ContractError: On a group without fields.
    """

    name: str
    handler: StepPath
    fields: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "handler", tuple(self.handler))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if not self.fields:
            raise ContractError(f"Handler group '{self.name}' selects no outputs")

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        handler = scoped_name(self.handler)
        description = {
            "anchor": f"$handlers.{handler}",
            "fields": {
                name: f"$handlers.{handler}.{output}"
                for name, output in self.fields.items()
            },
        }

        return description


@dataclass(frozen=True)
class StateDefaults:
    """Initial managed-state values declared by the workflow ``state`` key.

    Values are seeded with set-if-absent when an active run starts, so an
    explicitly shared namespace keeps its existing values.

    Args:
        global_: Execution-global key to initial value.
        source: Per-source key to initial value; seeded for every source the
            first time state for that source is touched.

    Raises:
        ContractError: On a key that is not a non-empty string or a value that
            is not portable JSON (no NaN/inf, no tensors).
    """

    global_: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    source: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        for scope, values in (("global", self.global_), ("source", self.source)):
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ContractError(
                        f"State {scope} key must be a non-empty string, got {key!r}"
                    )
                _check_portable(value, where=f"State {scope} '{key}'")
        object.__setattr__(self, "global_", MappingProxyType(dict(self.global_)))
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))

    @property
    def is_empty(self) -> bool:
        """Whether no initial value is declared."""
        return not self.global_ and not self.source

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {"global": dict(self.global_), "source": dict(self.source)}

        return description


def _check_portable(value: Any, *, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{where} must be finite, got {value!r}")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{where} must be portable JSON: {error}") from error


@dataclass(frozen=True)
class PlannedTransition:
    """One transition of a state machine.

    A fixed transition fires on its ``trigger`` event; a handler-selected one
    is applied by a ``state_machine_set`` step of ``handler``, which picks one
    of the ``targets``.

    Args:
        name: Transition name, unique in its machine.
        machine: Path of the owning machine.
        sources: States the transition leaves (``from``).
        targets: States it may enter (``to``); exactly one when fixed.
        trigger: Triggering event of a fixed transition.
        handler: Path of the handler that selects the target.
        emits: Machine event emitted after the transition is applied.
        fields: Emitted field to ``("event", trigger field)``,
            ``("transition", "from" | "to" | "name")`` or
            ``("literal", value)``.

    Raises:
        ContractError: On a missing or double trigger, a fixed transition
            without exactly one target, empty sources or targets, an emitted
            event of another machine, fields without an event, ``$event``
            fields on a handler-selected transition or a bad field source.
    """

    name: str
    machine: StepPath
    sources: FrozenSet[str]
    targets: Tuple[str, ...]
    trigger: Optional[EventOrigin] = None
    handler: Optional[StepPath] = None
    emits: Optional[EventOrigin] = None
    fields: Mapping[str, Tuple[FieldSource, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "machine", tuple(self.machine))
        object.__setattr__(self, "sources", frozenset(self.sources))
        object.__setattr__(self, "targets", tuple(self.targets))
        if self.handler is not None:
            object.__setattr__(self, "handler", tuple(self.handler))
        fields = {name: tuple(value) for name, value in self.fields.items()}
        object.__setattr__(self, "fields", MappingProxyType(fields))
        where = f"Transition '{self.label}'"
        if (self.trigger is None) == (self.handler is None):
            raise ContractError(f"{where} needs exactly one of a trigger or a handler")
        if self.trigger is not None and not isinstance(self.trigger, EventOrigin):
            raise ContractError(f"{where} trigger must be an EventOrigin")
        if not self.sources or not self.targets:
            raise ContractError(f"{where} needs 'from' and 'to' states")
        if self.is_fixed and len(self.targets) != 1:
            raise ContractError(
                f"{where} is fixed and needs exactly one target, got "
                f"{list(self.targets)}"
            )
        if self.emits is None:
            if self.fields:
                raise ContractError(f"{where} declares fields but emits no event")
            return
        if self.emits.kind != "machine" or self.emits.path != self.machine:
            raise ContractError(
                f"{where} must emit an event of its own machine, got "
                f"{self.emits.selector}"
            )
        for name, (source, value) in fields.items():
            if source == "event" and not self.is_fixed:
                raise ContractError(
                    f"{where} field '{name}' reads $event.{value}, but a "
                    "handler-selected transition does not retain the payload of "
                    "the event that started the handler"
                )
            if source == "transition" and value not in TRANSITION_VALUES:
                raise ContractError(
                    f"{where} field '{name}' reads $transition.{value}; known "
                    f"values are {list(TRANSITION_VALUES)}"
                )
            if source == "literal":
                _check_portable(value, where=f"{where} field '{name}'")
            elif source not in ("event", "transition"):
                raise ContractError(
                    f"{where} field '{name}' has unknown source {source!r}"
                )

    @property
    def is_fixed(self) -> bool:
        """Whether an event triggers the transition (not a handler)."""
        return self.trigger is not None

    @property
    def label(self) -> str:
        """Machine and transition name, e.g. ``child/m.start_review``."""
        return f"{scoped_name(self.machine)}.{self.name}"

    @property
    def event_fields(self) -> FrozenSet[str]:
        """Trigger event fields the emitted event copies (``$event.<f>``)."""
        return frozenset(
            value for source, value in self.fields.values() if source == "event"
        )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        texts = {"event": "$event.{}", "transition": "$transition.{}"}
        description = {
            "from": sorted(self.sources),
            "to": list(self.targets),
            "on": self.trigger.selector if self.trigger is not None else None,
            "handler": scoped_name(self.handler) if self.handler else None,
            "emit": self.emits.selector if self.emits is not None else None,
            "fields": {
                name: texts[source].format(value) if source in texts else value
                for name, (source, value) in self.fields.items()
            },
        }

        return description


@dataclass(frozen=True)
class PlannedMachine:
    """A compiled state machine: one record per source or one global record.

    Args:
        path: Declaring scope plus machine name, e.g. ``("child", "gate")``.
        scope: ``"source"`` keeps one state per source, ``"global"`` one
            state for the whole execution.
        initial: State of a fresh record.
        states: Declared states.
        transitions: Transitions in declaration order.
        events: Emitted event name to its payload declaration.

    Raises:
        ContractError: On bad states, transitions of another machine, repeated
            transition names, fixed transitions of one trigger with
            overlapping ``from`` states, emitted events that are not declared
            or do not match, or a system trigger of a per-source machine.
    """

    path: StepPath
    scope: MachineScope
    initial: str
    states: Tuple[str, ...]
    transitions: Tuple[PlannedTransition, ...] = ()
    events: Mapping[str, Event] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "events", MappingProxyType(dict(self.events)))
        where = f"State machine '{self.label}'"
        if not self.path:
            raise ContractError("State machine path must not be empty")
        if self.scope not in MACHINE_SCOPES:
            raise ContractError(
                f"{where} scope must be one of {list(MACHINE_SCOPES)}, got "
                f"{self.scope!r}"
            )
        if not self.states or len(set(self.states)) != len(self.states):
            raise ContractError(f"{where} needs distinct states, got {self.states}")
        if self.initial not in self.states:
            raise ContractError(
                f"{where} initial state {self.initial!r} is not one of {self.states}"
            )
        for name, event in self.events.items():
            if not isinstance(event, Event):
                raise ContractError(f"{where} event '{name}' must be an Event")

        names = set()
        chosen: Dict[EventOrigin, List[PlannedTransition]] = {}
        for transition in self.transitions:
            _check_machine_transition(self, transition, where=where)
            if transition.name in names:
                raise ContractError(
                    f"{where} declares transition '{transition.name}' twice"
                )
            names.add(transition.name)
            if transition.trigger is None:
                continue
            for other in chosen.setdefault(transition.trigger, []):
                shared = sorted(other.sources & transition.sources)
                if shared:
                    raise ContractError(
                        f"{where} transitions '{other.name}' and "
                        f"'{transition.name}' both fire on "
                        f"{transition.trigger.selector} from {shared}; one event "
                        "selects at most one transition, so their 'from' states "
                        "must not overlap"
                    )
            chosen[transition.trigger].append(transition)

    @property
    def name(self) -> str:
        """Machine name inside its declaring scope."""
        return self.path[-1]

    @property
    def label(self) -> str:
        """Scoped machine name, e.g. ``child/gate``."""
        return scoped_name(self.path)

    def origin(self, event: str) -> EventOrigin:
        """Origin of an event this machine emits."""
        return EventOrigin(kind="machine", event=event, path=self.path)

    def transition(self, name: str) -> PlannedTransition:
        """Return the transition named ``name``.

        Raises:
            KeyError: When the machine has no such transition.
        """
        for transition in self.transitions:
            if transition.name == name:
                return transition

        raise KeyError(name)

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "scope": self.scope,
            "initial_state": self.initial,
            "states": list(self.states),
            "transitions": {item.name: item.describe() for item in self.transitions},
            "events": {name: event.describe() for name, event in self.events.items()},
        }

        return description


def _check_machine_transition(
    machine: PlannedMachine, transition: PlannedTransition, *, where: str
) -> None:
    if not isinstance(transition, PlannedTransition):
        raise ContractError(f"{where} transitions must be PlannedTransition objects")
    if transition.machine != machine.path:
        raise ContractError(
            f"{where} holds transition '{transition.label}' of another machine"
        )
    unknown = sorted(
        (transition.sources | set(transition.targets)) - set(machine.states)
    )
    if unknown:
        raise ContractError(
            f"{where} transition '{transition.name}' uses unknown states {unknown}; "
            f"states are {list(machine.states)}"
        )
    trigger = transition.trigger
    if machine.scope == "source" and trigger is not None and trigger.kind == "system":
        raise ReactionContractError(
            f"{where} is per source, but transition '{transition.name}' fires on "
            f"{trigger.selector}, which has no source. Declare the machine with "
            "scope 'global'",
            compile_error=SelectorError,
            step_path=machine.path,
            field_path=("on",),
        )
    if transition.emits is None:
        return
    event = machine.events.get(transition.emits.event)
    if event is None or set(event.fields) != set(transition.fields):
        raise ContractError(
            f"{where} transition '{transition.name}' emits "
            f"'{transition.emits.event}' with fields {sorted(transition.fields)}, "
            f"but the machine declares {_event_text(event)}"
        )


def _event_text(event: Optional[Event]) -> str:
    if event is None:
        return "no such event"

    return f"fields {sorted(event.fields)}"


@dataclass(frozen=True)
class ReactionPlan:
    """All reactions of one compiled workflow.

    Args:
        handlers: Compiled handlers of every scope, in declaration order.
        signals: Declared external signals by name.
        groups: Output groups anchored at handler outputs.
        state: Declared initial managed-state values.
        machines: State machines of every scope, in declaration order.

    Raises:
        ContractError: On repeated handler, group or machine identities, a
            group of an unknown handler or output, a subscription to an
            undeclared signal, system or machine event (or with another
            payload), a handler-selected transition of an unknown handler, an
            unauthorized ``state_machine_set`` step in a handler plan, or an
            automatic event cascade cycle. Setter steps and handler groups
            raise ``ReactionContractError``.
    """

    handlers: Tuple[PlannedHandler, ...] = ()
    signals: Mapping[str, PlannedSignal] = field(
        default_factory=lambda: MappingProxyType({})
    )
    groups: Tuple[PlannedHandlerGroup, ...] = ()
    state: StateDefaults = field(default_factory=StateDefaults)
    machines: Tuple[PlannedMachine, ...] = ()
    _by_origin: Mapping[EventOrigin, Tuple[PlannedHandler, ...]] = field(
        init=False, repr=False, compare=False
    )
    _by_path: Mapping[StepPath, PlannedHandler] = field(
        init=False, repr=False, compare=False
    )
    _machines: Mapping[StepPath, PlannedMachine] = field(
        init=False, repr=False, compare=False
    )
    _transitions: Mapping[EventOrigin, Tuple[PlannedTransition, ...]] = field(
        init=False, repr=False, compare=False
    )
    _settable: Mapping[StepPath, Tuple[PlannedTransition, ...]] = field(
        init=False, repr=False, compare=False
    )

    EMPTY: ClassVar["ReactionPlan"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "handlers", tuple(self.handlers))
        object.__setattr__(self, "signals", MappingProxyType(dict(self.signals)))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "machines", tuple(self.machines))
        for name, signal in self.signals.items():
            if signal.name != name:
                raise ContractError(f"Signal '{signal.name}' is keyed as '{name}'")

        machines: Dict[StepPath, PlannedMachine] = {}
        transitions: Dict[EventOrigin, List[PlannedTransition]] = {}
        for machine in self.machines:
            if not isinstance(machine, PlannedMachine):
                raise ContractError("Reaction machines must be PlannedMachine objects")
            if machine.path in machines:
                raise ContractError(
                    f"State machine '{machine.label}' is declared twice"
                )
            machines[machine.path] = machine
            for transition in machine.transitions:
                if transition.trigger is not None:
                    transitions.setdefault(transition.trigger, []).append(transition)
        object.__setattr__(self, "_machines", MappingProxyType(machines))
        object.__setattr__(
            self,
            "_transitions",
            MappingProxyType({key: tuple(value) for key, value in transitions.items()}),
        )

        by_path: Dict[StepPath, PlannedHandler] = {}
        by_origin: Dict[EventOrigin, List[PlannedHandler]] = {}
        for handler in self.handlers:
            if handler.path in by_path:
                raise ContractError(
                    f"Handler '{scoped_name(handler.path)}' is declared twice"
                )
            by_path[handler.path] = handler
            by_origin.setdefault(handler.origin, []).append(handler)
            self._check_subscription(
                handler.origin,
                handler.event,
                who=f"Handler '{scoped_name(handler.path)}'",
            )
        object.__setattr__(self, "_by_path", MappingProxyType(by_path))
        object.__setattr__(
            self,
            "_by_origin",
            MappingProxyType({key: tuple(value) for key, value in by_origin.items()}),
        )

        for trigger, triggered in self._transitions.items():
            if trigger.kind != "step":
                self._check_subscription(
                    trigger, None, who=f"Transition '{triggered[0].label}'"
                )
        settable: Dict[StepPath, List[PlannedTransition]] = {}
        for machine in self.machines:
            for transition in machine.transitions:
                if transition.handler is None:
                    continue
                if transition.handler not in by_path:
                    raise ContractError(
                        f"Transition '{transition.label}' is selected by unknown "
                        f"handler '{scoped_name(transition.handler)}'"
                    )
                settable.setdefault(transition.handler, []).append(transition)
        object.__setattr__(
            self,
            "_settable",
            MappingProxyType({key: tuple(value) for key, value in settable.items()}),
        )
        for handler in self.handlers:
            for step in handler.plan.steps:
                if step.spec.type != STATE_MACHINE_SET_TYPE:
                    continue
                try:
                    check_setter_step(self._machines, handler=handler.path, step=step)
                except ContractError as error:
                    raise ReactionContractError(
                        f"Handler '{scoped_name(handler.path)}' step "
                        f"'{scoped_name(step.path)}': {error}",
                        step_path=handler.path + step.path,
                    ) from error

        self._check_groups()
        _check_cascades(self)

    def _check_subscription(
        self, origin: EventOrigin, event: Optional[Event], *, who: str
    ) -> None:
        if origin.kind == "step":
            return
        try:
            declared = self.declared_event(origin)
        except KeyError:
            declared = None
        if declared is None or (event is not None and declared != event):
            raise ContractError(
                f"{who} subscribes to {origin.selector}, which is not a declared "
                f"{origin.kind} event with that payload"
            )

    def _check_groups(self) -> None:
        group_names = set()
        for group in self.groups:
            if group.name in group_names:
                raise ContractError(f"Handler group '{group.name}' is declared twice")
            group_names.add(group.name)
            selector = f"$handlers.{scoped_name(group.handler)}"
            handler = self._by_path.get(group.handler)
            if handler is None:
                raise ReactionContractError(
                    f"Handler group '{group.name}' is anchored at unknown handler "
                    f"{selector}; handlers: {sorted(map(scoped_name, self._by_path))}",
                    compile_error=SelectorError,
                    field_path=("anchor",),
                )
            unknown = sorted(set(group.fields.values()) - set(handler.outputs))
            if unknown:
                raise ReactionContractError(
                    f"Handler group '{group.name}' selects outputs {unknown} that "
                    f"{selector} does not declare; its workflow outputs are "
                    f"{list(handler.outputs)}",
                    compile_error=SelectorError,
                    field_path=("outputs",),
                )

    @property
    def is_empty(self) -> bool:
        """Whether the workflow declares no handlers, signals, machines or state."""
        return (
            not self.handlers
            and not self.signals
            and not self.machines
            and self.state.is_empty
        )

    @property
    def emitting_steps(self) -> FrozenSet[StepPath]:
        """Step paths with at least one subscribed handler or transition."""
        origins = [*self._by_origin, *self._transitions]

        return frozenset(origin.path for origin in origins if origin.kind == "step")

    def subscribed(self, origin: EventOrigin) -> bool:
        """Whether a handler or a fixed transition listens to ``origin``."""
        return origin in self._by_origin or origin in self._transitions

    def handlers_for(self, origin: EventOrigin) -> Tuple[PlannedHandler, ...]:
        """Handlers subscribed to ``origin``, in declaration order.

        Args:
            origin: Event origin.

        Returns:
            Subscribed handlers; empty when nobody listens.
        """
        return self._by_origin.get(origin, ())

    def step_handlers(self, path: StepPath, event: str) -> Tuple[PlannedHandler, ...]:
        """Handlers subscribed to one step event, in declaration order.

        Args:
            path: Emitting step path.
            event: Event name.

        Returns:
            Subscribed handlers; empty when nobody listens.
        """
        return self._by_origin.get(EventOrigin(kind="step", event=event, path=path), ())

    def transitions_for(self, origin: EventOrigin) -> Tuple[PlannedTransition, ...]:
        """Fixed transitions triggered by ``origin``.

        Machines come in declaration order, and so do the transitions of one
        machine. From-states of one machine and trigger are disjoint, so at
        most one transition per machine matches its current state.

        Args:
            origin: Event origin.

        Returns:
            Triggered transitions; empty when none.
        """
        return self._transitions.get(origin, ())

    def bound_fields(self, origin: EventOrigin) -> FrozenSet[str]:
        """Union of event fields any subscriber of ``origin`` reads.

        Handlers read their ``$event`` bindings; fixed transitions read the
        fields their emitted event copies.

        Args:
            origin: Event origin.

        Returns:
            Field names to retain for that event; others can be dropped.
        """
        fields = frozenset(
            name
            for subscriber in (
                *self.handlers_for(origin),
                *self.transitions_for(origin),
            )
            for name in (
                subscriber.bound_fields
                if isinstance(subscriber, PlannedHandler)
                else subscriber.event_fields
            )
        )

        return fields

    def handler(self, path: StepPath) -> PlannedHandler:
        """Return the handler at ``path``.

        Raises:
            KeyError: When no handler has that path.
        """
        return self._by_path[tuple(path)]

    def machine(self, path: StepPath) -> PlannedMachine:
        """Return the state machine at ``path``.

        Raises:
            KeyError: When no machine has that path.
        """
        return self._machines[tuple(path)]

    def causes(self, handler: StepPath) -> FrozenSet[EventOrigin]:
        """Machine events a handler can cause through the transitions it sets.

        Derived from the machines' handler-selected transitions; the cascade
        check uses the same transitions.
        """
        settable = self._settable.get(tuple(handler), ())
        caused = frozenset(
            transition.emits for transition in settable if transition.emits is not None
        )

        return caused

    def setter(
        self, handler: StepPath, machine_ref: str, transition: str
    ) -> PlannedTransition:
        """Authorize a handler to apply a handler-selected transition.

        Args:
            handler: Path of the handler whose run sets the machine.
            machine_ref: Machine name relative to the handler's declaring
                scope, e.g. ``"gate"`` or ``"child/gate"``.
            transition: Transition name.

        Returns:
            The authorized transition.

        Raises:
            ContractError: On an unknown handler, machine or transition, a
                fixed transition, or a transition another handler selects.
        """
        handler = tuple(handler)
        if handler not in self._by_path:
            raise ContractError(f"Unknown handler '{scoped_name(handler)}'")
        planned = authorize_setter(
            self._machines,
            handler=handler,
            machine_ref=machine_ref,
            transition=transition,
        )

        return planned

    def declared_event(self, origin: EventOrigin) -> Event:
        """Payload declaration of a signal, system or machine event.

        Step events are declared by their block's spec, not here.

        Raises:
            KeyError: When ``origin`` is a step event or is not declared.
        """
        if origin.kind == "signal":
            return self.signals[origin.event].event
        if origin.kind == "system":
            return SYSTEM_EVENTS[origin.event]
        if origin.kind == "machine":
            return self._machines[origin.path].events[origin.event]

        raise KeyError(origin.selector)

    def groups_of(self, path: StepPath) -> Tuple[PlannedHandlerGroup, ...]:
        """Output groups anchored at handler ``path``."""
        return tuple(group for group in self.groups if group.handler == tuple(path))

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "handlers": {
                scoped_name(handler.path): {
                    **handler.describe(),
                    "causes": sorted(
                        origin.selector for origin in self.causes(handler.path)
                    ),
                }
                for handler in self.handlers
            },
            "signals": {
                name: signal.event.describe() for name, signal in self.signals.items()
            },
            "handler_groups": {group.name: group.describe() for group in self.groups},
            "state": self.state.describe(),
            "state_machines": {
                machine.label: machine.describe() for machine in self.machines
            },
        }

        return description


def authorize_setter(
    machines: Mapping[StepPath, PlannedMachine],
    *,
    handler: StepPath,
    machine_ref: str,
    transition: str,
) -> PlannedTransition:
    """Resolve and authorize one handler-selected transition.

    Args:
        machines: Machines by path.
        handler: Path of the handler whose run sets the machine.
        machine_ref: Machine name relative to the handler's declaring scope.
        transition: Transition name.

    Returns:
        The transition ``handler`` may apply.

    Raises:
        ContractError: On an unknown machine or transition, a fixed
            transition, or a transition another handler selects.
    """
    path = tuple(handler[:-1]) + tuple(str(machine_ref).split("/"))
    machine = machines.get(path)
    if machine is None:
        raise ContractError(
            f"Handler '{scoped_name(handler)}' names state machine {machine_ref!r}, "
            f"but its workflow has no machine '{scoped_name(path)}'; machines: "
            f"{sorted(scoped_name(item) for item in machines)}"
        )
    try:
        planned = machine.transition(transition)
    except KeyError:
        raise ContractError(
            f"State machine '{machine.label}' has no transition {transition!r}; "
            f"transitions: {[item.name for item in machine.transitions]}"
        ) from None
    if planned.handler != tuple(handler):
        selected_by = (
            f"fires on {planned.trigger.selector}"
            if planned.trigger is not None
            else f"is selected by handler '{scoped_name(planned.handler)}'"
        )
        raise ContractError(
            f"Transition '{planned.label}' {selected_by}; handler "
            f"'{scoped_name(handler)}' may not set it"
        )

    return planned


def check_setter_step(
    machines: Mapping[StepPath, PlannedMachine], *, handler: StepPath, step: Any
) -> PlannedTransition:
    """Check a ``state_machine_set`` step of a handler plan.

    ``machine`` and ``transition`` must be literal and authorized for the
    handler; a literal ``next_state`` must be one of the transition's
    targets. A selected ``next_state`` is checked when the step runs.

    Args:
        machines: Machines by path.
        handler: Path of the handler whose plan holds the step.
        step: Planned step of type ``STATE_MACHINE_SET_TYPE``.

    Returns:
        The transition the step applies.

    Raises:
        ContractError: When the step cannot apply its transition.
    """
    bound = {binding.field for binding in step.bindings}
    selected = [name for name in ("machine", "transition") if name in bound]
    if selected:
        raise ContractError(
            f"{selected} read selectors; machine and transition must be literal names"
        )
    params = step.params
    transition = authorize_setter(
        machines,
        handler=handler,
        machine_ref=params.machine,
        transition=params.transition,
    )
    if "next_state" not in bound and params.next_state not in transition.targets:
        raise ContractError(
            f"next_state {params.next_state!r} is not a target of transition "
            f"'{transition.label}', which may enter {list(transition.targets)}"
        )

    return transition


def _check_cascades(reactions: ReactionPlan) -> None:
    """Reject automatic event cascades that can reach their own origin again.

    Edges go per transition, not per machine, so a state graph that cycles
    on independent events is valid::

        trigger ──(transition m.t)──▶ emitted machine event
        origin  ──(handler h sets m.t)──▶ emitted machine event
    """
    edges: Dict[EventOrigin, List[Tuple[str, EventOrigin]]] = {}
    for machine in reactions.machines:
        for transition in machine.transitions:
            if transition.trigger is not None and transition.emits is not None:
                edges.setdefault(transition.trigger, []).append(
                    (f"transition {transition.label}", transition.emits)
                )
    for handler in reactions.handlers:
        for transition in reactions._settable.get(handler.path, ()):
            if transition.emits is not None:
                edges.setdefault(handler.origin, []).append(
                    (
                        f"handler {scoped_name(handler.path)} sets {transition.label}",
                        transition.emits,
                    )
                )

    finished: set = set()
    active: List[EventOrigin] = []
    labels: List[str] = []

    def visit(origin: EventOrigin) -> None:
        if origin in finished:
            return
        if origin in active:
            start = active.index(origin)
            chain = [active[start].selector]
            for label, item in zip(labels[start:], [*active[start + 1 :], origin]):
                chain.append(f"({label})")
                chain.append(item.selector)
            raise ContractError(
                "Events form an automatic cascade cycle: " + " -> ".join(chain)
            )
        active.append(origin)
        for label, caused in edges.get(origin, ()):
            labels.append(label)
            visit(caused)
            labels.pop()
        active.pop()
        finished.add(origin)

    for origin in list(edges):
        visit(origin)


ReactionPlan.EMPTY = ReactionPlan()
