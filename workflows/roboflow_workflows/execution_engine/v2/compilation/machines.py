"""Resolve event selectors and plan the state machines of a composed workflow.

One ``EventResolver`` answers every ``on`` selector, for machine transitions
and handlers alike, from the scope that declares the subscriber::

    $steps.<path>.events.<e>            the block step, walked downward
    $signals.<name>                     a root signal
    $system.events.started|ended        active workflows only
    $state_machines.<path>.events.<e>   the payload its transitions emit

A machine's path is its declaring scope plus its name, so a child used twice
(a diamond) yields two machines with their own records and handlers::

    scope ("a",)  state_machines["gate"] ──▶ PlannedMachine ("a", "gate")
    scope ("b",)  state_machines["gate"] ──▶ PlannedMachine ("b", "gate")

Emitted events get their payload kinds from the transitions that emit them::

    $event.<field>          kinds of that field of the trigger event
    $transition.from|to|name  string
    literal                 kind of the JSON value (null: wildcard)

A trigger may itself be a machine event, so emitted events resolve on demand
and once each. An event whose payload needs itself is an automatic cascade
cycle and is rejected here; ``ReactionPlan`` checks every other cascade.

``state_machine_set`` steps run only inside a handler workflow, for a
handler-selected transition that names that handler; ``ReactionPlan``
authorizes them.
"""

from typing import Any, Dict, Iterable, List, Mapping, Tuple

from roboflow_workflows.execution_engine.v2.compilation.composition import Scope
from roboflow_workflows.execution_engine.v2.compilation.definition import (
    EventSelector,
    MachineDeclaration,
    NestedStepDeclaration,
    TransitionDeclaration,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    SelectorError,
    StepPath,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    WILDCARD_KIND_NAME,
    Kind,
)
from roboflow_workflows.execution_engine.v2.plan import PlannedStep
from roboflow_workflows.execution_engine.v2.reactions.plan import (
    STATE_MACHINE_SET_TYPE,
    SYSTEM_EVENTS,
    EventOrigin,
    PlannedMachine,
    PlannedSignal,
    PlannedTransition,
    as_compile_error,
    scoped_name,
)

# JSON literal type to kind; bool before int, since bool is an int.
_LITERAL_KINDS: Tuple[Tuple[type, Kind], ...] = (
    (bool, BOOLEAN_KIND),
    (int, INTEGER_KIND),
    (float, FLOAT_KIND),
    (str, STRING_KIND),
    (dict, DICTIONARY_KIND),
    (list, LIST_OF_VALUES_KIND),
)


def plan_machines(events: "EventResolver") -> Tuple[PlannedMachine, ...]:
    """Plan the state machines of every scope, root first.

    Args:
        events: Resolver over the composition's scopes.

    Returns:
        Planned machines in declaration order.

    Raises:
        WorkflowCompileError: A subclass naming the machine, transition and
            reason.
    """
    machines = tuple(_plan_machine(events, path) for path in events.machines)

    return machines


def reject_main_flow_setters(steps: Iterable[PlannedStep]) -> None:
    """Reject ``state_machine_set`` steps outside handler workflows.

    Raises:
        WorkflowCompileError: On the first such step.
    """
    for step in steps:
        if step.spec.type == STATE_MACHINE_SET_TYPE:
            raise WorkflowCompileError(
                f"$steps.{scoped_name(step.path)} is a {STATE_MACHINE_SET_TYPE} step "
                "of the main flow; it runs only inside the workflow of the handler "
                "that a handler-selected transition names. Fire a transition from "
                "the main flow with 'on' and an event instead",
                step_path=step.path,
            )


class EventResolver:
    """Resolves ``on`` selectors of machine transitions and handlers.

    Args:
        scopes: Scopes of the composition, without handler workflows.
        steps: Planned block steps by path.
        signals: Root signals by name.
        active: Whether the workflow declares sources.
    """

    def __init__(
        self,
        scopes: Iterable[Scope],
        *,
        steps: Mapping[StepPath, PlannedStep],
        signals: Mapping[str, PlannedSignal],
        active: bool,
    ):
        self.machines: Dict[StepPath, Tuple[Scope, MachineDeclaration]] = {
            scope.path + (machine.name,): (scope, machine)
            for scope in scopes
            for machine in scope.workflow.machines
        }
        self._steps = steps
        self._signals = signals
        self._active = active
        self._triggers: Dict[Tuple[StepPath, str], Tuple[EventOrigin, Event]] = {}
        self._events: Dict[Tuple[StepPath, str], Event] = {}
        self._resolving: List[Tuple[StepPath, str]] = []

    def resolve(
        self, scope: Scope, on: EventSelector, *, where: str, path: StepPath
    ) -> Tuple[EventOrigin, Event]:
        """Resolve an ``on`` selector from its declaring scope.

        Args:
            scope: Scope declaring the subscriber.
            on: Parsed selector.
            where: Subscriber description for messages.
            path: Subscriber path for errors.

        Returns:
            The origin and the declaration of its payload.

        Raises:
            SelectorError: On an unknown step, signal, machine or event, or a
                system event of a passive workflow.
            WorkflowCompileError: On a machine event whose payload needs
                itself.
        """
        location = f"{where} on {on.text!r}"
        if on.kind == "signal":
            signal = self._signals.get(on.event)
            if signal is None:
                raise SelectorError(
                    f"{location} names an undeclared signal; the root workflow "
                    f"declares signals {sorted(self._signals)}",
                    step_path=path,
                    field_path=("on",),
                )
            return signal.origin, signal.event
        if on.kind == "system":
            if not self._active:
                raise SelectorError(
                    f"{location}: system events belong to an active run, but the "
                    "workflow declares no sources",
                    step_path=path,
                    field_path=("on",),
                )
            return EventOrigin(kind="system", event=on.event), SYSTEM_EVENTS[on.event]
        if on.kind == "machine":
            machine = scope.path + on.path
            event = self.machine_event(machine, on.event, location=location, path=path)
            return EventOrigin(kind="machine", event=on.event, path=machine), event

        return self._step_event(scope, on, location=location, path=path)

    def machine_event(
        self, machine: StepPath, name: str, *, location: str, path: StepPath
    ) -> Event:
        """Payload of event ``name`` of ``machine``; derived once.

        Args:
            machine: Machine path.
            name: Emitted event name.
            location: Subscriber and selector description for messages.
            path: Subscriber path for errors.

        Returns:
            The event, its field kinds unioned over the emitting transitions.

        Raises:
            SelectorError: On an unknown machine or event.
            WorkflowCompileError: On an event whose payload needs itself.
        """
        if machine not in self.machines:
            raise SelectorError(
                f"{location} names unknown state machine '{scoped_name(machine)}'; "
                f"machines: {sorted(scoped_name(item) for item in self.machines)}",
                step_path=path,
                field_path=("on",),
            )
        key = (machine, name)
        if key in self._events:
            return self._events[key]
        if key in self._resolving:
            chain = self._resolving[self._resolving.index(key) :] + [key]
            raise WorkflowCompileError(
                "State machine events form an automatic cascade cycle: "
                + " -> ".join(
                    f"$state_machines.{scoped_name(item)}.events.{event}"
                    for item, event in chain
                ),
                step_path=machine,
            )

        _, declaration = self.machines[machine]
        emitting = [item for item in declaration.transitions if item.emit == name]
        if not emitting:
            raise SelectorError(
                f"{location}: state machine '{scoped_name(machine)}' emits no event "
                f"{name!r}; it emits "
                f"{sorted({item.emit for item in declaration.transitions if item.emit})}",  # noqa: E501
                step_path=path,
                field_path=("on",),
            )
        self._resolving.append(key)
        try:
            event = self._emitted_event(machine, name, emitting)
        finally:
            self._resolving.pop()
        self._events[key] = event

        return event

    def trigger(
        self, machine: StepPath, transition: TransitionDeclaration
    ) -> Tuple[EventOrigin, Event]:
        """Resolve the ``on`` selector of a fixed transition; once each."""
        key = (machine, transition.name)
        if key not in self._triggers:
            scope, _ = self.machines[machine]
            self._triggers[key] = self.resolve(
                scope, transition.on, where=transition.location, path=machine
            )

        return self._triggers[key]

    def _step_event(
        self, scope: Scope, on: EventSelector, *, location: str, path: StepPath
    ) -> Tuple[EventOrigin, Event]:
        # Walk nested steps from the declaring scope: $steps.child/zone.
        current = scope
        for name in on.path[:-1]:
            step = current.workflow.step(name)
            if not isinstance(step, NestedStepDeclaration):
                raise SelectorError(
                    f"{location}: {name!r} is not a nested workflow step of "
                    f"{_scope_text(current)}; nested workflow steps there: "
                    f"{sorted(current.children)}",
                    step_path=path,
                    field_path=("on",),
                )
            current = current.children[name]

        name = on.path[-1]
        step = current.workflow.step(name)
        if step is None:
            raise SelectorError(
                f"{location} references unknown step {name!r} of "
                f"{_scope_text(current)}; steps there: "
                f"{[item.name for item in current.workflow.steps]}",
                step_path=path,
                field_path=("on",),
            )
        if isinstance(step, NestedStepDeclaration):
            raise SelectorError(
                f"{location} names nested workflow step {name!r}, which emits no "
                "events itself; subscribe to a step inside it as "
                f"$steps.{'/'.join(on.path)}/<step>.events.{on.event}",
                step_path=path,
                field_path=("on",),
            )

        emitter = self._steps[current.step_path(name)]
        event = emitter.spec.events.get(on.event)
        if event is None:
            raise SelectorError(
                f"{location}: block '{emitter.block_type}' declares no event "
                f"{on.event!r}; it declares {sorted(emitter.spec.events)}",
                step_path=path,
                field_path=("on",),
            )
        origin = EventOrigin(kind="step", event=on.event, path=emitter.path)

        return origin, event

    def _emitted_event(
        self, machine: StepPath, name: str, emitting: List[TransitionDeclaration]
    ) -> Event:
        first = emitting[0]
        for other in emitting[1:]:
            if set(other.fields) != set(first.fields):
                raise WorkflowCompileError(
                    f"{other.location} emits {name!r} with fields "
                    f"{sorted(other.fields)}, but {first.location} emits it with "
                    f"{sorted(first.fields)}; one machine event has one field set",
                    step_path=machine,
                )
        fields = {
            field_name: _union(
                self._field_kinds(machine, transition, field_name)
                for transition in emitting
            )
            for field_name in first.fields
        }

        return Event(fields)

    def _field_kinds(
        self, machine: StepPath, transition: TransitionDeclaration, field_name: str
    ) -> Tuple[Kind, ...]:
        source, value = transition.fields[field_name]
        if source == "transition":
            return (STRING_KIND,)
        if source == "literal":
            return _literal_kinds(value)

        _, event = self.trigger(machine, transition)
        if value not in event.fields:
            raise SelectorError(
                f"{transition.location}.emit.fields.{field_name} reads $event.{value}, "
                f"but {transition.on.text} declares fields {sorted(event.fields)}",
                step_path=machine,
                field_path=("emit", "fields", field_name),
            )

        return tuple(event.fields[value])


def _plan_machine(events: EventResolver, path: StepPath) -> PlannedMachine:
    scope, declaration = events.machines[path]
    transitions = tuple(
        _plan_transition(events, path, scope, item) for item in declaration.transitions
    )
    emitted = dict.fromkeys(
        item.emit for item in declaration.transitions if item.emit is not None
    )
    emitted_events = {
        name: events.machine_event(path, name, location=declaration.location, path=path)
        for name in emitted
    }
    try:
        machine = PlannedMachine(
            path=path,
            scope=declaration.scope,
            initial=declaration.initial,
            states=declaration.states,
            transitions=transitions,
            events=emitted_events,
        )
    except ContractError as error:
        raise as_compile_error(
            error, where=declaration.location, step_path=path
        ) from error

    return machine


def _plan_transition(
    events: EventResolver,
    path: StepPath,
    scope: Scope,
    declaration: TransitionDeclaration,
) -> PlannedTransition:
    trigger = handler = None
    if declaration.on is not None:
        trigger, _ = events.trigger(path, declaration)
    else:
        if declaration.handler not in scope.handlers:
            raise SelectorError(
                f"{declaration.location}.handler names {declaration.handler!r}, "
                "but the workflow declaring the machine has handlers "
                f"{sorted(scope.handlers)}; a handler-selected transition names "
                "a handler of the same workflow",
                step_path=path,
                field_path=("handler",),
            )
        handler = scope.step_path(declaration.handler)
    emits = None
    if declaration.emit is not None:
        emits = EventOrigin(kind="machine", event=declaration.emit, path=path)
    try:
        transition = PlannedTransition(
            name=declaration.name,
            machine=path,
            sources=frozenset(declaration.sources),
            targets=declaration.targets,
            trigger=trigger,
            handler=handler,
            emits=emits,
            fields=declaration.fields,
        )
    except ContractError as error:
        raise as_compile_error(
            error, where=declaration.location, step_path=path
        ) from error

    return transition


def _literal_kinds(value: Any) -> Tuple[Kind, ...]:
    for python_type, kind in _LITERAL_KINDS:
        if isinstance(value, python_type):
            return (kind,)

    return ()


def _scope_text(scope: Scope) -> str:
    if not scope.path:
        return "the root workflow"

    return f"nested workflow $steps.{scoped_name(scope.path)}"


def _union(kind_sets: Iterable[Tuple[Kind, ...]]) -> Tuple[Kind, ...]:
    merged: Dict[str, Kind] = {}
    for kinds in kind_sets:
        if not kinds or any(kind.name == WILDCARD_KIND_NAME for kind in kinds):
            return ()
        for kind in kinds:
            merged.setdefault(kind.name, kind)

    return tuple(merged.values())
