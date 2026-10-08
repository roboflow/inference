"""Runtime controls: live enablement and configuration updates of a compiled plan.

A root ``controls`` section declares what a running session may change
without a new plan (``compilation.definition.ControlDeclaration``)::

    "controls": {
      "overlay":    {"type": "enable", "steps": ["$steps.painter"]},
      "tracking":   {"type": "enable", "steps": ["$steps.tracker"],
                     "enabled": false, "state": "reset_on_enable"},
      "confidence": {"type": "input", "input": "$inputs.confidence", "default": 0.5}
    }

The compiler turns it into ``CompiledWorkflow.controls`` (a ``ControlPlan``);
every ``ExecutionSession`` owns one ``ControlPanel`` over it::

    panel = session.controls
    receipt = panel.update(overlay=False, confidence=0.7)   # all-or-nothing
    receipt.version, receipt.boundary                        # acknowledgement
    panel.current                                            # the ControlSnapshot
    result.controls.version                                  # what a result saw

Boundary: an update publishes one immutable ``ControlSnapshot``; a source pulse
takes the current snapshot when it is admitted (under the driver's admission
lock), an operator pulse inherits the snapshot of the pulse whose ``push``
emitted it, an end-of-input or finish emission takes the snapshot of its
domain end (pipelined: the current one when the end was scheduled, under the
driver's lock; serial: the one the source end that caused it was sequenced
under), and a passive run takes it when it starts (a pipeline: at
``submit``). Every step, phase, callback and result of that pulse uses that
one snapshot; admitted work finishes with the version it was admitted under.
The active internals take the snapshot as a required argument and never
fall back to the current one. ``update()`` never waits for execution. ``input``
control values are built-in scalars (``SCALAR_TYPES``), so a published value
cannot change without a new version.

Omission: a disabled ``enable`` control stops its members: under
``reset_on_enable`` every member; under ``keep_ticking`` (default) only the
``prunable`` members no kept member needs (a non-prunable member keeps
ticking, with ``self.wants(name)`` answering ``False`` for every output nobody
live reads, and the prunable preprocessing it reads keeps running). A stopped
step is *omitted* in that pulse: not called, no outputs, its stages retired at
once. A result field reading a member's output has status ``"omitted"`` and no
key in ``rows()``, distinct from ``"filtered"`` (``None``) and from a valid
empty value.

Reset: enabling a ``reset_on_enable`` control at version ``v`` stamps its
``epoch``; the first call of a member whose pulse carries that epoch runs
``reset_state()`` once, after the pulse took its turn at the step and while no
other call of that step is in progress (``StepActivity``). Five ordering
rules make the reset causal, so no call admitted under an older version ever
runs against the reset instance:

1. Admission is version-monotonic: the snapshot is read where the work is
   sequenced, under the driver's lock (source admission, domain-end
   scheduling, passive ``submit``).
2. Dispatch is version-monotonic: the serial driver is a FIFO; the pipelined
   dispatcher hands out the lowest-version work first.
3. Inside one domain, stage tickets follow admission order.
4. Work whose snapshot carries a reset epoch ``e`` waits, before it takes an
   operator's ``push``/``end_input``/``finish`` turn, until no admitted work
   of a version below ``e`` is in flight (``InFlightWork``).
5. Work about to reset a static member (domain ``None``) of a plan with
   several domains waits the same way first: the member runs in every
   domain's route, while rule 3 orders turns within one domain only.

Rules 4 and 5 are the only pauses: a bounded drain in the window after a
reset-enable, holding no turn older work needs; older work never waits for
newer work (rule 2), so it cannot deadlock. Rule 5 waits only in pulses that
reach the member before its reset for that epoch. Ordinary updates change no
epoch and never wait.
"""

import importlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    ControlError,
    StepPath,
    format_step_path,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow

__all__ = [
    "ControlPanel",
    "ControlPlan",
    "ControlReceipt",
    "ControlSnapshot",
    "ControlView",
    "InFlightWork",
    "PlannedControl",
    "SCALAR_TYPES",
    "STATE_CLASSES",
    "StepActivity",
    "UPDATE_BOUNDARY",
    "non_scalar_problem",
    "producers_of",
]

DEMAND_MODULE = "roboflow_workflows.execution_engine.v2.compilation.demand"
INPUTS_MODULE = "roboflow_workflows.execution_engine.v2.execution.inputs"

STATE_CLASSES = ("pure", "local", "managed", "opaque")
"""How a member holds state: ``pure`` (prunable), ``local`` (declares
``reset_state``), ``managed`` (requests the shared ``managed_state`` resource)
or ``opaque`` (not prunable, no reset capability)."""

UPDATE_BOUNDARY = (
    "source pulses admitted after this call, operator pulses they feed, "
    "domain ends scheduled after this call and passive runs started after "
    "this call; admitted work finishes with the version it was admitted under"
)
"""Where a published update takes effect (``ControlReceipt.boundary``)."""

SCALAR_TYPES = (type(None), bool, int, float, str)
"""Exact Python types an ``input`` control value may have: built-in scalars.

Scalars are immutable, so a published snapshot, its receipts, views and
descriptions can share them with every caller and run without copies.
Structured values (``dict``, ``list``, arrays, custom objects, subclasses of
the scalar types) are refused: they could change after publication without a
new version. Non-finite floats (``nan``, ``inf``) are accepted: they are
immutable, though not strict JSON."""

DISABLED = "control {name!r} is disabled"
UNREAD = "no live reader while {names} disabled; the block is prunable"


@dataclass(frozen=True)
class PlannedControl:
    """One compiled control.

    Args:
        name: Control name.
        type: ``"enable"`` or ``"input"``.
        members: Listed member steps of an ``enable`` control, expanded
            (a nested step selector lists every step inside it), plan order.
        closure: Members plus every ``prunable`` step that transitively reads
            a member's output or decision, plan order. Disabling the control
            touches exactly these steps.
        downstream: Why each non-listed closure step belongs to it.
        enabled: Initial state.
        state: ``"keep_ticking"`` or ``"reset_on_enable"``.
        suspends_effects: Declared consent that disabling stops member effects.
        state_classes: ``STATE_CLASSES`` entry per closure step.
        input: Root input of an ``input`` control.
        default: Initial, validated value of an ``input`` control.
    """

    name: str
    type: str
    members: Tuple[StepPath, ...] = ()
    closure: Tuple[StepPath, ...] = ()
    downstream: Mapping[StepPath, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    enabled: bool = True
    state: str = "keep_ticking"
    suspends_effects: bool = False
    state_classes: Mapping[StepPath, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    input: Optional[str] = None
    default: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(tuple(p) for p in self.members))
        object.__setattr__(self, "closure", tuple(tuple(p) for p in self.closure))
        object.__setattr__(
            self,
            "downstream",
            MappingProxyType({tuple(p): r for p, r in self.downstream.items()}),
        )
        object.__setattr__(
            self,
            "state_classes",
            MappingProxyType({tuple(p): c for p, c in self.state_classes.items()}),
        )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        if self.type == "input":
            return {"type": "input", "input": self.input, "default": self.default}

        description = {
            "type": "enable",
            "enabled": self.enabled,
            "state": self.state,
            "suspends_effects": self.suspends_effects,
            "members": [format_step_path(path) for path in self.members],
            "closure": [format_step_path(path) for path in self.closure],
            "downstream": {
                format_step_path(path): reason
                for path, reason in self.downstream.items()
            },
            "state_classes": {
                format_step_path(path): kind
                for path, kind in self.state_classes.items()
            },
        }

        return description


@dataclass(frozen=True)
class ControlPlan:
    """Every control of a plan, by name (``CompiledWorkflow.controls``)."""

    controls: Mapping[str, PlannedControl] = field(
        default_factory=lambda: MappingProxyType({})
    )

    EMPTY: "ControlPlan" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))

    @property
    def is_empty(self) -> bool:
        """Whether the plan declares no controls."""
        return not self.controls

    @property
    def enable_controls(self) -> Dict[str, PlannedControl]:
        """The ``enable`` controls by name."""
        return {n: c for n, c in self.controls.items() if c.type == "enable"}

    @property
    def input_controls(self) -> Dict[str, PlannedControl]:
        """The ``input`` controls by name."""
        return {n: c for n, c in self.controls.items() if c.type == "input"}

    @property
    def controlled_inputs(self) -> Dict[str, str]:
        """Root input name to the control that owns it."""
        return {c.input: n for n, c in self.controls.items() if c.type == "input"}

    def controls_of(self, path: StepPath) -> Tuple[str, ...]:
        """Names of the ``enable`` controls whose closure contains ``path``."""
        path = tuple(path)
        names = tuple(
            name for name, c in self.enable_controls.items() if path in c.closure
        )

        return names

    def reset_members(self) -> FrozenSet[StepPath]:
        """Steps that ``reset_state()`` may be called on (``reset_on_enable`` members)."""
        members = frozenset(
            path
            for control in self.enable_controls.values()
            if control.state == "reset_on_enable"
            for path in control.closure
            if control.state_classes.get(path) == "local"
        )

        return members

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description, by control name."""
        return {name: control.describe() for name, control in self.controls.items()}


ControlPlan.EMPTY = ControlPlan()


@dataclass(frozen=True)
class ControlView:
    """What a result saw: the snapshot's version and settings, JSON-friendly.

    Args:
        version: Snapshot version; ``0`` is the compiled initial state.
        enabled: ``enable`` control name to its state.
        values: ``input`` control name to its value.
    """

    version: int
    enabled: Mapping[str, bool]
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", MappingProxyType(dict(self.enabled)))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {
            "version": self.version,
            "enabled": dict(self.enabled),
            "values": dict(self.values),
        }


@dataclass(frozen=True)
class ControlSnapshot:
    """One immutable, versioned state of every control and what follows from it.

    Args:
        version: Monotonic version; ``0`` is the compiled initial state.
        enabled: ``enable`` control name to its state.
        values: ``input`` control name to its validated value.
        epochs: ``enable`` control name to the version at which it last became
            enabled (``0`` for the initial state).
        reset_epoch: Newest epoch of an enabled ``reset_on_enable`` control;
            work carrying it drains older in-flight work before an operator
            entry (module docstring, rule 4). ``0`` when nothing may reset.
        omitted_steps: Steps a pulse under this snapshot does not run, with
            the reason naming the control(s).
        omitted_outputs: Flat workflow output names whose producer is omitted
            or suppressed; absent from results with status ``"omitted"``.
        omitted_fields: Output group name to such field names.
        wanted: Demanded outputs per running step under this snapshot;
            ``None`` when every control is enabled (the plan's demand applies).
    """

    version: int
    enabled: Mapping[str, bool]
    values: Mapping[str, Any]
    epochs: Mapping[str, int]
    reset_epoch: int = 0
    omitted_steps: Mapping[StepPath, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    omitted_outputs: FrozenSet[str] = frozenset()
    omitted_fields: Mapping[str, FrozenSet[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    wanted: Optional[Mapping[StepPath, FrozenSet[str]]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", MappingProxyType(dict(self.enabled)))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "epochs", MappingProxyType(dict(self.epochs)))
        object.__setattr__(
            self, "omitted_steps", MappingProxyType(dict(self.omitted_steps))
        )
        object.__setattr__(self, "omitted_outputs", frozenset(self.omitted_outputs))
        object.__setattr__(
            self,
            "omitted_fields",
            MappingProxyType({g: frozenset(f) for g, f in self.omitted_fields.items()}),
        )
        if self.wanted is not None:
            object.__setattr__(self, "wanted", MappingProxyType(dict(self.wanted)))

    @property
    def narrows(self) -> bool:
        """Whether this snapshot omits anything beyond the compiled plan."""
        return self.wanted is not None

    def omitted_reason(self, path: StepPath) -> Optional[str]:
        """Why ``path`` is omitted under this snapshot; ``None`` when it runs."""
        return self.omitted_steps.get(tuple(path))

    def wanted_outputs(
        self, path: StepPath, *, default: FrozenSet[str]
    ) -> FrozenSet[str]:
        """Outputs of ``path`` some live reader demands; ``default`` when not narrowing."""
        if self.wanted is None:
            return default

        wanted = default & self.wanted.get(tuple(path), frozenset())

        return wanted

    def omitted_in(self, group: Optional[str]) -> FrozenSet[str]:
        """Omitted output names of a group (``None``: the flat outputs)."""
        if group is None:
            return self.omitted_outputs

        return self.omitted_fields.get(group, frozenset())

    def view(self) -> ControlView:
        """The result-facing projection of this snapshot."""
        return ControlView(
            version=self.version, enabled=self.enabled, values=self.values
        )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            **self.view().describe(),
            "epochs": dict(self.epochs),
            "reset_epoch": self.reset_epoch,
            "omitted_steps": {
                format_step_path(path): reason
                for path, reason in self.omitted_steps.items()
            },
            "omitted_outputs": sorted(self.omitted_outputs),
            "omitted_fields": {
                group: sorted(fields) for group, fields in self.omitted_fields.items()
            },
        }

        return description


@dataclass(frozen=True)
class ControlReceipt:
    """Acknowledgement of one published update.

    Args:
        version: The published snapshot's version.
        previous_version: The version it replaced.
        changes: Control name to ``{"from": old, "to": new}``.
        boundary: Where the update takes effect (``UPDATE_BOUNDARY``).
    """

    version: int
    previous_version: int
    changes: Mapping[str, Mapping[str, Any]]
    boundary: str = UPDATE_BOUNDARY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "changes",
            MappingProxyType(
                {n: MappingProxyType(dict(c)) for n, c in self.changes.items()}
            ),
        )

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {
            "version": self.version,
            "previous_version": self.previous_version,
            "changes": {name: dict(change) for name, change in self.changes.items()},
            "boundary": self.boundary,
        }


class ControlPanel:
    """The one writer of a session's controls; publishes immutable snapshots.

    Args:
        plan: The compiled plan whose ``controls`` this panel drives.
    """

    def __init__(self, plan: "CompiledWorkflow"):
        self._plan = plan
        self._lock = threading.Lock()
        self.in_flight = InFlightWork()
        controls = plan.controls
        enabled = {n: c.enabled for n, c in controls.enable_controls.items()}
        values = {n: c.default for n, c in controls.input_controls.items()}
        epochs = {n: 0 for n in enabled}
        self._snapshot = self._build(0, enabled=enabled, values=values, epochs=epochs)

    @property
    def plan(self) -> "CompiledWorkflow":
        """The plan the panel belongs to."""
        return self._plan

    @property
    def current(self) -> ControlSnapshot:
        """The latest published snapshot."""
        return self._snapshot

    @property
    def version(self) -> int:
        """Version of the latest published snapshot."""
        return self._snapshot.version

    def describe(self) -> Dict[str, Any]:
        """Describe the declared controls and the current snapshot."""
        return {
            "declared": self._plan.controls.describe(),
            "current": self._snapshot.describe(),
            "in_flight": self.in_flight.describe(),
            "boundary": UPDATE_BOUNDARY,
        }

    def update(self, **changes: Any) -> ControlReceipt:
        """Validate every change, then publish one new snapshot (all or nothing).

        Args:
            **changes: Control name to its new value: ``bool`` for an
                ``enable`` control, a built-in scalar (``SCALAR_TYPES``) of the
                input's kinds for an ``input`` control.

        Returns:
            The receipt naming the new version and where it takes effect.

        Raises:
            ControlError: For no change, an unknown control, a wrong type or
                kind, a value that is not a built-in scalar, or a change that
                needs a new compiled plan. Nothing is applied and the current
                snapshot stays.
        """
        controls = self._plan.controls
        if not changes:
            raise ControlError("update() needs at least one <control>=<value> change")
        unknown = sorted(name for name in changes if name not in controls.controls)
        if unknown:
            raise ControlError(self._unknown_message(unknown))
        normalized = {
            name: self._normalize(controls.controls[name], value)
            for name, value in changes.items()
        }
        with self._lock:
            current = self._snapshot
            version = current.version + 1
            enabled = dict(current.enabled)
            values = dict(current.values)
            epochs = dict(current.epochs)
            receipt_changes: Dict[str, Dict[str, Any]] = {}
            for name, value in normalized.items():
                if controls.controls[name].type == "enable":
                    previous = enabled[name]
                    if value and not previous:
                        epochs[name] = version
                    enabled[name] = value
                else:
                    previous = values[name]
                    values[name] = value
                receipt_changes[name] = {"from": previous, "to": value}
            self._snapshot = self._build(
                version, enabled=enabled, values=values, epochs=epochs
            )
        receipt = ControlReceipt(
            version=version, previous_version=current.version, changes=receipt_changes
        )

        return receipt

    def _unknown_message(self, unknown: List[str]) -> str:
        plan = self._plan
        declared = sorted(plan.controls.controls)
        hints = []
        for name in unknown:
            if name in plan.inputs:
                hints.append(
                    f"{name!r} is a workflow input without a control; declare "
                    f'{{"type": "input", "input": "$inputs.{name}"}} under controls'
                )
            elif any(step.path == (name,) for step in plan.steps):
                hints.append(
                    f"{name!r} is a step without a control; declare "
                    f'{{"type": "enable", "steps": ["$steps.{name}"]}} under controls'
                )
            elif name in ("quality", "implementation", "target"):
                hints.append(
                    f"{name!r} is chosen at compile time; compile a new plan with "
                    "the setting and start a new session"
                )
        message = f"unknown control(s) {unknown}; declared controls: {declared}"
        if hints:
            message += ". " + "; ".join(hints)

        return message

    def _normalize(self, control: PlannedControl, value: Any) -> Any:
        if control.type == "enable":
            if not isinstance(value, bool):
                raise ControlError(
                    f"control {control.name!r} is an enable control and takes True "
                    f"or False, got {value!r}"
                )
            return value

        planned = self._plan.inputs[control.input]
        rejected = (
            f"control {control.name!r} rejected {value!r} for input "
            f"{control.input!r} of kinds {list(planned.kinds)}"
        )
        problem = non_scalar_problem(value)
        if problem is not None:
            raise ControlError(f"{rejected}: {problem}")

        inputs = importlib.import_module(INPUTS_MODULE)
        try:
            decoded = inputs.decode_payload(
                value,
                kinds=inputs.kinds_named(self._plan, planned.kinds),
                location=f"control {control.name!r} (input {control.input!r})",
            )
        except Exception as error:
            raise ControlError(f"{rejected}: {error}") from error
        problem = non_scalar_problem(decoded)
        if problem is not None:
            raise ControlError(f"{rejected}: after kind decoding, {problem}")

        return decoded

    def _build(
        self,
        version: int,
        *,
        enabled: Mapping[str, bool],
        values: Mapping[str, Any],
        epochs: Mapping[str, int],
    ) -> ControlSnapshot:
        plan = self._plan
        controls = plan.controls
        reset_epoch = max(
            (
                epochs[name]
                for name, control in controls.enable_controls.items()
                if control.state == "reset_on_enable" and enabled[name]
            ),
            default=0,
        )
        disabled = {
            name: controls.controls[name]
            for name, state in enabled.items()
            if not state
        }
        if not disabled:
            return ControlSnapshot(
                version=version,
                enabled=enabled,
                values=values,
                epochs=epochs,
                reset_epoch=reset_epoch,
            )

        # ``touched``: every closure step of a disabled control; its visible
        # outputs are omitted. ``stopped``: steps that may not run at all
        # (every member of a disabled reset_on_enable control). A touched
        # keep_ticking member runs when it is not prunable (it keeps ticking)
        # or when a kept member still needs it; otherwise it is omitted.
        stopped: Dict[StepPath, str] = {}
        touched: Dict[StepPath, str] = {}
        for name, control in disabled.items():
            for path in control.closure:
                touched.setdefault(path, name)
                if control.state == "reset_on_enable":
                    stopped.setdefault(path, name)
        omitted_outputs = set()
        omitted_fields: Dict[str, set] = {}
        roots = []
        for group, output in _result_fields(plan):
            producers = producers_of(plan, output.source)
            if producers & set(touched):
                if group is None:
                    omitted_outputs.add(output.name)
                else:
                    omitted_fields.setdefault(group, set()).add(output.name)
                continue
            where = (
                f"output {output.name!r}"
                if group is None
                else (f"group {group!r} field {output.name!r}")
            )
            roots.append((output.source, f"live {where}"))
        for operator in plan.operators.values():
            for item in operator.inputs:
                roots.append(
                    (item.source, f"$operators.{operator.name} input {item.name!r}")
                )
        demand_module = importlib.import_module(DEMAND_MODULE)
        demand = demand_module.compute_demand(
            plan,
            root_sources=roots,
            always=[
                (step.path, demand_module.NOT_PRUNABLE)
                for step in plan.steps
                if not step.spec.prunable and step.path not in stopped
            ],
        )
        running_stopped = sorted(
            set(demand.retained) & set(stopped), key=format_step_path
        )
        if running_stopped:
            raise ContractError(
                "a live reader still needs stopped step(s) "
                f"{[format_step_path(p) for p in running_stopped]}; the control "
                "closure check should have rejected this plan"
            )
        omitted_steps: Dict[StepPath, str] = {}
        disabled_names = ", ".join(repr(name) for name in sorted(disabled))
        for step in plan.steps:
            if step.path in demand.retained:
                continue
            if step.path in touched:
                omitted_steps[step.path] = DISABLED.format(name=touched[step.path])
            else:
                omitted_steps[step.path] = UNREAD.format(names=disabled_names)
        snapshot = ControlSnapshot(
            version=version,
            enabled=enabled,
            values=values,
            epochs=epochs,
            reset_epoch=reset_epoch,
            omitted_steps=omitted_steps,
            omitted_outputs=frozenset(omitted_outputs),
            omitted_fields=omitted_fields,
            wanted=demand.wanted,
        )

        return snapshot


def _result_fields(plan: "CompiledWorkflow") -> List[Tuple[Optional[str], Any]]:
    """Every result field: ``(None, output)`` flat, ``(group, field)`` grouped."""
    fields: List[Tuple[Optional[str], Any]] = [(None, o) for o in plan.outputs]
    for group in plan.output_groups:
        fields.extend((group.name, output) for output in group.outputs)

    return fields


def producers_of(plan: "CompiledWorkflow", source: Any) -> FrozenSet[StepPath]:
    """Steps whose omission makes ``source`` unreadable.

    Args:
        plan: The compiled plan.
        source: A value source: step port, child boundary port, input,
            source port or constant.

    Returns:
        The producing step of a step port (followed through nested
        boundaries) plus the controllers of every gated child output on the
        way; empty for inputs, source ports and constants.
    """
    from roboflow_workflows.execution_engine.v2.plan import (  # local: no cycle at import
        ChildInputPort,
        ChildOutputPort,
        PlannedChildOutput,
        StepPort,
    )

    boundaries = {item.port: item for item in plan.child_inputs + plan.child_outputs}
    producers = set()
    while isinstance(source, (ChildInputPort, ChildOutputPort)):
        boundary = boundaries[source]
        if isinstance(boundary, PlannedChildOutput):
            producers.update(gate.controller for gate in boundary.gates)
        source = boundary.source
    if isinstance(source, StepPort):
        producers.add(source.step)

    return frozenset(producers)


class StepActivity:
    """Shared/exclusive guard of one step's calls, for a causally ordered reset.

    Calls of the step hold it shared; a reset holds it exclusively, which
    waits until every in-progress call of the step (from any source, at any
    phase) finished and keeps new calls waiting meanwhile. Only the steps a
    ``reset_on_enable`` control may reset have one, so ordinary steps pay
    nothing. Waiters observe ``aborted`` so a failing run never hangs here.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._exclusive = False
        self._pending = 0

    @contextmanager
    def shared(
        self, aborted: Optional[Callable[[], Optional[BaseException]]] = None
    ) -> Iterator[None]:
        """Hold the step for one call; waits while a reset is pending or running.

        Args:
            aborted: Returns the exception to raise once the run aborts, so a
                waiter never hangs; ``None`` waits without that check.
        """
        with self._condition:
            self._wait_until(lambda: not (self._exclusive or self._pending), aborted)
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    @contextmanager
    def exclusive(
        self, aborted: Optional[Callable[[], Optional[BaseException]]] = None
    ) -> Iterator[None]:
        """Hold the step alone; waits for in-progress calls, blocks new ones."""
        with self._condition:
            self._pending += 1
            try:
                self._wait_until(lambda: not (self._exclusive or self._active), aborted)
                self._exclusive = True
            finally:
                self._pending -= 1
        try:
            yield
        finally:
            with self._condition:
                self._exclusive = False
                self._condition.notify_all()

    def _wait_until(
        self,
        ready: Callable[[], bool],
        aborted: Optional[Callable[[], Optional[BaseException]]],
    ) -> None:
        while not ready():
            if aborted is not None:
                error = aborted()
                if error is not None:
                    raise error
            self._condition.wait(0.05)


class InFlightWork:
    """Admitted, unfinished work of a session counted per control version.

    Drivers ``enter`` a source pulse at admission and a domain end when they
    schedule it, and ``leave`` when it completed or was cancelled; operator
    pulses run inside the work that fed them and need no entry of their own.
    Work carrying reset epoch ``e`` calls ``wait_drained_below(e)`` before an
    operator entry and before it resets a static member (module docstring,
    rules 4 and 5).
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._counts: Dict[int, int] = {}
        self._oldest: Optional[int] = None

    def enter(self, version: int) -> None:
        """Count one unit of work admitted under ``version``."""
        with self._condition:
            self._counts[version] = self._counts.get(version, 0) + 1
            if self._oldest is None or version < self._oldest:
                self._oldest = version

    def leave(self, version: int) -> None:
        """The work admitted under ``version`` completed or was cancelled."""
        with self._condition:
            remaining = self._counts[version] - 1
            if remaining:
                self._counts[version] = remaining
            else:
                del self._counts[version]
                self._oldest = min(self._counts) if self._counts else None
            self._condition.notify_all()

    def drained_below(self, epoch: int) -> bool:
        """Whether no work admitted under a version below ``epoch`` is in flight."""
        oldest = self._oldest

        return oldest is None or oldest >= epoch

    def wait_drained_below(
        self,
        epoch: int,
        aborted: Optional[Callable[[], Optional[BaseException]]] = None,
    ) -> bool:
        """Wait until ``drained_below(epoch)``; ``True`` when a wait happened.

        Args:
            epoch: The reset epoch the caller's snapshot carries.
            aborted: Returns the exception to raise once the run aborts, so a
                waiter never hangs; ``None`` waits without that check.
        """
        if self.drained_below(epoch):
            return False
        with self._condition:
            while not self.drained_below(epoch):
                if aborted is not None:
                    error = aborted()
                    if error is not None:
                        raise error
                self._condition.wait(0.05)

        return True

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description: units in flight per version."""
        with self._condition:
            return {
                str(version): count for version, count in sorted(self._counts.items())
            }


def non_scalar_problem(value: Any) -> Optional[str]:
    """Explain why ``value`` cannot be an ``input`` control value.

    Args:
        value: A control default or update value, before or after decoding.

    Returns:
        ``None`` for a built-in scalar (``SCALAR_TYPES``); otherwise the reason
        and the fix, for the caller's error message.
    """
    if type(value) in SCALAR_TYPES:
        return None

    problem = (
        f"control values are built-in scalars (None, bool, int, float, str), got "
        f"{type(value).__name__}; a structured or mutable value could change "
        "after publication without a new version. Split it into scalar "
        "controls, or pass it as an ordinary workflow input"
    )

    return problem
