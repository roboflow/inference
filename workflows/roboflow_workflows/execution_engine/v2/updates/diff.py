"""Pure comparison of two compiled plans for a live graph update.

``compare_plans(old, new)`` constructs nothing and touches no session. It
matches steps by path and every other element by name, so reordered but
equivalent declarations compare equal. Each difference becomes one
``PlanChange`` with a stable reason code::

    retained   old step whose execution contract is unchanged: same block
               class and block contract, selected implementation and its
               contract, quality, execution mode, params, bindings, layout,
               outputs, gates, dependencies, domain
    added      new step with no old instance; ``newly_demanded`` when the old
               plan had pruned it
    breaking   anything a retained instance or the running graph could notice:
               a changed or removed step, source, operator, input, nested
               boundary, control, reaction, recording, catalogue provider or
               compile target; new managed state; a new compile warning

Every change also says the least disruptive update that applies it
(``requires``); a ``breaking`` change is one that requires more than
``preserve``::

    preserve      a preserving update keeps the graph and applies it
    reset         only a reset applies it: every step, operator, handler and
                  engine-owned state starts fresh, sources stay
    unsupported   no update applies it: a changed, added or removed source,
                  a changed input, recording, retrospective, or a catalogue
                  provider a source resolved a resource from

``PlanDiff.kind`` is the most demanding of them, in the same words. Inputs
keep their declared schema across a reset in this version; that is a
limitation of the first reset, not a rule of the graph.

Output selections may be added, changed or removed (``output`` and
``output_group``) and a retained step's wanted outputs may change: both are
read per call, never by a constructed instance. Removing an output is
compatible only while every step it fed is still demanded: removing the last
consumer of a step prunes the step, which is breaking (``newly_pruned``).

A changed step parameter is breaking even when the constructor never sees it.
A block may build state from run-time parameters on its first call and keep
it, e.g. a tracker created from its thresholds and cached per video. The
engine cannot see what an instance cached, so it never reuses an instance
under different parameters. Tune running graphs with controls instead.

Values are compared structurally without ``==`` on unknown objects, so an
array or tensor constant never decides equality through truth conversion:
such values are equal only when they are the same object.
"""

import dataclasses
import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from roboflow_workflows.execution_engine.v2.controls import PlannedControl
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    StepPath,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    PlannedStep,
    requests_managed_state,
)

ADDED = "added"
NEWLY_DEMANDED = "newly_demanded"
REMOVED = "removed"
NEWLY_PRUNED = "newly_pruned"
CHANGED = "changed"
WANTED_OUTPUTS_CHANGED = "wanted_outputs_changed"
PROVIDER_CHANGED = "provider_changed"

# The least disruptive update that applies a change (``PlanChange.requires``)
# or a whole plan (``PlanDiff.kind``, ``UpdateAssessment.kind``).
PRESERVE = "preserve"
RESET = "reset"
UNSUPPORTED = "unsupported"

_KINDS = (PRESERVE, RESET, UNSUPPORTED)


@dataclass(frozen=True, init=False)
class PlanChange:
    """One difference between two plans.

    ``requires`` is the one stored classification; ``breaking`` is read from
    it. A call written before ``requires`` existed passes ``breaking``
    instead: ``True`` stands for ``unsupported``, ``False`` for ``preserve``.

    Args:
        component: Kind of element: ``step``, ``source``, ``operator``,
            ``input``, ``control``, ``child_input``, ``child_output``,
            ``output``, ``output_group``, ``reactions``, ``recording``,
            ``retrospective``, ``catalogue``, ``target``, ``managed_state``
            or ``warning``; a session blocker (``UpdateAssessment``) also
            ``resource`` or ``run``.
        name: The element, e.g. ``$steps.counter`` or ``$sources.camera``.
        reason: Stable code, e.g. ``params_changed`` or ``added``.
        breaking: The older form of ``requires``; give one of the two.
        detail: Human-readable explanation.
        requires: ``preserve``, ``reset`` or ``unsupported``: the least
            disruptive update that applies the change.

    Raises:
        ContractError: When ``requires`` is unknown, or when neither or both
            of ``breaking`` and ``requires`` are given.
    """

    component: str
    name: str
    reason: str
    requires: str
    detail: str = ""

    def __init__(
        self,
        component: str,
        name: str,
        reason: str,
        breaking: Optional[bool] = None,
        detail: str = "",
        *,
        requires: Optional[str] = None,
    ) -> None:
        if (breaking is None) == (requires is None):
            raise ContractError(
                f"PlanChange {name} needs exactly one of requires and breaking"
            )
        if requires is None:
            requires = UNSUPPORTED if breaking else PRESERVE
        if requires not in _KINDS:
            raise ContractError(
                f"PlanChange requires must be one of {list(_KINDS)}, got {requires!r}"
            )
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "requires", requires)
        object.__setattr__(self, "detail", detail)

    @property
    def breaking(self) -> bool:
        """Whether a preserving update rejects the change."""
        return self.requires != PRESERVE

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {**dataclasses.asdict(self), "breaking": self.breaking}

        return description


@dataclass(frozen=True)
class PlanDiff:
    """What an update from one plan to another keeps, adds and breaks.

    Args:
        retained: Old step paths whose instances the new plan can reuse.
        added: New step paths that need fresh instances, in plan order.
        changes: Every difference, compatible or breaking.
    """

    retained: Tuple[StepPath, ...]
    added: Tuple[StepPath, ...]
    changes: Tuple[PlanChange, ...]

    @property
    def breaking(self) -> Tuple[PlanChange, ...]:
        """The changes a preserving update rejects: they require more."""
        return tuple(change for change in self.changes if change.breaking)

    @property
    def compatible(self) -> bool:
        """Whether the new plan can reuse the current graph."""
        return not self.breaking

    @property
    def unsupported(self) -> Tuple[PlanChange, ...]:
        """The changes no update can apply, not even a reset."""
        unsupported = tuple(
            change for change in self.changes if change.requires == UNSUPPORTED
        )

        return unsupported

    @property
    def kind(self) -> str:
        """``preserve``, ``reset`` or ``unsupported``: what the plans allow.

        ``preserve``: a preserving update applies every change. ``reset``:
        only a reset does. ``unsupported``: no update does. The session can
        add reasons of its own (``ExecutionSession.assess_update``).
        """
        if self.unsupported:
            return UNSUPPORTED
        if self.breaking:
            return RESET

        return PRESERVE

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {
            "kind": self.kind,
            "compatible": self.compatible,
            "retained": [format_step_path(path) for path in self.retained],
            "added": [format_step_path(path) for path in self.added],
            "changes": [change.describe() for change in self.changes],
        }


def compare_plans(old: CompiledWorkflow, new: CompiledWorkflow) -> PlanDiff:
    """Compare two compiled plans without constructing anything.

    Args:
        old: The plan a session currently runs.
        new: The candidate plan.

    Returns:
        The retained and added steps and every change with its reason.
    """
    changes: List[PlanChange] = []
    retained, added = _compare_steps(old, new, changes=changes)
    _compare_named(
        "source",
        old.sources,
        new.sources,
        prefix="$sources.",
        changes=changes,
        requires=UNSUPPORTED,
        detail=_SOURCE_DETAIL,
    )
    _compare_named(
        "operator", old.operators, new.operators, prefix="$operators.", changes=changes
    )
    _compare_named(
        "input",
        old.inputs,
        new.inputs,
        prefix="$inputs.",
        changes=changes,
        requires=UNSUPPORTED,
        detail=_INPUT_DETAIL,
    )
    _compare_controls(old.controls.controls, new.controls.controls, changes=changes)
    _compare_boundaries(old, new, changes=changes)
    _compare_selections(old, new, changes=changes)
    _compare_sections(old, new, changes=changes)

    diff = PlanDiff(retained=retained, added=added, changes=tuple(changes))

    return diff


def _compare_steps(
    old: CompiledWorkflow, new: CompiledWorkflow, *, changes: List[PlanChange]
) -> Tuple[Tuple[StepPath, ...], Tuple[StepPath, ...]]:
    old_steps = {step.path: step for step in old.steps}
    new_steps = {step.path: step for step in new.steps}
    retained: List[StepPath] = []
    for path, old_step in old_steps.items():
        name = format_step_path(path)
        if path not in new_steps:
            pruned = new.demand is not None and path in new.demand.pruned
            reason = NEWLY_PRUNED if pruned else REMOVED
            changes.append(
                PlanChange(
                    "step",
                    name,
                    reason,
                    detail="the new plan no longer runs this step; a preserving "
                    "update cannot remove a step, a reset replaces the processing "
                    "without it",
                    requires=RESET,
                )
            )
            continue

        differences = [
            (reason, detail)
            for reason, same, detail in _STEP_ASPECTS
            if not same(old_step, new_steps[path])
        ]
        for reason, detail in differences:
            changes.append(
                PlanChange(
                    "step",
                    name,
                    reason,
                    detail=detail,
                    requires=RESET,
                )
            )
        if differences:
            continue

        retained.append(path)
        old_wanted, new_wanted = old.wanted_outputs(path), new.wanted_outputs(path)
        if old_wanted != new_wanted:
            changes.append(
                PlanChange(
                    "step",
                    name,
                    WANTED_OUTPUTS_CHANGED,
                    requires=PRESERVE,
                    detail=f"wanted outputs {sorted(old_wanted)} -> {sorted(new_wanted)}",
                )
            )

    added: List[StepPath] = []
    for path in new_steps:
        if path in old_steps:
            continue
        added.append(path)
        demanded = old.demand is not None and path in old.demand.pruned
        changes.append(
            PlanChange(
                "step",
                format_step_path(path),
                NEWLY_DEMANDED if demanded else ADDED,
                requires=PRESERVE,
                detail="constructed fresh, with fresh block-local state",
            )
        )

    return tuple(retained), tuple(added)


def _compare_named(
    component: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    prefix: str,
    changes: List[PlanChange],
    requires: str = RESET,
    detail: str = "",
) -> None:
    for name in [*old, *(name for name in new if name not in old)]:
        if name in old and name in new:
            if _same(old[name], new[name]):
                continue
            reason = CHANGED
        else:
            reason = REMOVED if name in old else ADDED
        changes.append(
            PlanChange(
                component,
                f"{prefix}{name}",
                reason,
                detail=detail,
                requires=requires,
            )
        )


def _compare_controls(
    old: Mapping[str, PlannedControl],
    new: Mapping[str, PlannedControl],
    *,
    changes: List[PlanChange],
) -> None:
    for name in [*old, *(name for name in new if name not in old)]:
        if name in old and name in new:
            if _same_control(old[name], new[name]):
                continue
            reason = CHANGED
        else:
            reason = REMOVED if name in old else ADDED
        changes.append(
            PlanChange(
                "control",
                f"controls.{name}",
                reason,
                requires=RESET,
            )
        )


def _same_control(old: PlannedControl, new: PlannedControl) -> bool:
    """The same declaration; the closure may only gain prunable (pure) readers.

    A control's closure records the readers of the steps it governs, so a
    consumer attached below a controlled step extends it. A prunable reader
    joins as a pure step: it is suspended with the control and has nothing
    to reset, so the retained members keep their state classes, reset
    guards and history. Any other reader must be listed, which changes the
    declaration.
    """
    if not same_control_declaration(old, new):
        return False
    retained = set(old.closure)
    if not retained <= set(new.closure):
        return False
    for path in new.closure:
        if path not in retained:
            if new.state_classes.get(path) != "pure":
                return False
        elif not _same(old.state_classes.get(path), new.state_classes.get(path)):
            return False
        elif not _same(old.downstream.get(path), new.downstream.get(path)):
            return False

    return True


def same_control_declaration(old: PlannedControl, new: PlannedControl) -> bool:
    """Whether two controls are declared alike, whatever the graph around them.

    The closure, its reasons and its state classes follow from the graph and
    are ignored; name, type, members, initial state, state policy, input and
    default must match.

    Args:
        old: The control of the current plan.
        new: The control of the new plan.

    Returns:
        Whether the declarations match.
    """
    same = _same_fields(old, new, ignore=_CONTROL_GRAPH_FIELDS)

    return same


def same_value(a: Any, b: Any) -> bool:
    """Compare two plan values structurally, never trusting ``==`` of objects.

    Args:
        a: A value of the current plan.
        b: A value of the new plan.

    Returns:
        Whether they are equal under the comparison's rules (module docstring).
    """
    same = _same(a, b)

    return same


def _compare_boundaries(
    old: CompiledWorkflow, new: CompiledWorkflow, *, changes: List[PlanChange]
) -> None:
    # A new nested step brings new boundaries; changing an existing one would
    # rebind retained nested steps whose own bindings look unchanged.
    for component, attribute in (
        ("child_input", "child_inputs"),
        ("child_output", "child_outputs"),
    ):
        old_items = _by_boundary(getattr(old, attribute))
        new_items = _by_boundary(getattr(new, attribute))
        for name in [
            *old_items,
            *(name for name in new_items if name not in old_items),
        ]:
            if name not in new_items:
                reason = REMOVED
            elif name not in old_items:
                reason = ADDED
            elif not _same(old_items[name], new_items[name]):
                reason = CHANGED
            else:
                continue
            requires = PRESERVE if reason == ADDED else RESET
            changes.append(
                PlanChange(
                    component,
                    name,
                    reason,
                    requires=requires,
                )
            )


def _by_boundary(items: Sequence[Any]) -> Dict[str, Any]:
    keyed = {f"{'/'.join(item.scope)}.{item.name}": item for item in items}

    return keyed


def _compare_selections(
    old: CompiledWorkflow, new: CompiledWorkflow, *, changes: List[PlanChange]
) -> None:
    _compare_named(
        "output",
        {output.name: output for output in old.outputs},
        {output.name: output for output in new.outputs},
        prefix="",
        changes=changes,
        requires=PRESERVE,
    )
    _compare_named(
        "output_group",
        {group.name: group for group in old.output_groups},
        {group.name: group for group in new.output_groups},
        prefix="",
        changes=changes,
        requires=PRESERVE,
    )


def _compare_sections(
    old: CompiledWorkflow, new: CompiledWorkflow, *, changes: List[PlanChange]
) -> None:
    sections = (
        ("reactions", old.reactions, new.reactions, RESET, ""),
        (
            "recording",
            _recorded(old),
            _recorded(new),
            UNSUPPORTED,
            _RECORDING_DETAIL,
        ),
        (
            "retrospective",
            old.retrospective,
            new.retrospective,
            UNSUPPORTED,
            _RECORDING_DETAIL,
        ),
        (
            "catalogue",
            old.catalogue.providers,
            new.catalogue.providers,
            RESET,
            "",
        ),
        ("target", old.options.target, new.options.target, RESET, ""),
    )
    for component, old_value, new_value, requires, detail in sections:
        if not _same(old_value, new_value):
            changes.append(
                PlanChange(
                    component,
                    component,
                    CHANGED,
                    detail=detail,
                    requires=requires,
                )
            )
    _compare_source_providers(old, new, changes=changes)

    if requests_managed_state(new) and not requests_managed_state(old):
        changes.append(
            PlanChange(
                "managed_state",
                "managed_state",
                ADDED,
                detail="the session was created without managed state",
                requires=RESET,
            )
        )
    # The plan compiled under the user's mutation policy. A new warning (e.g.
    # a new consumer aliasing a mutated value) is a new interference risk for
    # retained instances, so the update rejects it rather than run it live.
    for warning in new.warnings:
        if warning not in old.warnings:
            changes.append(
                PlanChange(
                    "warning",
                    "warning",
                    ADDED,
                    detail=f"new compile warning, rejected for a live graph: {warning}",
                    requires=RESET,
                )
            )


def _compare_source_providers(
    old: CompiledWorkflow, new: CompiledWorkflow, *, changes: List[PlanChange]
) -> None:
    # A source keeps the resources it was resolved with; a changed catalogue
    # provider of one of them would silently not apply to it.
    for name, source in new.sources.items():
        if name not in old.sources:
            continue
        namespace_old = old.catalogue.providers.get(source.namespace, {})
        namespace_new = new.catalogue.providers.get(source.namespace, {})
        changed = sorted(
            spec.name
            for spec in source.spec.resources
            if not _same(namespace_old.get(spec.name), namespace_new.get(spec.name))
        )
        if changed:
            changes.append(
                PlanChange(
                    "source",
                    f"$sources.{name}",
                    PROVIDER_CHANGED,
                    detail=f"catalogue provider(s) {changed} of this source changed; "
                    "a source keeps the resources it was resolved with",
                    requires=UNSUPPORTED,
                )
            )


def _recorded(plan: CompiledWorkflow) -> Any:
    # The definition digest changes with every edit; a later run records the
    # new plan's digest. What is recorded and where must stay.
    if plan.recording is None:
        return None

    recorded = (plan.recording.directory, plan.recording.groups, plan.recording.schema)

    return recorded


def _same_unordered(old: Sequence[Any], new: Sequence[Any]) -> bool:
    remaining = list(new)
    if len(old) != len(remaining):
        return False

    for item in old:
        match = next(
            (i for i, other in enumerate(remaining) if _same(item, other)), None
        )
        if match is None:
            return False
        remaining.pop(match)

    return True


def _quality(step: PlannedStep) -> Optional[Any]:
    quality = (
        None if step.implementation is None else step.implementation.selected_quality
    )

    return quality


# Fields of a control that follow from the graph, not from its declaration.
_CONTROL_GRAPH_FIELDS = frozenset({"closure", "downstream", "state_classes"})

# Fields of a block contract that only document the block.
_BLOCK_DOCUMENTATION = frozenset(
    {"aliases", "description", "metadata", "engine_compatibility"}
)

_PARAMS_DETAIL = (
    "a block may cache state built from its parameters on the first call (e.g. "
    "a tracker from its thresholds); the engine cannot see what an instance "
    "cached, so it never reuses one under new parameters; use a control to "
    "tune a running graph"
)
_INSTANCE_DETAIL = "the retained instance was built and is scheduled for the old one"
_SOURCE_DETAIL = (
    "a source stays open across an update with the contract it was opened "
    "with; start a new session to change a source"
)
_INPUT_DETAIL = (
    "inputs keep their declared schema across an update in this version; "
    "start a new session to change an input"
)
_RECORDING_DETAIL = (
    "what a session records cannot change across an update; start a new "
    "session to change it"
)

_STEP_ASPECTS: Tuple[
    Tuple[str, Callable[[PlannedStep, PlannedStep], bool], str], ...
] = (
    (
        "block_changed",
        lambda a, b: a.spec.block_class is b.spec.block_class
        and a.block_type == b.block_type
        and a.namespace == b.namespace,
        _INSTANCE_DETAIL,
    ),
    (
        # accepts_empty, prunable, events, fields, mutates and the declared
        # implementations change how the engine calls a retained instance.
        "contract_changed",
        lambda a, b: _same_fields(a.spec, b.spec, ignore=_BLOCK_DOCUMENTATION),
        _INSTANCE_DETAIL,
    ),
    (
        # Name, class, resources, phase graph and overlap of the selection.
        "implementation_changed",
        lambda a, b: _same(a.selected, b.selected),
        _INSTANCE_DETAIL,
    ),
    ("quality_changed", lambda a, b: _same(_quality(a), _quality(b)), _INSTANCE_DETAIL),
    ("execution_changed", lambda a, b: a.execution == b.execution, _INSTANCE_DETAIL),
    ("params_changed", lambda a, b: _same(a.params, b.params), _PARAMS_DETAIL),
    (
        "bindings_changed",
        lambda a, b: _same(a.bindings, b.bindings)
        and _same(a.invocation_layout, b.invocation_layout),
        _INSTANCE_DETAIL,
    ),
    ("outputs_changed", lambda a, b: _same(a.outputs, b.outputs), _INSTANCE_DETAIL),
    (
        "gates_changed",
        lambda a, b: _same_unordered(a.gates, b.gates)
        and _same(a.control_targets, b.control_targets),
        _INSTANCE_DETAIL,
    ),
    (
        "dependencies_changed",
        lambda a, b: sorted(a.dependencies) == sorted(b.dependencies),
        _INSTANCE_DETAIL,
    ),
    ("domain_changed", lambda a, b: a.domain == b.domain, _INSTANCE_DETAIL),
)


def _same_fields(a: Any, b: Any, *, ignore: frozenset) -> bool:
    if a is b:
        return True

    same = all(
        _same(getattr(a, item.name), getattr(b, item.name))
        for item in dataclasses.fields(a)
        if item.compare and item.name not in ignore
    )

    return same


_SCALARS = (str, bytes, int, float, complex, bool, type(None), enum.Enum)


def _same(a: Any, b: Any) -> bool:
    """Structural equality that never trusts ``==`` of an arbitrary object.

    Scalars and frozensets use ``==`` (frozenset elements are hashable plan
    values); dataclasses, pydantic params, mappings, tuples and lists compare
    by content; any other object is equal only to itself.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, _SCALARS):
        return a == b
    if dataclasses.is_dataclass(a):
        return all(
            _same(getattr(a, item.name), getattr(b, item.name))
            for item in dataclasses.fields(a)
            if item.compare
        )
    if getattr(type(a), "model_fields", None) is not None:
        # Pydantic models (block params): compare field values structurally.
        return _same(vars(a), vars(b)) and _same(
            getattr(a, "__pydantic_extra__", None),
            getattr(b, "__pydantic_extra__", None),
        )
    if isinstance(a, Mapping):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, frozenset):
        return a == b

    return False
