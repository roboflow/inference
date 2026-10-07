"""Demand: which steps and outputs a compiled plan keeps for the requested outputs.

``apply_demand`` runs on a finished ``CompiledWorkflow`` and returns a plan that
contains only the steps the requested outputs need, with a ``DemandPlan``
record of what was kept, what was dropped and why::

    roots     producers (and child-output gate controllers) of every requested
              workflow output or group field, of every field of a recorded
              group, and of every operator input; plus every step whose block
              is not ``prunable``, so effects, state and unknown behaviour stay
    closure   the roots and everything they depend on, transitively: data
              producers, gating controllers, controllers of gated child outputs
    pruned    prunable steps outside the closure; they leave the plan entirely,
              so a pipelined run has no stage for them and nothing waits
    wanted    per retained step, the outputs some retained reader binds
              (through child boundaries), a requested or recorded field selects
              or an operator reads; ``self.wants(name)`` answers from it

Requested names are flat output names of a passive definition, or output group
names and ``<group>.<field>`` entries of an active one. ``None`` requests every
declared output. Unrequested outputs and groups are absent from the plan and
from results: a missing key, never a ``None``. A recorded group is demanded
whole whatever the request says; requesting one of its fields alone is an
error, so nothing recorded is ever silently left out. Nested workflows need no
special case: their steps are plan steps and their boundaries resolve through
``plan.origin``, so two uses of one saved child are demanded independently.

Live controls (a later milestone) narrow the same computation per control
snapshot: ``compute_demand`` takes the root sources and the always-retained
steps explicitly, so a runtime caller can pass a narrower request without
touching the compiler.
"""

from collections import deque
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import (
    Any,
    Deque,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DemandError,
    StepPath,
    WorkflowCompileError,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    ChildOutputPort,
    CompiledWorkflow,
    PlannedOutputGroup,
    PlannedStep,
    PlannedWorkflowOutput,
    Source,
    StepPort,
    derive_dependencies,
)

__all__ = [
    "Demand",
    "DemandPlan",
    "OutputDemand",
    "PrunedStep",
    "apply_demand",
    "compute_demand",
    "requested_fields",
]

NOT_PRUNABLE = "block is not prunable"
"""Retention reason of a step whose block keeps every call."""

UNDEMANDED = "no retained step, output or operator reads it, and its block is prunable"
"""Pruning reason of a dropped step."""


@dataclass(frozen=True)
class OutputDemand:
    """Why a declared workflow output or group is in, or out of, the plan.

    Args:
        name: Output name (passive), group name or ``<group>.<field>`` (active).
        status: ``requested``, ``recorded`` (an unrequested recorded group) or
            ``omitted`` (not requested; absent from results).
        fields: For a group, the fields the plan delivers; empty otherwise.
    """

    name: str
    status: str
    fields: Tuple[str, ...] = ()

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description: Dict[str, Any] = {"status": self.status}
        if self.fields:
            description["fields"] = list(self.fields)

        return description


@dataclass(frozen=True)
class PrunedStep:
    """A step the plan dropped.

    Args:
        path: Step path.
        block_type: Canonical block type.
        reason: Why it was dropped.
    """

    path: StepPath
    block_type: str
    reason: str = UNDEMANDED

    def describe(self) -> Dict[str, str]:
        """Return a JSON-friendly description."""
        return {"type": self.block_type, "reason": self.reason}


@dataclass(frozen=True)
class Demand:
    """Result of one demand computation over a plan's steps.

    Args:
        retained: Retention reason per retained step path.
        wanted: Demanded output names per retained step path.
    """

    retained: Mapping[StepPath, str]
    wanted: Mapping[StepPath, FrozenSet[str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "retained", MappingProxyType(dict(self.retained)))
        object.__setattr__(
            self,
            "wanted",
            MappingProxyType(
                {path: frozenset(names) for path, names in self.wanted.items()}
            ),
        )


@dataclass(frozen=True)
class DemandPlan:
    """What a compiled plan computes for its requested outputs, and why.

    Args:
        requested: The caller's request; ``None`` when every output was
            requested.
        outputs: Status of every declared output or group, in declaration
            order, with a separate entry per requested field of a group.
        retained: Retention reason per step kept, in plan order.
        pruned: Steps dropped from the plan, in original plan order.
        wanted: Demanded output names per retained step.
    """

    requested: Optional[Tuple[str, ...]]
    outputs: Tuple[OutputDemand, ...]
    retained: Mapping[StepPath, str]
    pruned: Mapping[StepPath, PrunedStep]
    wanted: Mapping[StepPath, FrozenSet[str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "retained", MappingProxyType(dict(self.retained)))
        object.__setattr__(self, "pruned", MappingProxyType(dict(self.pruned)))
        object.__setattr__(
            self,
            "wanted",
            MappingProxyType(
                {path: frozenset(names) for path, names in self.wanted.items()}
            ),
        )

    def wanted_outputs(self, path: StepPath) -> FrozenSet[str]:
        """Return the demanded outputs of one retained step.

        Args:
            path: Step path.

        Returns:
            The output names some retained reader needs.

        Raises:
            ContractError: When the step is not retained by this plan.
        """
        path = tuple(path)
        if path not in self.wanted:
            raise ContractError(
                f"{format_step_path(path)} is not a retained step of the plan; "
                f"retained steps: {[format_step_path(item) for item in self.wanted]}"
            )

        return self.wanted[path]

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "requested": list(self.requested) if self.requested is not None else None,
            "outputs": {item.name: item.describe() for item in self.outputs},
            "retained": {
                format_step_path(path): reason for path, reason in self.retained.items()
            },
            "pruned": {
                format_step_path(path): item.describe()
                for path, item in self.pruned.items()
            },
            "wanted": {
                format_step_path(path): sorted(names)
                for path, names in self.wanted.items()
            },
        }

        return description


def apply_demand(
    plan: CompiledWorkflow,
    *,
    requested: Optional[Sequence[str]],
    recorded_groups: Sequence[str] = (),
) -> CompiledWorkflow:
    """Narrow ``plan`` to the requested outputs and record the demand.

    Args:
        plan: A compiled plan without a demand record.
        requested: Requested output names (``CompileOptions.requested_outputs``);
            ``None`` requests every declared output.
        recorded_groups: Output groups the root ``recording`` declaration
            records; demanded whole regardless of ``requested``.

    Returns:
        A plan with the undemanded steps, outputs and groups removed and
        ``demand`` set. The given plan is unchanged.

    Raises:
        DemandError: When a requested name matches no declared output, group
            or field, or requests one field of a recorded group.
        WorkflowCompileError: When the narrowed plan is inconsistent.
    """
    if plan.demand is not None:
        raise ContractError("apply_demand expects a plan without a demand record")

    outputs, groups, statuses = requested_fields(
        plan, requested=requested, recorded_groups=recorded_groups
    )
    root_sources: List[Tuple[Source, str]] = [
        (output.source, f"requested output {output.name!r}") for output in outputs
    ]
    for group_name, fields in groups.items():
        status = next(item.status for item in statuses if item.name == group_name)
        for output in fields:
            root_sources.append(
                (output.source, f"{status} group {group_name!r} field {output.name!r}")
            )
    for operator in plan.operators.values():
        for item in operator.inputs:
            root_sources.append(
                (item.source, f"$operators.{operator.name} input {item.name!r}")
            )
    demand = compute_demand(
        plan,
        root_sources=root_sources,
        always=[
            (step.path, NOT_PRUNABLE) for step in plan.steps if not step.spec.prunable
        ],
    )

    narrowed = _narrow(
        plan, demand=demand, outputs=outputs, groups=groups, statuses=statuses
    )
    record = DemandPlan(
        requested=tuple(requested) if requested is not None else None,
        outputs=statuses,
        retained=demand.retained,
        pruned={
            step.path: PrunedStep(path=step.path, block_type=step.block_type)
            for step in plan.steps
            if step.path not in demand.retained
        },
        wanted=demand.wanted,
    )
    try:
        applied = replace(narrowed, demand=record)
    except ContractError as error:
        raise WorkflowCompileError(
            f"Plan narrowed to the requested outputs is inconsistent: {error}"
        ) from error

    return applied


def compute_demand(
    plan: CompiledWorkflow,
    *,
    root_sources: Iterable[Tuple[Source, str]],
    always: Iterable[Tuple[StepPath, str]],
) -> Demand:
    """Compute the retained steps and their wanted outputs for given roots.

    This is the seam a runtime control snapshot narrows: pass fewer root
    sources (or fewer always-retained steps) and read the result.

    Args:
        plan: The plan whose steps are considered.
        root_sources: Value sources that must be terminal, each with the
            reason it is demanded; their producers and the controllers of
            gated child outputs on the way become roots.
        always: Steps retained regardless of readers, each with its reason.

    Returns:
        The retained steps with reasons and the wanted outputs per step.
    """
    steps = {step.path: step for step in plan.steps}
    boundaries = _boundaries(plan)
    retained: Dict[StepPath, str] = {}
    worklist: Deque[Tuple[StepPath, str]] = deque()
    for source, reason in root_sources:
        for path in derive_dependencies([source], steps=steps, boundaries=boundaries):
            worklist.append((path, reason))
    worklist.extend((tuple(path), reason) for path, reason in always)

    while worklist:
        path, reason = worklist.popleft()
        if path in retained:
            continue
        retained[path] = reason
        for dependency in steps[path].dependencies:
            if dependency not in retained:
                worklist.append((dependency, f"dependency of {format_step_path(path)}"))

    wanted: Dict[StepPath, Set[str]] = {
        path: set() for path in steps if path in retained
    }
    readers: List[Source] = [source for source, _ in root_sources]
    readers.extend(
        binding.source for path in wanted for binding in steps[path].bindings
    )
    for source in readers:
        origin = plan.origin(source)
        if not isinstance(origin, StepPort) or origin.step not in wanted:
            continue
        if origin.output == "*":
            wanted[origin.step].update(steps[origin.step].outputs)
        else:
            wanted[origin.step].add(origin.output)

    demand = Demand(
        retained={path: retained[path] for path in steps if path in retained},
        wanted={path: frozenset(names) for path, names in wanted.items()},
    )

    return demand


def requested_fields(
    plan: CompiledWorkflow,
    *,
    requested: Optional[Sequence[str]],
    recorded_groups: Sequence[str] = (),
) -> Tuple[
    Tuple[PlannedWorkflowOutput, ...],
    Mapping[str, Tuple[PlannedWorkflowOutput, ...]],
    Tuple[OutputDemand, ...],
]:
    """Resolve requested names to the plan's flat outputs and group fields.

    Args:
        plan: The compiled plan.
        requested: Requested names; ``None`` for everything.
        recorded_groups: Groups demanded whole by the recording declaration.

    Returns:
        The kept flat outputs, the kept fields per kept group (declaration
        order) and the status of every declared output and group.

    Raises:
        DemandError: On an unknown name, a field of a passive output, or a
            single field of a recorded group.
    """
    recorded = tuple(recorded_groups)
    declared_outputs = {output.name: output for output in plan.outputs}
    declared_groups = {group.name: group for group in plan.output_groups}
    handler_groups = {group.name for group in plan.reactions.groups}

    if requested is None:
        statuses = [
            OutputDemand(name=name, status="requested") for name in declared_outputs
        ]
        statuses.extend(
            OutputDemand(
                name=name,
                status="requested",
                fields=tuple(output.name for output in group.outputs),
            )
            for name, group in declared_groups.items()
        )
        groups = {name: tuple(group.outputs) for name, group in declared_groups.items()}
        return tuple(plan.outputs), groups, tuple(statuses)

    wanted_outputs: Set[str] = set()
    wanted_fields: Dict[str, Set[str]] = {}
    whole_groups: Set[str] = set()
    for name in requested:
        if name in declared_outputs:
            wanted_outputs.add(name)
            continue
        if name in declared_groups:
            whole_groups.add(name)
            continue
        if name in handler_groups:
            # Handler groups are delivered by the reaction runtime regardless.
            continue
        group_name, separator, field_name = name.partition(".")
        group = declared_groups.get(group_name) if separator else None
        if group is not None and any(
            output.name == field_name for output in group.outputs
        ):
            wanted_fields.setdefault(group_name, set()).add(field_name)
            continue
        raise DemandError(_unknown_request(name, plan=plan))

    for group_name in wanted_fields:
        if group_name in recorded and group_name not in whole_groups:
            raise DemandError(
                f"requested_outputs names field(s) "
                f"{sorted(wanted_fields[group_name])} of output group "
                f"{group_name!r}, which the recording declaration records whole; "
                f"request {group_name!r} itself or leave it out, so nothing "
                "recorded is silently left out"
            )

    outputs = tuple(output for output in plan.outputs if output.name in wanted_outputs)
    groups: Dict[str, Tuple[PlannedWorkflowOutput, ...]] = {}
    statuses: List[OutputDemand] = [
        OutputDemand(
            name=name, status="requested" if name in wanted_outputs else "omitted"
        )
        for name in declared_outputs
    ]
    for name, group in declared_groups.items():
        if name in whole_groups:
            fields = tuple(group.outputs)
            status = "requested"
        elif name in recorded:
            fields = tuple(group.outputs)
            status = "recorded"
        elif name in wanted_fields:
            fields = tuple(
                output for output in group.outputs if output.name in wanted_fields[name]
            )
            status = "requested"
        else:
            statuses.append(OutputDemand(name=name, status="omitted"))
            continue
        groups[name] = fields
        statuses.append(
            OutputDemand(
                name=name, status=status, fields=tuple(item.name for item in fields)
            )
        )
        if status == "requested" and name in wanted_fields and name not in whole_groups:
            statuses.extend(
                OutputDemand(
                    name=f"{name}.{output.name}",
                    status=(
                        "requested" if output.name in wanted_fields[name] else "omitted"
                    ),
                )
                for output in group.outputs
            )

    return outputs, groups, tuple(statuses)


def _unknown_request(name: str, *, plan: CompiledWorkflow) -> str:
    if plan.is_active:
        declared = [
            f"{group.name}.{output.name}"
            for group in plan.output_groups
            for output in group.outputs
        ]
        known = sorted({group.name for group in plan.output_groups}) + declared
        hint = "output group names or <group>.<field> entries"
    else:
        known = list(output.name for output in plan.outputs)
        hint = "workflow output names"

    return (
        f"requested_outputs names {name!r}, which the definition does not declare; "
        f"it accepts {hint}: {known}"
    )


def _boundaries(plan: CompiledWorkflow) -> Dict[Any, Any]:
    boundaries: Dict[Any, Any] = {}
    for item in plan.child_inputs + plan.child_outputs:
        boundaries[item.port] = item

    return boundaries


def _narrow(
    plan: CompiledWorkflow,
    *,
    demand: Demand,
    outputs: Tuple[PlannedWorkflowOutput, ...],
    groups: Mapping[str, Tuple[PlannedWorkflowOutput, ...]],
    statuses: Tuple[OutputDemand, ...],
) -> CompiledWorkflow:
    """Build the plan that contains only the retained steps, outputs and groups."""
    retained = demand.retained
    steps = tuple(
        _narrow_targets(step, retained=retained)
        for step in plan.steps
        if step.path in retained
    )
    by_path = {step.path: step for step in steps}
    boundaries = _boundaries(plan)
    kept_groups = tuple(
        PlannedOutputGroup(
            name=group.name,
            anchor=group.anchor,
            outputs=groups[group.name],
            dependencies=derive_dependencies(
                [output.source for output in groups[group.name]],
                steps=by_path,
                boundaries=boundaries,
            ),
        )
        for group in plan.output_groups
        if group.name in groups
    )

    used_ports: Set[Any] = set()
    sources: List[Source] = [output.source for output in outputs]
    sources.extend(output.source for group in kept_groups for output in group.outputs)
    sources.extend(
        item.source for operator in plan.operators.values() for item in operator.inputs
    )
    sources.extend(binding.source for step in steps for binding in step.bindings)
    for source in sources:
        while isinstance(source, (ChildInputPort, ChildOutputPort)):
            used_ports.add(source)
            source = boundaries[source].source
    child_inputs = tuple(item for item in plan.child_inputs if item.port in used_ports)
    child_outputs = tuple(
        item for item in plan.child_outputs if item.port in used_ports
    )

    narrowed = replace(
        plan,
        steps=steps,
        outputs=outputs,
        output_groups=kept_groups,
        child_inputs=child_inputs,
        child_outputs=child_outputs,
    )

    return narrowed


def _narrow_targets(
    step: PlannedStep, *, retained: Mapping[StepPath, str]
) -> PlannedStep:
    """Keep a controller's target keys, but only the retained steps behind them."""
    if not step.control_targets:
        return step

    targets = {
        key: tuple(path for path in paths if path in retained)
        for key, paths in step.control_targets.items()
    }
    if all(targets[key] == tuple(paths) for key, paths in step.control_targets.items()):
        return step

    narrowed = replace(step, control_targets=targets)

    return narrowed
