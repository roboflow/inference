"""Compile the root ``controls`` section against a compiled plan.

Two passes, around the demand pass::

    members = control_members(plan, declarations)    # selectors -> step paths
    plan = apply_demand(plan, ..., retained=members)  # members survive narrowing
    plan = compile_controls(plan, declarations, members=members, ...)

``compile_controls`` computes every ``enable`` control's closure (members plus
the ``prunable`` steps that transitively read them), classifies how each
closure step holds state, checks the state policy and the effect consent,
validates ``input`` controls (built-in scalar defaults only) and refuses
anything that would change a recording's static schema or settings. Every refusal is a
``ControlDefinitionError`` that names the step and the fix.
"""

from collections import deque
from dataclasses import replace
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from roboflow_workflows.execution_engine.v2.compilation.definition import (
    ControlDeclaration,
)
from roboflow_workflows.execution_engine.v2.controls import (
    ControlPlan,
    PlannedControl,
    non_scalar_problem,
    producers_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ControlDefinitionError,
    StepPath,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.execution.inputs import (
    decode_payload,
    kinds_named,
)
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    InputPort,
    PlannedStep,
)
from roboflow_workflows.execution_engine.v2.resources import MANAGED_STATE_RESOURCE

__all__ = ["compile_controls", "control_members"]

_Reader = Tuple[str, Any]
"""``("step", path)`` or ``("operator", name)``: who reads a step's outputs or decisions."""


def control_members(
    plan: CompiledWorkflow, declarations: Sequence[ControlDeclaration]
) -> Dict[str, Tuple[StepPath, ...]]:
    """Resolve every ``enable`` control's step selectors to plan steps.

    A selector naming a nested workflow step lists every block step inside
    it. Runs on the plan before demand narrowing, so the members can be
    retained through it.

    Args:
        plan: The compiled, not yet narrowed plan.
        declarations: The parsed ``controls`` entries.

    Returns:
        Control name to its member step paths, in plan order.

    Raises:
        ControlDefinitionError: For a selector that names no block step, an
            operator, a handler-plan step, or a step two controls list.
    """
    members: Dict[str, Tuple[StepPath, ...]] = {}
    listed_by: Dict[StepPath, str] = {}
    order = {step.path: position for position, step in enumerate(plan.steps)}
    for declaration in declarations:
        if declaration.type != "enable":
            continue
        where = f"controls[{declaration.name!r}]"
        paths: List[StepPath] = []
        for selector in declaration.steps:
            resolved = _resolve_step_selector(plan, selector, where=where)
            for path in resolved:
                if path in paths:
                    continue
                owner = listed_by.get(path)
                if owner is not None and owner != declaration.name:
                    raise ControlDefinitionError(
                        f"{where} lists {format_step_path(path)}, which control "
                        f"{owner!r} already lists; a step belongs to one enable "
                        "control"
                    )
                listed_by[path] = declaration.name
                paths.append(path)
        members[declaration.name] = tuple(sorted(paths, key=order.__getitem__))

    return members


def _resolve_step_selector(
    plan: CompiledWorkflow, selector: str, *, where: str
) -> List[StepPath]:
    path = tuple(selector[len("$steps.") :].split("/"))
    exact = [step.path for step in plan.steps if step.path == path]
    if exact:
        return exact
    nested = [
        step.path
        for step in plan.steps
        if len(step.path) > len(path) and step.path[: len(path)] == path
    ]
    if nested:
        return nested

    name = path[0]
    if name in plan.operators:
        raise ControlDefinitionError(
            f"{where}.steps names {selector!r}, an operator; operators are not "
            "controllable: they keep collecting while their consumers are disabled. "
            "Control the steps that read the operator instead"
        )
    if name in plan.sources:
        raise ControlDefinitionError(
            f"{where}.steps names {selector!r}, a source; sources are not "
            "controllable, stop the run or control the steps reading the source"
        )
    for handler in plan.reactions.handlers:
        if any(step.path == path for step in handler.plan.steps):
            raise ControlDefinitionError(
                f"{where}.steps names {selector!r}, a step of handler "
                f"{format_step_path(handler.path)}; handler plans run on events "
                "and are not controllable"
            )
    known = [format_step_path(step.path) for step in plan.steps]
    raise ControlDefinitionError(
        f"{where}.steps names {selector!r}, which is not a block step of the "
        f"workflow; block steps: {known}"
    )


def compile_controls(
    plan: CompiledWorkflow,
    declarations: Sequence[ControlDeclaration],
    *,
    members: Mapping[str, Tuple[StepPath, ...]],
    recorded_groups: Sequence[str] = (),
    recording: Optional[Mapping[str, Any]] = None,
) -> CompiledWorkflow:
    """Attach the compiled ``ControlPlan`` to a narrowed plan.

    Args:
        plan: The compiled plan after demand narrowing.
        declarations: The parsed ``controls`` entries.
        members: ``control_members`` of the un-narrowed plan.
        recorded_groups: Output groups the root ``recording`` records.
        recording: The raw root ``recording`` declaration, if any.

    Returns:
        The plan with ``controls`` set; ``plan`` itself without declarations.

    Raises:
        ControlDefinitionError: See the module docstring.
    """
    if not declarations:
        return plan

    readers = _readers(plan)
    compiled: Dict[str, PlannedControl] = {}
    for declaration in declarations:
        if declaration.type == "enable":
            compiled[declaration.name] = _compile_enable(
                plan, declaration, members=members[declaration.name], readers=readers
            )
        else:
            compiled[declaration.name] = _compile_input(
                plan, declaration, compiled=compiled, recording=recording
            )
    _check_recorded_groups(plan, compiled, recorded_groups=recorded_groups)
    controlled = replace(plan, controls=ControlPlan(compiled))

    return controlled


def _readers(plan: CompiledWorkflow) -> Dict[StepPath, List[_Reader]]:
    """Who reads each step: steps bound to its outputs or gated by it, operators."""
    readers: Dict[StepPath, List[_Reader]] = {}
    for step in plan.steps:
        for binding in step.bindings:
            for producer in producers_of(plan, binding.source):
                readers.setdefault(producer, []).append(("step", step.path))
        for gate in step.gates:
            readers.setdefault(gate.controller, []).append(("step", step.path))
    for operator in plan.operators.values():
        for item in operator.inputs:
            for producer in producers_of(plan, item.source):
                readers.setdefault(producer, []).append(("operator", operator.name))

    return readers


def _compile_enable(
    plan: CompiledWorkflow,
    declaration: ControlDeclaration,
    *,
    members: Tuple[StepPath, ...],
    readers: Mapping[StepPath, List[_Reader]],
) -> PlannedControl:
    where = f"controls[{declaration.name!r}]"
    order = {step.path: position for position, step in enumerate(plan.steps)}
    closure: List[StepPath] = [path for path in members if path in order]
    downstream: Dict[StepPath, str] = {}
    queue: Deque[StepPath] = deque(closure)
    while queue:
        producer = queue.popleft()
        for kind, who in readers.get(producer, ()):
            if kind == "operator":
                raise ControlDefinitionError(
                    f"{where}: $operators.{who} reads {format_step_path(producer)}; "
                    "operators are not controllable and keep collecting, so a "
                    "disabled producer would starve the operator. Feed it from a "
                    "step outside the control"
                )
            if who in closure:
                continue
            reader = plan.step(who)
            if not reader.spec.prunable:
                raise ControlDefinitionError(
                    f"{where}: step {format_step_path(who)} ({reader.block_type}) "
                    f"reads {format_step_path(producer)} but is not prunable and not "
                    "listed in the control. List it under steps (it is then "
                    "disabled with the others) or reroute it; the engine never "
                    "skips a block that may hold state or effects on its own"
                )
            closure.append(who)
            downstream[who] = f"prunable reader of {format_step_path(producer)}"
            queue.append(who)
    closure.sort(key=order.__getitem__)
    state_classes = {path: _state_class(plan.step(path)) for path in closure}
    if declaration.state == "reset_on_enable":
        _check_reset_policy(plan, declaration, closure=closure, classes=state_classes)
    control = PlannedControl(
        name=declaration.name,
        type="enable",
        members=members,
        closure=tuple(closure),
        downstream=downstream,
        enabled=declaration.enabled,
        state=declaration.state,
        suspends_effects=declaration.suspends_effects,
        state_classes=state_classes,
    )

    return control


def _state_class(step: PlannedStep) -> str:
    if step.spec.prunable:
        return "pure"
    selected = step.selected
    if any(resource.name == MANAGED_STATE_RESOURCE for resource in selected.resources):
        return "managed"
    if callable(getattr(selected.implementation_class, "reset_state", None)):
        return "local"

    return "opaque"


def _has_effects(step: PlannedStep) -> bool:
    spec = step.spec
    effects = (
        bool(spec.events)
        or bool(spec.mutates)
        or (not step.outputs and not spec.is_control)
    )

    return effects


def _check_reset_policy(
    plan: CompiledWorkflow,
    declaration: ControlDeclaration,
    *,
    closure: Sequence[StepPath],
    classes: Mapping[StepPath, str],
) -> None:
    where = f"controls[{declaration.name!r}]"
    for path in closure:
        kind = classes[path]
        step = plan.step(path)
        if kind == "managed":
            raise ControlDefinitionError(
                f"{where}: state 'reset_on_enable' cannot reset "
                f"{format_step_path(path)} ({step.block_type}): it requests "
                "managed_state, which other steps, handlers and sessions share; "
                "a control never resets managed state. Use state 'keep_ticking' "
                "or keep that state local to the block"
            )
        if kind == "opaque":
            owner = step.selected.implementation_class.__qualname__
            raise ControlDefinitionError(
                f"{where}: state 'reset_on_enable' needs every stopped member to "
                f"declare reset_state(); {format_step_path(path)} "
                f"({step.block_type}, {owner}) does not and is not prunable. Add "
                "reset_state() to the class, mark the block prunable, or use "
                "state 'keep_ticking'"
            )
    if declaration.suspends_effects:
        return
    # Skipping a block needs the workflow's consent when the block may have
    # effects: declared ones, or undeclared ones of any non-prunable block
    # (reset_state() proves a reset capability, not the absence of effects).
    effectful = [path for path in closure if _has_effects(plan.step(path))]
    if effectful:
        raise ControlDefinitionError(
            f"{where}: disabling stops the events or in-place mutations of "
            f"{[format_step_path(path) for path in effectful]}; set "
            "suspends_effects: true to confirm that those effects may pause, or "
            "use state 'keep_ticking'"
        )
    stateful = [path for path in closure if classes[path] != "pure"]
    if stateful:
        raise ControlDefinitionError(
            f"{where}: state 'reset_on_enable' skips "
            f"{[format_step_path(path) for path in stateful]} while disabled; a "
            "block that is not prunable may have effects its declaration does "
            "not show, and reset_state() does not prove otherwise. Set "
            "suspends_effects: true to confirm that its effects may pause, or "
            "use state 'keep_ticking'"
        )


def _compile_input(
    plan: CompiledWorkflow,
    declaration: ControlDeclaration,
    *,
    compiled: Mapping[str, PlannedControl],
    recording: Optional[Mapping[str, Any]],
) -> PlannedControl:
    where = f"controls[{declaration.name!r}]"
    name = declaration.input
    planned = plan.inputs.get(name)
    if planned is None:
        raise ControlDefinitionError(
            f"{where}.input names {name!r}, which is not a root input; inputs: "
            f"{sorted(plan.inputs)}"
        )
    if planned.layout.depth:
        raise ControlDefinitionError(
            f"{where}.input {name!r} is a grouped input over axes "
            f"{list(planned.layout.axis_ids)}; only ungrouped (scalar) inputs are "
            "controllable"
        )
    for other_name, other in compiled.items():
        if other.type == "input" and other.input == name:
            raise ControlDefinitionError(
                f"{where} controls input {name!r}, which control {other_name!r} "
                "already controls"
            )
    for source in plan.sources.values():
        if any(
            isinstance(binding.source, InputPort) and binding.source.name == name
            for binding in source.bindings
        ):
            raise ControlDefinitionError(
                f"{where}: source {source.name!r} reads $inputs.{name} as a "
                "parameter; sources read static inputs when the run starts, so "
                "the input is not controllable"
            )
    if recording is not None and recording.get("directory") == f"$inputs.{name}":
        raise ControlDefinitionError(
            f"{where}: recording.directory reads $inputs.{name}; recording "
            "settings are static for the whole recording, so the input is not "
            "controllable"
        )
    if declaration.has_default:
        value = declaration.default
    elif planned.required:
        raise ControlDefinitionError(
            f"{where}: input {name!r} is required and the control declares no "
            "default; add a default so the control has a value before the first "
            "update"
        )
    else:
        value = planned.default
    rejected = (
        f"{where}.default {value!r} is not a control value of input {name!r} "
        f"(kinds {list(planned.kinds)})"
    )
    problem = non_scalar_problem(value)
    if problem is not None:
        raise ControlDefinitionError(f"{rejected}: {problem}")

    try:
        decoded = decode_payload(
            value, kinds=kinds_named(plan, planned.kinds), location=f"{where}.default"
        )
    except Exception as error:
        raise ControlDefinitionError(f"{rejected}: {error}") from error
    problem = non_scalar_problem(decoded)
    if problem is not None:
        raise ControlDefinitionError(f"{rejected}: after kind decoding, {problem}")
    control = PlannedControl(
        name=declaration.name, type="input", input=name, default=decoded
    )

    return control


def _check_recorded_groups(
    plan: CompiledWorkflow,
    compiled: Mapping[str, PlannedControl],
    *,
    recorded_groups: Sequence[str],
) -> None:
    """A recording's schema is static: no control may omit a recorded field."""
    closures = {
        path: name
        for name, control in compiled.items()
        if control.type == "enable"
        for path in control.closure
    }
    if not closures or not recorded_groups:
        return

    groups = {group.name: group for group in plan.output_groups}
    for group_name in recorded_groups:
        group = groups.get(group_name)
        if group is None:
            continue
        for output in group.outputs:
            touched = sorted(
                closures[path]
                for path in producers_of(plan, output.source)
                if path in closures
            )
            if touched:
                raise ControlDefinitionError(
                    f"controls{touched} would omit field {output.name!r} of "
                    f"recorded group {group_name!r}; a recording's schema is static "
                    "and has no 'omitted' status. Keep the field's producers "
                    "outside the control, or do not record the group"
                )
