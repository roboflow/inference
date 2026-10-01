"""Compiled-plan description and connection discovery.

Both read only the ``CompiledWorkflow``: nested workflows are already flattened
into scoped step paths (``("child", "scale")`` is ``$steps.child/scale``), each
used nested input is one child input port (``$steps.child: $inputs.x``), each
child output that forwards a value past the child's steps is one gated child
output port (``$steps.child.out``, decision 026), and every selector leaf
already has a resolved source. Nothing is constructed and no kind decoder
runs; constants are shown as written.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout
from roboflow_workflows.execution_engine.v2.errors import StepPath, format_step_path
from roboflow_workflows.execution_engine.v2.introspection._sources import (
    domain_node,
    node_of,
    operator_node,
    output_of,
    selector_of,
    source_node,
)
from roboflow_workflows.execution_engine.v2.operators.contract import INPUT_MAP_ROLES
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    ChildInputPort,
    ChildOutputPort,
    CompiledWorkflow,
    Constant,
    InputPort,
    PlannedChildInput,
    PlannedChildOutput,
    PlannedInput,
    PlannedOperator,
    PlannedOutputGroup,
    PlannedSource,
    PlannedStep,
    SourcePort,
    StepPort,
)

_MAP_KEYS: Mapping[str, str] = {role: key for key, role in INPUT_MAP_ROLES.items()}
"""Definition map holding each operator input role, e.g. ``input`` -> ``inputs``."""


@dataclass(frozen=True)
class Connection:
    """One edge of a compiled plan.

    Args:
        kind: ``data`` (a value bound into a step field), ``child_input`` (a
            value entering a nested workflow input), ``child_output`` (a value
            leaving a nested workflow past its steps), ``control`` (a gate
            governing a step or a child output), ``operator_input`` (a value
            an operator consumes), ``output`` (a flat workflow output),
            ``anchor`` (the source or operator port whose pulses deliver an
            output group) or ``group_output`` (a field of an output group).
        source: ``$inputs.<name>``, a step node id, a source or operator node
            id (``$sources.<name>``, ``$operators.<name>``), a child input
            port (``$steps.child: $inputs.x``) or a child output port
            (``$steps.child.out``).
        target: Consuming step node id, child port, operator node id,
            ``$outputs.<name>``, ``$output_groups.<group>`` or
            ``$output_groups.<group>.<field>``.
        selector: Selector text as written at the target; for control, the
            target key as the controller received it.
        output: Producing step output (``"*"`` for all) or source port for
            step and source sources.
        field_path: Consuming field and position for data edges; the selector
            map and input name for operator inputs, e.g. ``("collect", "frames")``.
        mode: Binding mode for data edges (see ``plan``).
        produced_kinds: Kinds the source declares (for a child input port,
            the kinds the child workflow declared).
        accepted_kinds: Kinds the consuming field or child input accepts.
    """

    kind: str
    source: str
    target: str
    selector: str
    output: Optional[str] = None
    field_path: Tuple[Any, ...] = ()
    mode: Optional[str] = None
    produced_kinds: Tuple[str, ...] = ()
    accepted_kinds: Tuple[str, ...] = ()

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "kind": self.kind,
            "source": self.source,
            "output": self.output,
            "target": self.target,
            "field_path": list(self.field_path),
            "selector": self.selector,
            "mode": self.mode,
            "produced_kinds": list(self.produced_kinds),
            "accepted_kinds": list(self.accepted_kinds),
        }

        return description


def describe_workflow(plan: CompiledWorkflow) -> Dict[str, Any]:
    """Describe a compiled plan for tools and contributors.

    Adds to ``CompiledWorkflow.describe()`` what an editor needs: input
    defaults, nested workflow inputs, each step's scope and configured
    parameters (literal values as written, and every selector with its
    immediate source and its origin through nested inputs, including
    compile-time constants), and every axis with its kind and origin. An
    active plan adds its declared sources (configured parameters, ports with
    scoped axes, the steps one pulse runs and the groups it delivers), its
    operators (parameters, inputs with their domains, planned ports, route,
    groups and downstream operators), its output groups (anchor, fields,
    readiness dependencies) and each step's causal ``domain``. Every step
    names the implementation selected for the plan's ``target``, why each
    alternative was or was not chosen, its phase graph and the ``execution``
    it will use (``run`` also when phases were requested but the selected
    implementation has none).

    Args:
        plan: Compiled plan.

    Returns:
        JSON-friendly mapping with ``target``, ``block_execution``,
        ``inputs``, ``sources``, ``operators``,
        ``child_inputs``
        (outer before inner), ``child_outputs``, ``steps`` (execution order),
        ``outputs``, ``output_groups``, ``axes`` and ``warnings``. Defaults
        and constants use ``{"value": v}``; ``None`` means none. ``sources``
        ``operators`` and ``output_groups`` are empty for a passive plan.
    """
    description = {
        "target": plan.options.target.describe(),
        "block_execution": plan.options.block_execution,
        "inputs": {name: _describe_input(item) for name, item in plan.inputs.items()},
        "sources": {
            name: _describe_declared_source(item, plan=plan)
            for name, item in plan.sources.items()
        },
        "operators": {
            name: _describe_operator(item, plan=plan)
            for name, item in plan.operators.items()
        },
        "child_inputs": [
            _describe_child_input(item, plan=plan) for item in plan.child_inputs
        ],
        "child_outputs": [
            _describe_child_output(item, plan=plan) for item in plan.child_outputs
        ],
        "steps": [_describe_step(step, plan=plan) for step in plan.steps],
        "outputs": {output.name: output.describe() for output in plan.outputs},
        "output_groups": {
            group.name: _describe_group(group, plan=plan)
            for group in plan.output_groups
        },
        "axes": {
            axis.id: {
                "kind": axis.kind,
                "stationary": axis.stationary,
                "origin": plan.axis_origin(axis.id).describe(),
            }
            for axis in _collect_axes(plan)
        },
        "warnings": list(plan.warnings),
    }

    return description


def discover_connections(plan: CompiledWorkflow) -> Tuple[Connection, ...]:
    """List the child input, data, control and output edges of a compiled plan.

    Constants (literal bindings and defaults) have no source node, so they
    create no edge; ``describe_workflow`` lists them.

    Args:
        plan: Compiled plan.

    Returns:
        Child input edges (outer before inner), child output edges with the
        control edges of their gates, then step edges in execution order of
        their targets, then operator input edges in operator order, then flat
        workflow outputs, then for each output group its ``anchor`` edge
        followed by one ``group_output`` edge per field.
    """
    steps_by_path = {step.path: step for step in plan.steps}

    def produced(source: Any) -> Tuple[str, ...]:
        return _produced_kinds(source, plan=plan, steps_by_path=steps_by_path)

    connections = [
        Connection(
            kind="child_input",
            source=node_of(item.source),
            target=item.port.describe(),
            selector=selector_of(item.source),
            output=output_of(item.source),
            produced_kinds=produced(item.source),
            accepted_kinds=item.kinds,
        )
        for item in plan.child_inputs
        if not isinstance(item.source, Constant)
    ]
    for item in plan.child_outputs:
        target = item.port.describe()
        connections.append(
            Connection(
                kind="child_output",
                source=node_of(item.source),
                target=target,
                selector=selector_of(item.source),
                output=output_of(item.source),
                produced_kinds=produced(item.source),
            )
        )
        connections.extend(
            Connection(
                kind="control",
                source=format_step_path(gate.controller),
                target=target,
                selector=gate.target,
            )
            for gate in item.gates
        )
    for step in plan.steps:
        target = format_step_path(step.path)
        for binding in step.bindings:
            if isinstance(binding.source, Constant):
                continue
            marker = step.spec.fields[binding.field].marker_at(binding.position)
            connections.append(
                Connection(
                    kind="data",
                    source=node_of(binding.source),
                    target=target,
                    selector=binding.selector,
                    output=output_of(binding.source),
                    field_path=binding.field_path,
                    mode=binding.mode,
                    produced_kinds=produced(binding.source),
                    accepted_kinds=marker.kind_names,
                )
            )
        connections.extend(
            Connection(
                kind="control",
                source=format_step_path(gate.controller),
                target=target,
                selector=gate.target,
            )
            for gate in step.gates
        )

    connections.extend(
        Connection(
            kind="operator_input",
            source=node_of(item.source),
            target=operator_node(operator.name),
            selector=item.selector,
            output=output_of(item.source),
            field_path=(_MAP_KEYS[item.role], item.name),
            produced_kinds=produced(item.source),
        )
        for operator in plan.operators.values()
        for item in operator.inputs
    )
    connections.extend(
        Connection(
            kind="output",
            source=node_of(output.source),
            target=f"$outputs.{output.name}",
            selector=output.selector,
            output=output_of(output.source),
            produced_kinds=produced(output.source),
        )
        for output in plan.outputs
    )
    for group in plan.output_groups:
        target = f"$output_groups.{group.name}"
        connections.append(
            Connection(
                kind="anchor",
                source=node_of(group.anchor),
                target=target,
                selector=group.anchor.describe(),
                output=group.anchor.output,
                produced_kinds=produced(group.anchor),
            )
        )
        connections.extend(
            Connection(
                kind="group_output",
                source=node_of(output.source),
                target=f"{target}.{output.name}",
                selector=output.selector,
                output=output_of(output.source),
                produced_kinds=produced(output.source),
            )
            for output in group.outputs
            if not isinstance(output.source, Constant)
        )

    return tuple(connections)


def _describe_declared_source(
    item: PlannedSource, *, plan: CompiledWorkflow
) -> Dict[str, Any]:
    description = {
        "node_id": source_node(item.name),
        "type": item.spec.type,
        "namespace": item.namespace,
        "parameters": _describe_parameters(
            item.spec,
            item.params,
            bindings=item.bindings,
            control_targets={},
            plan=plan,
        ),
        "outputs": {
            name: {
                "kinds": list(output.kinds),
                "axes": list(output.layout.axis_ids),
                "selector": item.port(name).describe(),
            }
            for name, output in item.outputs.items()
        },
        "resources": [resource.describe() for resource in item.spec.resources],
        "route": [format_step_path(step.path) for step in plan.route(item.name)],
        "groups": [group.name for group in plan.groups_of(item.name)],
    }

    return description


def _describe_operator(
    item: PlannedOperator, *, plan: CompiledWorkflow
) -> Dict[str, Any]:
    description = {
        "node_id": operator_node(item.name),
        "type": item.spec.type,
        "namespace": item.namespace,
        "parameters": item.params.model_dump(mode="json"),
        "inputs": {
            planned.name: {
                "role": planned.role,
                "selector": planned.selector,
                "domain": planned.domain,
                "axes": list(planned.layout.axis_ids),
                **_describe_source(planned.source, plan=plan),
            }
            for planned in item.inputs
        },
        "upstream_domains": list(item.upstream_domains),
        "outputs": {
            name: {
                "kinds": list(output.kinds),
                "axes": list(output.layout.axis_ids),
                "selector": item.port(name).describe(),
            }
            for name, output in item.outputs.items()
        },
        "route": [format_step_path(step.path) for step in plan.route(item.name)],
        "groups": [group.name for group in plan.groups_of(item.name)],
        "consumers": [
            operator_node(consumer.name) for consumer in plan.consumers_of(item.name)
        ],
    }

    return description


def _describe_group(
    group: PlannedOutputGroup, *, plan: CompiledWorkflow
) -> Dict[str, Any]:
    description = {
        "anchor": group.anchor.describe(),
        "source": domain_node(group.anchor),
        "outputs": {
            output.name: {
                **output.describe(),
                "axes": list(_layout_of(output.source, plan=plan).axis_ids),
                **_describe_source(output.source, plan=plan),
            }
            for output in group.outputs
        },
        "dependencies": [format_step_path(path) for path in group.dependencies],
    }

    return description


def _layout_of(source: Any, *, plan: CompiledWorkflow):
    """Layout of a selected port; a wildcard has none, reported as empty."""
    if isinstance(source, InputPort):
        return plan.inputs[source.name].layout
    if isinstance(source, SourcePort):
        return plan.source_port(source).layout
    if isinstance(source, ChildInputPort):
        return plan.child_input(source).layout
    if isinstance(source, ChildOutputPort):
        return plan.child_output(source).layout
    if isinstance(source, StepPort) and source.output != "*":
        return plan.step(source.step).outputs[source.output].layout

    return EntryLayout()


def _describe_input(item: PlannedInput) -> Dict[str, Any]:
    description = {
        "kinds": list(item.kinds),
        "axes": list(item.layout.axis_ids),
        "required": item.required,
        "default": None if item.required else {"value": item.default},
        "declared_type": item.declared_type,
    }

    return description


def _describe_child_input(
    item: PlannedChildInput, *, plan: CompiledWorkflow
) -> Dict[str, Any]:
    description = {
        "port": item.port.describe(),
        "scope": list(item.scope),
        "name": item.name,
        "kinds": list(item.kinds),
        "axes": list(item.layout.axis_ids),
        **_describe_source(item.source, plan=plan),
    }

    return description


def _describe_child_output(
    item: PlannedChildOutput, *, plan: CompiledWorkflow
) -> Dict[str, Any]:
    description = {
        "port": item.port.describe(),
        "scope": list(item.scope),
        "name": item.name,
        "axes": list(item.layout.axis_ids),
        "gates": [gate.describe() for gate in item.gates],
        **_describe_source(item.source, plan=plan),
    }

    return description


def _describe_step(step: PlannedStep, *, plan: CompiledWorkflow) -> Dict[str, Any]:
    description = {
        "node_id": format_step_path(step.path),
        "path": list(step.path),
        "scope": list(step.path[:-1]),
        "type": step.block_type,
        "namespace": step.namespace,
        "invocation_axes": list(step.invocation_layout.axis_ids),
        "delivers_batches": step.delivers_batches,
        "accepts_empty": step.spec.accepts_empty,
        "parameters": _describe_parameters(
            step.spec,
            step.params,
            bindings=step.bindings,
            control_targets=step.control_targets,
            plan=plan,
        ),
        "outputs": {name: output.describe() for name, output in step.outputs.items()},
        "gates": [gate.describe() for gate in step.gates],
        "control_targets": {
            key: [format_step_path(path) for path in paths]
            for key, paths in step.control_targets.items()
        },
        "dependencies": [format_step_path(path) for path in step.dependencies],
        "domain": _domain_node(step.domain, plan=plan),
        "implementation": _describe_implementation(step),
        "execution": step.execution,
    }

    return description


def _describe_implementation(step: PlannedStep) -> Dict[str, Any]:
    """The selected implementation and, when compiled, every alternative's reason."""
    choice = step.implementation
    description = {
        **step.selected.describe(),
        "considered": choice.describe()["considered"] if choice is not None else None,
    }

    return description


def _domain_node(domain: Optional[str], *, plan: CompiledWorkflow) -> Optional[str]:
    """Node id of a step's causal domain; ``None`` for a static step."""
    if domain is None:
        return None
    if domain in plan.operators:
        return operator_node(domain)

    return source_node(domain)


def _describe_parameters(
    spec: Any,
    params: Any,
    *,
    bindings: Tuple[Binding, ...],
    control_targets: Mapping[str, Tuple[StepPath, ...]],
    plan: CompiledWorkflow,
) -> Dict[str, Any]:
    """Each field's configured value and the selectors inside it.

    Works for a step (``BlockSpec``) and a declared source (``SourceSpec``).
    ``value`` is the validated parameter with selectors still as text.
    ``explicit`` says whether the definition set it (``False``: the default).
    """
    uses_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for binding in bindings:
        uses_by_field.setdefault(binding.field, []).append(
            _describe_binding(binding, plan=plan)
        )
    for use in spec.find_selectors(params):
        if use.marker.role != "step":
            continue
        uses_by_field.setdefault(use.field, []).append(
            {
                "position": list(use.position),
                "selector": use.selector,
                "governs": [
                    format_step_path(path)
                    for path in control_targets.get(use.selector, ())
                ],
            }
        )

    parameters = {
        name: {
            "value": getattr(params, name),
            "explicit": name in params.model_fields_set,
            "selectors": uses_by_field.get(name, []),
        }
        for name in spec.fields
    }

    return parameters


def _describe_binding(binding: Binding, *, plan: CompiledWorkflow) -> Dict[str, Any]:
    description = {
        "position": list(binding.position),
        "selector": binding.selector,
        **_describe_source(binding.source, plan=plan),
        "mode": binding.mode,
        "batch": binding.batch,
        "source_axes": list(binding.source_layout.axis_ids),
        "cast_axes": (
            list(binding.cast_layout.axis_ids) if binding.cast_layout else None
        ),
    }

    return description


def _describe_source(source: Any, *, plan: CompiledWorkflow) -> Dict[str, Any]:
    """Immediate source, origin through child ports, and any constant.

    ``constant`` is the literal as written (no decoder runs during
    inspection), whether it is bound directly or reached through ports.
    """
    origin = plan.origin(source)
    description = {
        "source": None if isinstance(source, Constant) else source.describe(),
        "origin": None if isinstance(origin, Constant) else origin.describe(),
        "constant": {"value": origin.value} if isinstance(origin, Constant) else None,
    }

    return description


def _collect_axes(plan: CompiledWorkflow) -> List[Axis]:
    layouts = [item.layout for item in plan.inputs.values()]
    layouts.extend(
        output.layout
        for item in plan.sources.values()
        for output in item.outputs.values()
    )
    layouts.extend(
        layout
        for item in plan.operators.values()
        for layout in [planned.layout for planned in item.inputs]
        + [output.layout for output in item.outputs.values()]
    )
    layouts.extend(item.layout for item in plan.child_inputs)
    layouts.extend(item.layout for item in plan.child_outputs)
    for step in plan.steps:
        layouts.append(step.invocation_layout)
        layouts.extend(output.layout for output in step.outputs.values())
        layouts.extend(binding.source_layout for binding in step.bindings)
        layouts.extend(
            binding.cast_layout for binding in step.bindings if binding.cast_layout
        )

    axes: Dict[str, Axis] = {}
    for layout in layouts:
        for axis in layout.axes:
            axes.setdefault(axis.id, axis)

    collected = list(axes.values())

    return collected


def _produced_kinds(
    source: Any,
    *,
    plan: CompiledWorkflow,
    steps_by_path: Mapping[StepPath, PlannedStep],
) -> Tuple[str, ...]:
    if isinstance(source, InputPort):
        return tuple(plan.inputs[source.name].kinds)
    if isinstance(source, SourcePort):
        return tuple(plan.source_port(source).kinds)
    if isinstance(source, ChildInputPort):
        return plan.child_input(source).kinds
    if isinstance(source, ChildOutputPort):
        forwarded = _produced_kinds(
            plan.child_output(source).source, plan=plan, steps_by_path=steps_by_path
        )
        return forwarded
    if not isinstance(source, StepPort):
        return ()

    producer = steps_by_path[source.step]
    if source.output != "*":
        return tuple(producer.outputs[source.output].kinds)

    all_kinds = tuple(
        dict.fromkeys(
            kind for output in producer.outputs.values() for kind in output.kinds
        )
    )

    return all_kinds
