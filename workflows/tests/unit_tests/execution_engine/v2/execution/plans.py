"""Hand-built plans for executor tests.

Each test states a step's invocation layout ``P`` explicitly; binding modes
follow from the source layout ``S`` exactly as the plan table defines them:

    S == P            element         S == ()   constant (item) / constant_group (group)
    S prefix of P     ancestor        S == P + axis (group field)   group

Expand axes are ``"<step>/<axis>"`` and cast axes ``"<step>/<field>/cast"``.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout
from roboflow_workflows.execution_engine.v2.declaration import parse_selector, spec_of
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    CompiledWorkflow,
    Constant,
    Gate,
    InputPort,
    PlannedInput,
    PlannedOutput,
    PlannedStep,
    PlannedWorkflowOutput,
    StepPort,
)

SCALAR = EntryLayout()
N = Axis(id="N", kind="sample")
BATCH = EntryLayout(axes=(N,))


def nested(*axis_ids: str) -> EntryLayout:
    """Return ``[N, *axis_ids]`` with dynamic nesting axes."""
    layout = EntryLayout(
        axes=(N,)
        + tuple(Axis(id=axis_id, kind="dynamic_nesting") for axis_id in axis_ids)
    )

    return layout


class PlanBuilder:
    """Collects inputs, steps and outputs of one hand-built plan."""

    def __init__(self, *, kinds: Sequence[Any] = ()):
        self._inputs: Dict[str, PlannedInput] = {}
        self._steps: List[PlannedStep] = []
        self._outputs: List[PlannedWorkflowOutput] = []
        self._layouts: Dict[str, EntryLayout] = {}
        self._kinds = tuple(kinds)

    def input(
        self,
        name: str,
        layout: EntryLayout = SCALAR,
        *,
        kinds: Tuple[str, ...] = ("*",),
        required: bool = True,
        default: Any = None,
    ) -> "PlanBuilder":
        self._inputs[name] = PlannedInput(
            name=name,
            kinds=kinds,
            layout=layout,
            required=required,
            default=default,
        )
        self._layouts[f"$inputs.{name}"] = layout
        return self

    def step(
        self,
        block: type,
        name: str,
        *,
        at: EntryLayout = SCALAR,
        gates: Sequence[Tuple[str, str]] = (),
        targets: Optional[Mapping[str, Sequence[str]]] = None,
        **raw_params: Any,
    ) -> "PlanBuilder":
        spec = spec_of(block)
        params = spec.validate_params(raw_params)
        bindings = []
        control_targets: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        for use in spec.find_selectors(params):
            if use.marker.role == "step":
                governed = (targets or {}).get(
                    use.selector, [parse_selector(use.selector).name]
                )
                control_targets[use.selector] = tuple((path,) for path in governed)
                continue
            bindings.append(
                self._binding(
                    use.field,
                    use.position,
                    use.selector,
                    role=use.marker.role,
                    batch=use.marker.batch,
                    at=at,
                    step_name=name,
                )
            )
        bindings.extend(
            self._literal_casts(spec, params, bindings, at=at, step_name=name)
        )

        outputs = {}
        for output_name, output in spec.resolve_outputs(params).items():
            if output.transform == "expand":
                kind = "static_nesting" if output.stationary else "dynamic_nesting"
                layout = at.append_axis(Axis(id=f"{name}/{output.expand}", kind=kind))
            elif output.transform == "preserve":
                group = next(b for b in bindings if b.field == output.preserve)
                layout = group.group_layout
            else:
                layout = at
            outputs[output_name] = PlannedOutput(
                name=output_name,
                kinds=output.kind_names,
                layout=layout,
                transform=output.transform,
                group_field=output.preserve,
                source_field=output.source,
                context_policy=output.context_policy,
            )
            self._layouts[f"$steps.{name}.{output_name}"] = layout

        planned_gates = []
        for controller, target in gates:
            controller_step = next(s for s in self._steps if s.path == (controller,))
            planned_gates.append(
                Gate(
                    controller=(controller,),
                    target=target,
                    controller_layout=controller_step.invocation_layout,
                )
            )

        self._steps.append(
            PlannedStep(
                path=(name,),
                spec=spec,
                namespace="",
                params=params,
                bindings=tuple(bindings),
                invocation_layout=at,
                outputs=outputs,
                gates=tuple(planned_gates),
                control_targets=control_targets,
            )
        )
        return self

    def output(
        self, name: str, selector: str, *, options: Optional[Mapping[str, Any]] = None
    ) -> "PlanBuilder":
        parsed = parse_selector(selector)
        source = (
            InputPort(parsed.name)
            if parsed.target == "input"
            else StepPort((parsed.name,), parsed.output)
        )
        self._outputs.append(
            PlannedWorkflowOutput(
                name=name, selector=selector, source=source, options=dict(options or {})
            )
        )
        return self

    def build(self) -> CompiledWorkflow:
        blocks = {step.spec.block_class for step in self._steps}
        plan = CompiledWorkflow(
            inputs=self._inputs,
            steps=tuple(self._steps),
            outputs=tuple(self._outputs),
            catalogue=Catalogue(
                sorted(blocks, key=lambda cls: cls.type), kinds=self._kinds
            ),
        )
        return plan

    def _literal_casts(self, spec, params, bindings, *, at, step_name) -> List[Binding]:
        """Mirror the compiler: a literal at a Group/batch="always" position is
        bound as a Constant with an empty selector and delivered cast."""
        bound = {(binding.field, binding.position) for binding in bindings}
        casts = []
        for name, field_spec in spec.fields.items():
            value = getattr(params, name)
            leaves = []
            if _casts(field_spec.whole) and not any(
                field == name for field, _ in bound
            ):
                leaves = [((), value, field_spec.whole)]
            elif _casts(field_spec.leaves) and isinstance(value, (list, dict)):
                items = enumerate(value) if isinstance(value, list) else value.items()
                leaves = [
                    ((position,), leaf, field_spec.leaves)
                    for position, leaf in items
                    if (name, (position,)) not in bound
                ]
            for position, leaf, marker in leaves:
                if leaf is None:
                    continue
                group = marker.role == "group"
                casts.append(
                    Binding(
                        field=name,
                        position=position,
                        selector="",
                        source=Constant(leaf),
                        source_layout=SCALAR,
                        mode="constant_group" if group else "constant",
                        batch=marker.batch,
                        cast_layout=(
                            at.append_axis(
                                Axis(
                                    id=f"{step_name}/{name}/cast",
                                    kind="dynamic_nesting",
                                )
                            )
                            if group
                            else None
                        ),
                    )
                )

        return casts

    def _binding(
        self,
        field: str,
        position: Tuple[Any, ...],
        selector: str,
        *,
        role: str,
        batch: str,
        at: EntryLayout,
        step_name: str,
    ) -> Binding:
        parsed = parse_selector(selector)
        source = (
            InputPort(parsed.name)
            if parsed.target == "input"
            else StepPort((parsed.name,), parsed.output)
        )
        source_layout = self._layouts[selector]
        source_ids, step_ids = source_layout.axis_ids, at.axis_ids
        cast_layout = None
        if role == "group" and not source_ids:
            mode = "constant_group"
            cast_layout = at.append_axis(
                Axis(id=f"{step_name}/{field}/cast", kind="dynamic_nesting")
            )
        elif role == "group":
            mode = "group"
        elif not source_ids:
            mode = "constant"
        elif source_ids == step_ids:
            mode = "element"
        else:
            mode = "ancestor"

        binding = Binding(
            field=field,
            position=position,
            selector=selector,
            source=source,
            source_layout=source_layout,
            mode=mode,
            batch=batch,
            cast_layout=cast_layout,
        )
        return binding


def _casts(marker) -> bool:
    return marker is not None and (marker.role == "group" or marker.batch == "always")
