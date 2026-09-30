"""Plan the declared sources of an active definition.

A source declaration is validated like a step, but it is not a step: it has
no invocation, and its parameters are static configuration::

    {"type": "demo/csv_temperature@v1", "name": "temp", "path": "$inputs.path"}

    PlannedSource(name="temp", spec=<CsvTemperature>, params=..., bindings=(
        Binding(field="path", source=InputPort("path"), mode="constant"), ...),
        outputs={"temperature": PlannedSourceOutput(kinds=("float",), layout=())})

A parameter selector must name an ungrouped workflow input; anything read
from a step or another source is rejected. Port layouts are the class's local
layouts scoped by ``plan.scoped_layout`` (``sources.<name>:<local id>``), so
two sources never share an axis by accident while one source's ports keep
their declared correspondence. The active definition rules (no grouped inputs,
no flat outputs beside sources, no groups without sources) are checked here as
well. Nothing here constructs a source.
"""

from typing import Dict, Mapping

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation.composition import Scope
from roboflow_workflows.execution_engine.v2.compilation.definition import (
    SourceDeclaration,
)
from roboflow_workflows.execution_engine.v2.data import EntryLayout
from roboflow_workflows.execution_engine.v2.declaration import parse_selector
from roboflow_workflows.execution_engine.v2.errors import (
    KindMismatchError,
    SelectorError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.kinds import kinds_compatible
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    InputPort,
    PlannedInput,
    PlannedSource,
    PlannedSourceOutput,
    scoped_layout,
)
from roboflow_workflows.execution_engine.v2.sources import source_step_path


def check_active_definition(root: Scope) -> None:
    """Reject definitions mixing sources with passive-only declarations.

    Args:
        root: Root scope of the composition.

    Raises:
        WorkflowCompileError: When sources come with grouped inputs or flat
            outputs, or output groups or operators come without sources.
    """
    workflow = root.workflow
    if not workflow.sources and workflow.operators:
        raise WorkflowCompileError(
            f"{workflow.location}operators "
            f"{[operator.name for operator in workflow.operators]} need sources: "
            "session.run() executes each call on its own and keeps no pulses to "
            "align or collect across calls. Declare sources, or pass values a host "
            "already collected as an input with a time axis"
        )
    if not workflow.sources:
        if workflow.output_groups:
            raise WorkflowCompileError(
                f"{workflow.location}outputs declares output groups "
                f"{[group.name for group in workflow.output_groups]}, but the "
                "definition declares no sources; use flat JsonField outputs"
            )
        return

    if workflow.outputs:
        raise WorkflowCompileError(
            f"{workflow.location}outputs declares flat outputs "
            f"{[output.name for output in workflow.outputs]} beside sources; an "
            "active definition delivers its outputs in OutputGroup entries"
        )
    grouped = [
        name
        for name, declaration in workflow.inputs.items()
        if declaration.layout.depth
    ]
    if grouped:
        raise WorkflowCompileError(
            f"{workflow.location}inputs {grouped} are grouped, but a definition "
            "with sources accepts only ungrouped inputs as static configuration; "
            "dynamic data comes from the sources"
        )


def plan_sources(
    root: Scope,
    *,
    catalogue: Catalogue,
    inputs: Mapping[str, PlannedInput],
) -> Dict[str, PlannedSource]:
    """Validate every source declaration and plan its bindings and ports.

    Args:
        root: Root scope holding the ``sources`` declarations.
        catalogue: Catalogue with the source classes.
        inputs: Planned workflow inputs the parameters may select.

    Returns:
        Planned sources by name, in declaration order.

    Raises:
        WorkflowCompileError: On an unknown source type.
        ParamsValidationError: When the parameters are invalid.
        SelectorError: When a parameter selects anything but an ungrouped
            workflow input.
        KindMismatchError: When the selected input's kinds do not fit.
    """
    planned: Dict[str, PlannedSource] = {}
    for declaration in root.workflow.sources:
        entry = catalogue.find_source(declaration.type)
        if entry is None:
            raise WorkflowCompileError(
                f"$sources.{declaration.name} uses unknown source type "
                f"{declaration.type!r}; known source types: "
                f"{sorted(catalogue.source_types)}",
                step_path=source_step_path(declaration.name),
            )

        spec = entry.spec
        params = spec.validate_params(declaration.params, source_name=declaration.name)
        bindings = tuple(
            _static_binding(declaration, use=use, inputs=inputs)
            for use in spec.find_selectors(params)
        )
        outputs = {
            name: PlannedSourceOutput(
                name=name,
                kinds=output.kind_names,
                layout=scoped_layout(declaration.name, output.layout),
            )
            for name, output in spec.outputs.items()
        }
        planned[declaration.name] = PlannedSource(
            name=declaration.name,
            spec=spec,
            namespace=entry.namespace,
            params=params,
            bindings=bindings,
            outputs=outputs,
        )

    return planned


def _static_binding(
    declaration: SourceDeclaration, *, use, inputs: Mapping[str, PlannedInput]
) -> Binding:
    where = f"{declaration.location} ($sources.{declaration.name}) parameter"
    field = ".".join(str(part) for part in use.field_path)
    step_path = source_step_path(declaration.name)
    parsed = parse_selector(use.selector)
    if parsed.target != "input":
        raise SelectorError(
            f"{where} {field} selects {use.selector!r}; source parameters are static "
            "configuration and accept literals or $inputs.<name> of an ungrouped "
            "input only",
            step_path=step_path,
            field_path=use.field_path,
        )
    planned_input = inputs.get(parsed.name)
    if planned_input is None:
        raise SelectorError(
            f"{where} {field} references unknown workflow input {parsed.name!r}; "
            f"inputs here: {list(inputs)}",
            step_path=step_path,
            field_path=use.field_path,
        )
    if planned_input.layout.depth:
        raise SelectorError(
            f"{where} {field} selects {use.selector!r}, which is grouped over "
            f"{list(planned_input.layout.axis_ids)}; a source parameter needs an "
            "ungrouped input",
            step_path=step_path,
            field_path=use.field_path,
        )
    if not kinds_compatible(planned_input.kinds, use.marker.kind_names):
        raise KindMismatchError(
            f"{where} {field} accepts kinds {list(use.marker.kind_names)}, but "
            f"{use.selector!r} provides {list(planned_input.kinds)}",
            step_path=step_path,
            field_path=use.field_path,
        )

    binding = Binding(
        field=use.field,
        position=use.position,
        selector=use.selector,
        source=InputPort(name=parsed.name),
        source_layout=EntryLayout(),
        mode="constant",
        batch=use.marker.batch,
    )

    return binding
