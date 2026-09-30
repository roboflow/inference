"""Plan the declared operators of an active definition.

An operator is a node of the same dependency graph as the block steps. It
reads values at the end of pulses of its upstream domains and emits pulses of
its own domain, so its inputs depend on their producers and every consumer of
its ports depends on it::

    {"type": "v2/window@v1", "name": "clip", "size": 4,
     "collect": {"frames": "$steps.crop.crops"}, "hold": {"ref": "$sources.cam.image"}}

    OperatorSite(path=("$operators", "clip"), dependencies={("crop",)})
    PlannedOperator(name="clip", inputs=(
        PlannedOperatorInput("frames", "collect", "$steps.crop.crops", ..., domain="cam"),
        PlannedOperatorInput("ref", "hold", "$sources.cam.image", ..., domain="cam")),
        outputs={"frames": [cam axes..., operators.clip:t], "ref": [cam axes...]})

Checked here: every input reaches exactly one pulse domain, never static data,
and ``collect``/``hold`` inputs share one upstream domain. Everything about the
inputs' dimensions belongs to the operator class: ``plan.plan_operator_ports``
asks it for the ports (a window rejects a collected axis that is not
stationary or already a time axis). When the class rejects one input
(``OperatorInputError``), the compiler adds what only it knows: the selector as
written and which producer declared the offending axis, including whether that
axis is a selected collection's K::

    $operators.clip (v2/window@v1): collect 'frames': axis 'pick:picked' ...
      collect.frames ($steps.pick.picked): axis 'pick:picked' comes from
      $steps.pick.picked, a selected collection. ...
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Set, Tuple

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue, OperatorEntry
from roboflow_workflows.execution_engine.v2.compilation.composition import (
    Resolution,
    Scope,
)
from roboflow_workflows.execution_engine.v2.compilation.definition import (
    OperatorDeclaration,
    OperatorInputDeclaration,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    KindMismatchError,
    LineageError,
    OperatorInputError,
    SelectorError,
    StepPath,
    UnknownBlockError,
    WorkflowCompileError,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.operators.contract import INPUT_MAP_ROLES
from roboflow_workflows.execution_engine.v2.plan import (
    Constant,
    InputPort,
    PlannedOperator,
    PlannedOperatorInput,
    PlannedStep,
    SourcePort,
    StepPort,
    derive_domain,
    operator_step_path,
    plan_operator_ports,
)

_MAP_KEYS: Mapping[str, str] = {role: key for key, role in INPUT_MAP_ROLES.items()}
"""Definition key of the map holding each role, e.g. ``collect`` -> ``collect``."""


@dataclass
class OperatorSite:
    """An operator declaration after parameter validation and selector resolution.

    Args:
        declaration: The parsed declaration.
        entry: Catalogue entry of the operator class.
        params: Validated literal parameters.
        inputs: Each declared input with its resolution, in declaration order.
        dependencies: Graph nodes that must be planned first: producing steps
            here; the compiler adds upstream operators and controllers of child
            outputs passed, as for block steps.
    """

    declaration: OperatorDeclaration
    entry: OperatorEntry
    params: Any
    inputs: List[Tuple[OperatorInputDeclaration, Resolution]] = field(
        default_factory=list
    )
    dependencies: Set[StepPath] = field(default_factory=set)

    @property
    def path(self) -> StepPath:
        """Graph node of the operator, ``("$operators", name)``."""
        return operator_step_path(self.declaration.name)

    @property
    def location(self) -> str:
        """``$operators.<name>`` for messages."""
        return format_step_path(self.path)

    def where(self, item: OperatorInputDeclaration) -> str:
        """Readable location of one input, e.g. ``$operators.clip collect.frames``."""
        return f"{self.location} {_MAP_KEYS[item.role]}.{item.name} ({item.selector})"

    def resolutions(self) -> List[Resolution]:
        """Resolutions of every input, for dependency analysis."""
        return [resolution for _, resolution in self.inputs]


def prepare_operator_sites(
    root: Scope, *, catalogue: Catalogue
) -> Dict[StepPath, OperatorSite]:
    """Look up, validate and resolve every declared operator.

    Args:
        root: Root scope holding the ``operators`` declarations.
        catalogue: Catalogue with the operator classes.

    Returns:
        Operator sites by graph node, in declaration order.

    Raises:
        UnknownBlockError: On an unknown operator type.
        ParamsValidationError: When the literal parameters are invalid.
        SelectorError: When an input selector does not resolve, or selects
            every output of a step.
    """
    sites: Dict[StepPath, OperatorSite] = {}
    for declaration in root.workflow.operators:
        path = operator_step_path(declaration.name)
        entry = catalogue.find_operator(declaration.type)
        if entry is None:
            raise UnknownBlockError(
                f"{declaration.location} ($operators.{declaration.name}) uses unknown "
                f"operator type {declaration.type!r}; known operator types: "
                f"{sorted(catalogue.operator_types)}",
                step_path=path,
            )

        params = entry.spec.validate_params(
            declaration.params, operator_name=declaration.name
        )
        site = OperatorSite(declaration=declaration, entry=entry, params=params)
        for item in declaration.inputs:
            resolution = root.resolve_data(
                item.selector,
                location=site.where(item),
                step_path=path,
                field_path=(_MAP_KEYS[item.role], item.name),
            )
            origin = resolution.origin
            if isinstance(origin, StepPort):
                if origin.output == "*":
                    raise SelectorError(
                        f"{site.where(item)} selects every output of a step; an "
                        "operator input names one output",
                        step_path=path,
                        field_path=(_MAP_KEYS[item.role], item.name),
                    )
                site.dependencies.add(origin.step)
            site.inputs.append((item, resolution))
        sites[path] = site

    return sites


def plan_operator(
    site: OperatorSite, *, boundaries: Any, catalogue: Catalogue
) -> PlannedOperator:
    """Plan one operator once every producer behind its inputs is planned.

    Args:
        site: The prepared operator.
        boundaries: The compiler's boundary records; gives kinds and layouts
            of value sources and records child boundaries passed.
        catalogue: Catalogue owning the kinds of the inputs.

    Returns:
        The planned operator with its ports.

    Raises:
        SelectorError: When an input names a missing output.
        LineageError: When an input is static or collect/hold inputs span
            several domains.
        OperatorInputError: When the operator class rejects one input; the
            message adds its selector and the offending axis's producer.
        WorkflowCompileError: When the operator class rejects the inputs as a
            whole.
    """
    inputs = [
        _plan_input(site, item=item, resolution=resolution, boundaries=boundaries)
        for item, resolution in site.inputs
    ]
    _check_one_collected_domain(site, inputs=inputs)

    described = [
        (
            planned.name,
            planned.role,
            planned.layout,
            tuple(catalogue.kind(name) for name in boundaries.kinds_of(origin)),
        )
        for planned, origin in zip(
            inputs, (resolution.origin for resolution in site.resolutions())
        )
    ]
    name = site.declaration.name
    try:
        outputs = plan_operator_ports(
            site.entry.spec, name=name, params=site.params, inputs=described
        )
    except OperatorInputError as error:
        raise _with_input_context(
            error, site=site, inputs=inputs, steps=boundaries.planned
        ) from error
    except WorkflowCompileError:
        raise
    except ContractError as error:
        raise WorkflowCompileError(
            f"{site.location} ({site.entry.spec.type}) cannot plan its ports: {error}",
            step_path=site.path,
        ) from error

    try:
        operator = PlannedOperator(
            name=name,
            spec=site.entry.spec,
            namespace=site.entry.namespace,
            params=site.params,
            inputs=tuple(inputs),
            outputs=outputs,
        )
    except ContractError as error:
        raise WorkflowCompileError(str(error), step_path=site.path) from error

    return operator


@dataclass(frozen=True)
class AxisProducer:
    """What declared an axis, read from the planned producers.

    Args:
        text: Readable producer, e.g. ``$steps.crop.crops`` or ``$sources.cam``.
        selected_collection: Whether the axis is the K of a selected collection
            (an ``expand`` output with ``context_policy="selected"``).
    """

    text: str
    selected_collection: bool = False


def axis_producer(
    axis_id: str, *, steps: Mapping[StepPath, PlannedStep]
) -> AxisProducer:
    """Find what introduced an axis, for compile errors.

    Args:
        axis_id: Axis identity.
        steps: Steps planned so far.

    Returns:
        The producing step output or cast, source, operator or workflow input.
    """
    for step in steps.values():
        for output in step.outputs.values():
            if output.transform == "expand" and output.layout.axis_ids[-1] == axis_id:
                producer = AxisProducer(
                    text=f"{format_step_path(step.path)}.{output.name}",
                    selected_collection=output.context_policy == "selected",
                )
                return producer
        for binding in step.bindings:
            if binding.cast_layout and binding.cast_layout.axis_ids[-1] == axis_id:
                return AxisProducer(
                    f"{format_step_path(step.path)} cast of {binding.field}"
                )

    scope, _, _ = axis_id.partition(":")
    if scope.startswith(("sources.", "operators.")):
        return AxisProducer(f"${scope}")

    return AxisProducer(f"workflow input axis {axis_id!r}")


def _with_input_context(
    error: OperatorInputError,
    *,
    site: OperatorSite,
    inputs: List[PlannedOperatorInput],
    steps: Mapping[StepPath, PlannedStep],
) -> OperatorInputError:
    """The class's error, followed by the selector and the axis's producer."""
    item = next(planned for planned in inputs if planned.name == error.input)
    context = f"{_MAP_KEYS[item.role]}.{item.name} ({item.selector})"
    if error.axis_id is not None:
        producer = axis_producer(error.axis_id, steps=steps)
        context += f": axis {error.axis_id!r} comes from {producer.text}"
        if producer.selected_collection:
            context += (
                ", a selected collection. Its K axis stays an axis even with one "
                "member and is never stationary; return one Selected sample to "
                "reduce the group before collecting again"
            )
    enriched = OperatorInputError(
        f"{error}\n  {context}",
        step_path=error.step_path,
        role=error.role,
        input=error.input,
        axis_id=error.axis_id,
    )

    return enriched


def _plan_input(
    site: OperatorSite,
    *,
    item: OperatorInputDeclaration,
    resolution: Resolution,
    boundaries: Any,
) -> PlannedOperatorInput:
    field_path = (_MAP_KEYS[item.role], item.name)
    origin = resolution.origin
    if isinstance(origin, (Constant, InputPort)):
        raise LineageError(
            f"{site.where(item)} reads a static value; an operator consumes pulse "
            "data of a source, an operator or a step fed by them",
            step_path=site.path,
            field_path=field_path,
        )
    if boundaries.kinds_of(origin) is None:
        raise SelectorError(
            f"{site.where(item)}: {_producer(origin)} has no output "
            f"{origin.output!r}; its outputs are {boundaries.output_names(origin)}",
            step_path=site.path,
            field_path=field_path,
        )

    layout = boundaries.layout_of(resolution, location=site.where(item))
    try:
        domain = derive_domain(
            [resolution.port],
            controllers=(),
            steps=boundaries.planned,
            boundaries=boundaries.records,
            operator_upstreams=boundaries.operator_upstreams,
        )
    except ContractError as error:
        raise LineageError(
            f"{site.where(item)} {error}", step_path=site.path, field_path=field_path
        ) from error
    if domain is None:
        raise LineageError(
            f"{site.where(item)} reads static data (no source or operator reaches "
            "it); an operator consumes pulse data",
            step_path=site.path,
            field_path=field_path,
        )

    planned = PlannedOperatorInput(
        name=item.name,
        role=item.role,
        selector=item.selector,
        source=resolution.port,
        layout=layout,
        domain=domain,
    )

    return planned


def _check_one_collected_domain(
    site: OperatorSite, *, inputs: List[PlannedOperatorInput]
) -> None:
    """``collect`` and ``hold`` inputs read one upstream domain; name them if not."""
    collecting = [item for item in inputs if item.role in ("collect", "hold")]
    if len({item.domain for item in collecting}) <= 1:
        return

    described = ", ".join(
        f"{_MAP_KEYS[item.role]}.{item.name} ({item.selector}) from {item.domain!r}"
        for item in collecting
    )
    raise LineageError(
        f"{site.location} collects from several pulse domains: {described}. "
        "Collected and held values come from one domain; relate independent "
        "sources with an alignment operator first",
        step_path=site.path,
    )


def _producer(origin: Any) -> str:
    if isinstance(origin, SourcePort):
        return f"${origin.origin}s.{origin.source}"

    return format_step_path(origin.step)


def check_operator_kinds(
    operators: Mapping[str, PlannedOperator], *, catalogue: Catalogue
) -> None:
    """Check that every planned port kind is known to the plan's catalogue.

    Args:
        operators: Planned operators.
        catalogue: Catalogue the plan keeps.

    Raises:
        KindMismatchError: When a port names a kind the catalogue lacks.
    """
    for operator in operators.values():
        for output in operator.outputs.values():
            unknown = [name for name in output.kinds if name not in catalogue.kinds]
            if unknown:
                raise KindMismatchError(
                    f"$operators.{operator.name} output {output.name!r} has kinds "
                    f"{unknown} unknown to the catalogue; an operator emitting new "
                    "kinds declares them in its class-owned spec so the catalogue "
                    "registers them",
                    step_path=operator.step_path,
                )
