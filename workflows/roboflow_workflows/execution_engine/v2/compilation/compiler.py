"""Turn a composed scope tree into a validated ``CompiledWorkflow``.

Steps are planned in dependency order, so every producer's output layouts are
known before its consumers are planned::

    for each block step (root first, children where they are declared):
        params = spec.validate_params(step)      # literals, defaults, selectors
        uses   = spec.find_selectors(params)     # whole fields, list/dict leaves
        each data use -> scope.resolve_data      # origin + child inputs passed
        each StepRef  -> scope.resolve_targets   # nested target = all its steps
    order = topological order of data and control edges          # CycleError
    for step in order:
        P        = deepest varying layout (items, groups minus last axis)
                   else deepest gate layout, else ()
        bindings = element | ancestor | constant | group | constant_group
        outputs  = P | P + own axis | the preserved group's layout
    child_inputs = every child input a binding or output passes, outer first

A binding that passes a nested workflow input reads that input's
``ChildInputPort``; the port's ``PlannedChildInput`` records the child's kinds
and what it reads, so the executor checks (and decodes a constant) once per
run. Layouts, kinds and dependencies come from the origin behind the ports.
Literal casts at ``Group`` and ``batch="always"`` positions are direct
``Constant`` bindings, not child inputs.

``P`` follows axis identity, never sizes: every candidate must be a prefix of
the deepest one. Control supplies ``P`` only to a step without varying data,
and a gate must decide over a prefix of ``P``. Gates are kept per step and
combine by conjunction at run time. Descendants of a gated step are not gated
transitively, so an empty-accepting join can still recover a surviving branch.

Mutation analysis: a step declaring ``mutates`` for a field conflicts with
another step that may see the same payload without an ordering dependency
between the two. Payloads are shared through the same source, the same child
input over a constant, or an output declaring ``source=`` a field bound to it.
Other child inputs pass their source's payload on. The policy
comes from ``CompileOptions.mutation_conflicts``.

Active definitions: declared sources are planned first (``compilation.sources``)
and their ports are ordinary value sources with the port's scoped layout. Each
step then records its causal ``domain``: the one source reached through its
bindings, gates and nested boundaries (``plan.derive_domain``), or ``None``
for a static step. Reaching two sources is a ``LineageError``; so is an
output group whose field comes from a source other than its anchor's.

Nothing here constructs a block, a source, or executes submitted code.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue, CatalogueEntry
from roboflow_workflows.execution_engine.v2.compilation.composition import (
    Composition,
    Hop,
    Resolution,
    Scope,
)
from roboflow_workflows.execution_engine.v2.compilation.definition import (
    OutputGroupDeclaration,
    WorkflowInputDeclaration,
    WorkflowOutputDeclaration,
)
from roboflow_workflows.execution_engine.v2.compilation.sources import (
    check_active_definition,
    plan_sources,
)
from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_STATIC_NESTING,
    Axis,
    EntryLayout,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    BlockParams,
    Output,
    SelectorMarker,
    SelectorUse,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    CycleError,
    KindMismatchError,
    LineageError,
    MutationConflictError,
    ParamsValidationError,
    SelectorError,
    StepPath,
    UnknownBlockError,
    WorkflowCompileError,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    WILDCARD_KIND,
    WILDCARD_KIND_NAME,
    Kind,
    kinds_compatible,
)
from roboflow_workflows.execution_engine.v2.plan import (
    Binding,
    BoundaryPort,
    ChildInputPort,
    ChildOutputPort,
    CompiledWorkflow,
    CompileOptions,
    Constant,
    Gate,
    InputPort,
    PlannedChildInput,
    PlannedChildOutput,
    PlannedInput,
    PlannedOutput,
    PlannedOutputGroup,
    PlannedSource,
    PlannedStep,
    PlannedWorkflowOutput,
    Source,
    SourcePort,
    StepPort,
    derive_dependencies,
    derive_domain,
)


@dataclass
class _Site:
    """A block step after parameter validation and selector resolution."""

    path: StepPath
    entry: CatalogueEntry
    params: BlockParams
    outputs: Mapping[str, Output]
    data: List[Tuple[SelectorUse, Resolution]] = field(default_factory=list)
    targets: Dict[str, Tuple[StepPath, ...]] = field(default_factory=dict)
    child_targets: Dict[str, StepPath] = field(default_factory=dict)
    dependencies: Set[StepPath] = field(default_factory=set)

    @property
    def location(self) -> str:
        return format_step_path(self.path)

    def where(self, use: SelectorUse) -> str:
        return f"{self.location} {_field_text(use.field_path)}"

    def bound_positions(self) -> Set[Tuple[str, Tuple[Any, ...]]]:
        return {(use.field, use.position) for use, _ in self.data}


@dataclass(frozen=True)
class _Bound:
    """A resolved data leaf with the layout of its source."""

    use: SelectorUse
    source: Source
    layout: EntryLayout

    @property
    def varying_layout(self) -> Optional[EntryLayout]:
        """Invocation layout this leaf requires; ``None`` for scalars."""
        if not self.layout.axes:
            return None
        if self.use.marker.role == "group":
            return self.layout.remove_last_axis()

        return self.layout


def compile_composition(
    composition: Composition,
    *,
    catalogue: Catalogue,
    options: CompileOptions,
) -> CompiledWorkflow:
    """Plan every block step of a composed workflow.

    Args:
        composition: Scope tree returned by ``compose_workflow``.
        catalogue: Catalogue with every block the steps name, including
            assembled dynamic blocks.
        options: Compile options, including the mutation conflict policy.

    Returns:
        The validated plan.

    Raises:
        WorkflowCompileError: A subclass naming the step, field and reason.
    """
    check_active_definition(composition.root)
    sites = _prepare_sites(composition.root, catalogue=catalogue)
    catalogue = _with_output_kinds(catalogue, sites=sites)
    _check_input_kinds(composition.root, catalogue=catalogue)
    inputs = {
        name: PlannedInput(
            name=name,
            kinds=declaration.kinds,
            layout=declaration.layout,
            required=declaration.required,
            default=declaration.default,
            declared_type=declaration.declared_type,
        )
        for name, declaration in composition.root.workflow.inputs.items()
    }
    sources = plan_sources(composition.root, catalogue=catalogue, inputs=inputs)
    gate_edges = _add_control_edges(sites)
    child_gates = _child_gates(sites)
    _add_child_output_dependencies(sites, child_gates=child_gates)
    order = _order(sites)

    planned: Dict[StepPath, PlannedStep] = {}
    boundaries = _Boundaries(
        inputs=inputs, sources=sources, planned=planned, child_gates=child_gates
    )
    for path in order:
        planned[path] = _plan_step(
            sites[path],
            planned=planned,
            boundaries=boundaries,
            gate_edges=gate_edges.get(path, []),
            order=order,
            catalogue=catalogue,
        )
    steps = tuple(planned[path] for path in order)
    _check_child_inputs(composition.root, boundaries=boundaries, catalogue=catalogue)
    outputs = _plan_workflow_outputs(
        composition.root, planned=planned, boundaries=boundaries
    )
    output_groups = _plan_output_groups(
        composition.root, planned=planned, boundaries=boundaries, sources=sources
    )
    mutation_warnings = _check_mutations(
        steps, boundaries=boundaries.records, options=options
    )

    try:
        plan = CompiledWorkflow(
            inputs=inputs,
            steps=steps,
            outputs=outputs,
            child_inputs=boundaries.child_inputs(),
            child_outputs=boundaries.child_outputs(),
            catalogue=catalogue,
            options=options,
            warnings=composition.warnings + mutation_warnings,
            sources=sources,
            output_groups=output_groups,
        )
    except ContractError as error:
        raise WorkflowCompileError(f"Compiled plan is inconsistent: {error}") from error

    return plan


def _with_output_kinds(
    catalogue: Catalogue, *, sites: Mapping[StepPath, _Site]
) -> Catalogue:
    """Add the kinds of configured outputs to a fresh catalogue for the plan.

    ``describe_outputs`` may return ``Kind`` objects the catalogue never saw.
    The executor looks kinds up by name in the plan's catalogue, so without
    them their validators, converters and serializers would be lost. A kind
    whose name another kind already uses is rejected, as when registering.
    The built-in wildcard is neutral; an explicit wildcard policy replaces it.

    Returns:
        ``catalogue`` itself when every output kind is known, else a new
        catalogue that also holds the new kinds; the given one is unchanged.

    Raises:
        KindMismatchError: When two different kinds share a name.
    """
    added: Dict[str, Tuple[Kind, _Site]] = {}
    for site in sites.values():
        for output in site.outputs.values():
            for kind in output.kinds:
                previous = added.get(kind.name)
                known = previous[0] if previous else catalogue.kinds.get(kind.name)
                if kind is WILDCARD_KIND:
                    continue
                if known is None or known is WILDCARD_KIND:
                    added[kind.name] = (kind, site)
                    continue
                if known == kind:
                    continue

                other = previous[1].location if previous else "the catalogue"
                raise KindMismatchError(
                    f"{site.location} configures an output of kind {kind.name!r}, but "
                    f"{other} already uses a different Kind with that name; blocks "
                    "must share one Kind object per name",
                    step_path=site.path,
                )
    if not added:
        return catalogue

    extended = Catalogue.merge(
        catalogue, Catalogue(kinds=[kind for kind, _ in added.values()])
    )

    return extended


def _check_input_kinds(scope: Scope, *, catalogue: Catalogue) -> None:
    for declaration in scope.workflow.inputs.values():
        unknown = [kind for kind in declaration.kinds if kind not in catalogue.kinds]
        if unknown:
            raise WorkflowCompileError(
                f"{declaration.location} ($inputs.{declaration.name}) declares unknown "
                f"kinds {unknown}; the catalogue knows {sorted(catalogue.kinds)}",
                step_path=scope.path,
            )
    for child in scope.children.values():
        _check_input_kinds(child, catalogue=catalogue)


def _prepare_sites(root: Scope, *, catalogue: Catalogue) -> Dict[StepPath, _Site]:
    sites: Dict[StepPath, _Site] = {}
    for scope, declaration in root.block_steps():
        path = scope.step_path(declaration.name)
        entry = catalogue.find(declaration.type)
        if entry is None:
            raise UnknownBlockError(
                f"{format_step_path(path)} uses unknown block type "
                f"{declaration.type!r}; known types: {sorted(catalogue.block_types)}",
                step_path=path,
            )

        params = entry.spec.validate_params(declaration.params, step_path=path)
        try:
            outputs = entry.spec.resolve_outputs(params)
        except ContractError as error:
            raise ParamsValidationError(
                f"{format_step_path(path)} ({entry.spec.type}) configures invalid "
                f"outputs: {error}",
                step_path=path,
            ) from error

        site = _Site(path=path, entry=entry, params=params, outputs=outputs)
        for use in entry.spec.find_selectors(params):
            if use.marker.role == "step":
                site.targets[use.selector] = scope.resolve_targets(
                    use.selector,
                    location=site.where(use),
                    step_path=path,
                    field_path=use.field_path,
                )
                target_name = use.selector[len("$steps.") :]
                if target_name in scope.children:
                    site.child_targets[use.selector] = scope.children[target_name].path
                continue

            resolution = scope.resolve_data(
                use.selector,
                location=site.where(use),
                step_path=path,
                field_path=use.field_path,
            )
            if isinstance(resolution.origin, StepPort):
                if resolution.origin.output == "*":
                    raise SelectorError(
                        f"{site.where(use)} binds {use.selector!r}, which selects every "
                        "output of a step; $steps.<step>.* is only valid in workflow "
                        "outputs",
                        step_path=path,
                        field_path=use.field_path,
                    )
                site.dependencies.add(resolution.origin.step)
            site.data.append((use, resolution))
        sites[path] = site

    return sites


def _add_control_edges(
    sites: Mapping[StepPath, _Site],
) -> Dict[StepPath, List[Tuple[StepPath, str]]]:
    gate_edges: Dict[StepPath, List[Tuple[StepPath, str]]] = {}
    for controller in sites.values():
        for target, paths in controller.targets.items():
            for path in paths:
                if path == controller.path:
                    raise CycleError(
                        f"{controller.location} targets itself via {target!r}",
                        step_path=controller.path,
                    )
                sites[path].dependencies.add(controller.path)
                edges = gate_edges.setdefault(path, [])
                if (controller.path, target) not in edges:
                    edges.append((controller.path, target))

    return gate_edges


def _order(sites: Mapping[StepPath, _Site]) -> Tuple[StepPath, ...]:
    # Depth-first in declaration order: producers and controllers first,
    # otherwise the definition order is kept.
    declared = list(sites)
    order: List[StepPath] = []
    state: Dict[StepPath, str] = {}
    stack: List[StepPath] = []

    def visit(path: StepPath) -> None:
        if state.get(path) == "done":
            return
        if state.get(path) == "active":
            cycle = stack[stack.index(path) :] + [path]
            raise CycleError(
                "Steps form a dependency cycle through data and control edges: "
                + " -> ".join(format_step_path(item) for item in cycle),
                step_path=path,
            )

        state[path] = "active"
        stack.append(path)
        for dependency in sorted(sites[path].dependencies, key=declared.index):
            visit(dependency)
        stack.pop()
        state[path] = "done"
        order.append(path)

    for path in declared:
        visit(path)

    return tuple(order)


def _plan_step(
    site: _Site,
    *,
    planned: Mapping[StepPath, PlannedStep],
    boundaries: "_Boundaries",
    gate_edges: List[Tuple[StepPath, str]],
    order: Tuple[StepPath, ...],
    catalogue: Catalogue,
) -> PlannedStep:
    bound = [
        _bind(
            site,
            use=use,
            resolution=resolution,
            boundaries=boundaries,
            catalogue=catalogue,
        )
        for use, resolution in site.data
    ]
    gates = tuple(
        Gate(
            controller=controller,
            target=target,
            controller_layout=planned[controller].invocation_layout,
        )
        for controller, target in gate_edges
    )
    invocation = _invocation_layout(site, bound=bound, gates=gates)
    bindings = _in_declaration_order(
        site,
        [_binding(site, item=item, invocation=invocation) for item in bound]
        + _literal_bindings(site, invocation=invocation),
    )
    domain = _step_domain(site, bindings=bindings, gates=gates, boundaries=boundaries)

    try:
        step = PlannedStep(
            path=site.path,
            spec=site.entry.spec,
            namespace=site.entry.namespace,
            params=site.params,
            bindings=bindings,
            invocation_layout=invocation,
            outputs=_plan_step_outputs(site, invocation=invocation, bindings=bindings),
            gates=gates,
            control_targets=site.targets,
            dependencies=tuple(path for path in order if path in site.dependencies),
            domain=domain,
        )
    except ContractError as error:
        raise LineageError(str(error), step_path=site.path) from error

    return step


def _step_domain(
    site: _Site,
    *,
    bindings: Tuple[Binding, ...],
    gates: Tuple[Gate, ...],
    boundaries: "_Boundaries",
) -> Optional[str]:
    """The one source this step's data, gates and boundaries reach, if any."""
    try:
        domain = derive_domain(
            [binding.source for binding in bindings],
            controllers=[gate.controller for gate in gates],
            steps=boundaries.planned,
            boundaries=boundaries.records,
        )
    except ContractError as error:
        raise LineageError(
            f"{site.location} {error}; give each source its own steps and output "
            "groups",
            step_path=site.path,
        ) from error

    return domain


def _bind(
    site: _Site,
    *,
    use: SelectorUse,
    resolution: Resolution,
    boundaries: "_Boundaries",
    catalogue: Catalogue,
) -> _Bound:
    origin = resolution.origin
    if isinstance(origin, Constant):
        _check_constant(site, use=use, resolution=resolution, catalogue=catalogue)
        layout = boundaries.layout_of(resolution)
        return _Bound(use=use, source=resolution.port, layout=layout)

    kinds = boundaries.kinds_of(origin)
    if kinds is None:
        raise SelectorError(
            f"{site.where(use)} binds {use.selector!r}, but {_producer_text(origin)} "
            f"has no output {origin.output!r}; its outputs are "
            f"{boundaries.output_names(origin)}",
            step_path=site.path,
            field_path=use.field_path,
        )
    if not kinds_compatible(kinds, use.marker.kind_names):
        raise KindMismatchError(
            f"{site.where(use)} accepts kinds {list(use.marker.kind_names)}, but "
            f"{use.selector!r} provides {list(kinds)}",
            step_path=site.path,
            field_path=use.field_path,
        )

    layout = boundaries.layout_of(resolution)
    bound = _Bound(use=use, source=resolution.port, layout=layout)

    return bound


def _check_constant(
    site: _Site, *, use: SelectorUse, resolution: Resolution, catalogue: Catalogue
) -> None:
    # A nested literal or default decoded by a kind's deserializer only
    # becomes a payload at run time; its JSON form is not checked here.
    constant = resolution.origin
    if _decoded_at_run_time(resolution, catalogue=catalogue):
        return

    if not _kinds_accept(use.marker.kind_names, constant.value, catalogue=catalogue):
        raise KindMismatchError(
            f"{site.where(use)} accepts kinds {list(use.marker.kind_names)}, but "
            f"{use.selector!r} is the nested workflow constant {constant.value!r}",
            step_path=site.path,
            field_path=use.field_path,
        )
    if use.marker.role != "item":
        return

    try:
        site.entry.spec.validate_resolved_value(
            use.field, constant.value, position=use.position
        )
    except ContractError as error:
        raise ParamsValidationError(
            f"{site.where(use)} receives nested workflow constant "
            f"{constant.value!r}: {error}",
            step_path=site.path,
            field_path=use.field_path,
        ) from error


def _check_child_inputs(
    scope: Scope, *, boundaries: "_Boundaries", catalogue: Catalogue
) -> None:
    # Every child input is checked against its own declaration, used or not.
    # Resolving it yields the source and every outer alias on the way, and
    # the outer aliases are checked when their own scope is visited.
    for child in scope.children.values():
        for name, declaration in child.workflow.inputs.items():
            field_path = (
                ("parameter_bindings", name)
                if name in child.bindings
                else ("inputs", name)
            )
            where = f"{format_step_path(child.path)} input {name!r}"
            resolution = child.resolve_data(
                f"$inputs.{name}",
                location=where,
                step_path=child.path,
                field_path=field_path,
            )
            problem = _child_input_problem(
                declaration,
                resolution=resolution,
                boundaries=boundaries,
                catalogue=catalogue,
            )
            if problem is not None:
                raise KindMismatchError(
                    f"{where} accepts kinds {list(declaration.kinds)}, but {problem}",
                    step_path=child.path,
                    field_path=field_path,
                )
        _check_child_inputs(child, boundaries=boundaries, catalogue=catalogue)


def _child_input_problem(
    declaration: WorkflowInputDeclaration,
    *,
    resolution: Resolution,
    boundaries: "_Boundaries",
    catalogue: Catalogue,
) -> Optional[str]:
    source = resolution.origin
    if isinstance(source, Constant):
        if _decoded_at_run_time(resolution, catalogue=catalogue) or _kinds_accept(
            declaration.kinds, source.value, catalogue=catalogue
        ):
            return None
        return f"it receives the constant {source.value!r}"

    kinds = boundaries.kinds_of(source)
    if kinds is None:
        return f"its binding {source.describe()} is not an output of {_producer_text(source)}"
    if kinds_compatible(kinds, declaration.kinds):
        return None

    return f"its binding {source.describe()} provides {list(kinds)}"


def _producer_text(source: Source) -> str:
    if isinstance(source, SourcePort):
        return f"$sources.{source.source}"

    return format_step_path(source.step)


def _decoded_at_run_time(resolution: Resolution, *, catalogue: Catalogue) -> bool:
    # The first hop declares the input the literal or default was written for.
    origin = resolution.hops[0].declaration
    decoded = any(catalogue.kind(name).deserialize is not None for name in origin.kinds)

    return decoded


def _kinds_accept(
    kind_names: Tuple[str, ...], value: Any, *, catalogue: Catalogue
) -> bool:
    # Same rule as the executor applies to payloads: None passes, the
    # wildcard passes, otherwise one kind's validator must accept the value.
    if value is None or WILDCARD_KIND_NAME in kind_names:
        return True

    for name in kind_names:
        try:
            catalogue.kind(name).check(value)
        except ContractError:
            continue
        return True

    return False


def _invocation_layout(
    site: _Site, *, bound: List[_Bound], gates: Tuple[Gate, ...]
) -> EntryLayout:
    varying = [
        (item, item.varying_layout) for item in bound if item.varying_layout is not None
    ]
    if varying:
        invocation = max((layout for _, layout in varying), key=_depth)
        if any(not _is_prefix(layout, of=invocation) for _, layout in varying):
            described = "; ".join(
                f"{_field_text(item.use.field_path)} ({item.use.selector}) needs "
                f"{list(layout.axis_ids)}"
                for item, layout in varying
            )
            raise LineageError(
                f"{site.location} binds data of unrelated lineages: {described}. "
                "Axes correspond by identity, never by size.",
                step_path=site.path,
            )
        for gate in gates:
            if not _is_prefix(gate.controller_layout, of=invocation):
                raise LineageError(
                    f"{site.location} runs over {list(invocation.axis_ids)}, but "
                    f"{format_step_path(gate.controller)} decides over "
                    f"{list(gate.controller_layout.axis_ids)}; control must decide "
                    "over the step's axes or an ancestor prefix of them",
                    step_path=site.path,
                )
        return invocation

    if not gates:
        return EntryLayout()

    invocation = max((gate.controller_layout for gate in gates), key=_depth)
    if any(not _is_prefix(gate.controller_layout, of=invocation) for gate in gates):
        described = "; ".join(
            f"{format_step_path(gate.controller)} over "
            f"{list(gate.controller_layout.axis_ids)}"
            for gate in gates
        )
        raise LineageError(
            f"{site.location} has no varying data and its controls decide over "
            f"unrelated axes: {described}",
            step_path=site.path,
        )

    return invocation


def _literal_bindings(site: _Site, *, invocation: EntryLayout) -> List[Binding]:
    # V1 casts a literal written at a batch-only or Group position, like a
    # selected scalar: Batch([7]) for batch="always", a one-element group for
    # Group (tasks/m1r-casting-check, artifacts/.../literal-batch-check).
    # Literals at "never" and "if_varying" positions, and None, stay plain.
    bound = site.bound_positions()
    bindings = []
    for name, field_spec in site.entry.spec.fields.items():
        value = getattr(site.params, name)
        field_is_bound = any(field_name == name for field_name, _ in bound)
        if value is None:
            continue
        if _casts_literals(field_spec.whole) and not field_is_bound:
            leaves = [((), value, field_spec.whole)]
        elif _casts_literals(field_spec.leaves):
            leaves = [
                ((position,), leaf, field_spec.leaves)
                for position, leaf in _container_leaves(value)
                if leaf is not None and (name, (position,)) not in bound
            ]
        else:
            continue
        bindings.extend(
            Binding(
                field=name,
                position=position,
                selector="",
                source=Constant(value=leaf),
                source_layout=EntryLayout(),
                mode="constant_group" if marker.role == "group" else "constant",
                batch=marker.batch,
                cast_layout=(
                    _cast_layout(site, field_name=name, invocation=invocation)
                    if marker.role == "group"
                    else None
                ),
            )
            for position, leaf, marker in leaves
        )

    return bindings


def _casts_literals(marker: Optional[SelectorMarker]) -> bool:
    return marker is not None and (marker.role == "group" or marker.batch == "always")


def _container_leaves(value: Any) -> List[Tuple[Any, Any]]:
    if isinstance(value, list):
        return list(enumerate(value))
    if isinstance(value, dict):
        return list(value.items())

    return []


def _in_declaration_order(site: _Site, bindings: List[Binding]) -> Tuple[Binding, ...]:
    fields = list(site.entry.spec.fields)

    def key(binding: Binding) -> Tuple[int, int]:
        if not binding.position:
            return fields.index(binding.field), -1
        positions = [
            position
            for position, _ in _container_leaves(getattr(site.params, binding.field))
        ]
        return fields.index(binding.field), positions.index(binding.position[0])

    ordered = tuple(sorted(bindings, key=key))

    return ordered


def _cast_layout(
    site: _Site, *, field_name: str, invocation: EntryLayout
) -> EntryLayout:
    # Every cast leaf of one field shares the step-owned cast axis.
    axis = Axis(
        id=f"{'/'.join(site.path)}/{field_name}/cast", kind=AXIS_KIND_STATIC_NESTING
    )
    layout = invocation.append_axis(axis)

    return layout


def _binding(site: _Site, *, item: _Bound, invocation: EntryLayout) -> Binding:
    use = item.use
    source_ids = item.layout.axis_ids
    cast_layout = None
    if use.marker.role == "group" and not source_ids:
        mode = "constant_group"
        cast_layout = _cast_layout(site, field_name=use.field, invocation=invocation)
    elif use.marker.role == "group":
        if source_ids[:-1] != invocation.axis_ids:
            raise LineageError(
                f"{site.where(use)} is a Group over the last axis of "
                f"{list(source_ids)}, but the step runs over "
                f"{list(invocation.axis_ids)}; a Group consumes the axis directly "
                "below the invocation level",
                step_path=site.path,
                field_path=use.field_path,
            )
        mode = "group"
    elif not source_ids:
        mode = "constant"
    elif source_ids == invocation.axis_ids:
        mode = "element"
    else:
        mode = "ancestor"

    if (
        site.entry.spec.accepts_batches
        and use.marker.batch == "never"
        and mode not in ("constant", "constant_group")
    ):
        raise LineageError(
            f"{site.where(use)} is passed once per call of a batch-accepting block, "
            f"so it must be constant, but {use.selector!r} varies over "
            f"{list(source_ids)}; declare batch delivery for it or bind a constant",
            step_path=site.path,
            field_path=use.field_path,
        )

    binding = Binding(
        field=use.field,
        position=use.position,
        selector=use.selector,
        source=item.source,
        source_layout=item.layout,
        mode=mode,
        batch=use.marker.batch,
        cast_layout=cast_layout,
    )

    return binding


def _plan_step_outputs(
    site: _Site, *, invocation: EntryLayout, bindings: Tuple[Binding, ...]
) -> Dict[str, PlannedOutput]:
    expanded_axes: Dict[str, Axis] = {}
    outputs: Dict[str, PlannedOutput] = {}
    for name, output in site.outputs.items():
        if output.transform == "expand":
            axis = _expanded_axis(site, output=output, name=name, shared=expanded_axes)
            layout = invocation.append_axis(axis)
        elif output.transform == "preserve":
            layout = _preserved_layout(
                site, output=output, name=name, bindings=bindings
            )
        else:
            layout = invocation
        outputs[name] = PlannedOutput(
            name=name,
            kinds=output.kind_names,
            layout=layout,
            transform=output.transform,
            group_field=output.preserve,
            source_field=output.source,
            context_policy=output.context_policy,
        )

    return outputs


def _expanded_axis(
    site: _Site, *, output: Output, name: str, shared: Dict[str, Axis]
) -> Axis:
    # One axis per (step path, axis key): outputs of one step sharing a key
    # describe the same children. Names contain no "/" or ":", so the id is
    # unique across scopes and distinct from input and cast axes.
    kind = AXIS_KIND_STATIC_NESTING if output.stationary else AXIS_KIND_DYNAMIC_NESTING
    axis = shared.setdefault(
        output.expand, Axis(id=f"{'/'.join(site.path)}:{output.expand}", kind=kind)
    )
    if axis.kind != kind:
        raise WorkflowCompileError(
            f"{site.location} output {name!r} shares axis {output.expand!r} with "
            "another output but declares a different stationarity",
            step_path=site.path,
        )

    return axis


def _preserved_layout(
    site: _Site, *, output: Output, name: str, bindings: Tuple[Binding, ...]
) -> EntryLayout:
    group_layouts = {
        binding.group_layout
        for binding in bindings
        if binding.field == output.preserve and binding.group_layout is not None
    }
    if len(group_layouts) != 1:
        found = sorted(list(layout.axis_ids) for layout in group_layouts)
        raise LineageError(
            f"{site.location} output {name!r} preserves Group field "
            f"{output.preserve!r}, which needs exactly one group layout; its "
            f"selectors give {found}",
            step_path=site.path,
            field_path=(output.preserve,),
        )

    layout = next(iter(group_layouts))

    return layout


def _plan_workflow_outputs(
    root: Scope,
    *,
    planned: Mapping[StepPath, PlannedStep],
    boundaries: "_Boundaries",
) -> Tuple[PlannedWorkflowOutput, ...]:
    _check_child_outputs(root, boundaries=boundaries)
    outputs = _plan_selections(
        root, root.workflow.outputs, planned=planned, boundaries=boundaries
    )

    return outputs


def _plan_output_groups(
    root: Scope,
    *,
    planned: Mapping[StepPath, PlannedStep],
    boundaries: "_Boundaries",
    sources: Mapping[str, PlannedSource],
) -> Tuple[PlannedOutputGroup, ...]:
    groups: List[PlannedOutputGroup] = []
    for declaration in root.workflow.output_groups:
        anchor = _resolve_anchor(root, declaration=declaration, sources=sources)
        outputs = _plan_selections(
            root, declaration.outputs, planned=planned, boundaries=boundaries
        )
        for output in outputs:
            try:
                domain = derive_domain(
                    [output.source],
                    controllers=(),
                    steps=planned,
                    boundaries=boundaries.records,
                )
            except ContractError as error:
                raise LineageError(
                    f"{declaration.location} ({declaration.name}) field "
                    f"{output.name!r} ({output.selector}) {error}",
                    field_path=("outputs", output.name),
                ) from error
            if domain not in (None, anchor.source):
                raise LineageError(
                    f"{declaration.location} ({declaration.name}) is anchored on "
                    f"{anchor.describe()}, but its field {output.name!r} "
                    f"({output.selector}) comes from source {domain!r}; a group "
                    "follows one source's pulses until an alignment operator "
                    "relates independent sources",
                    field_path=("outputs", output.name),
                )
        groups.append(
            PlannedOutputGroup(
                name=declaration.name,
                anchor=anchor,
                outputs=outputs,
                dependencies=derive_dependencies(
                    [output.source for output in outputs],
                    steps=planned,
                    boundaries=boundaries.records,
                ),
            )
        )

    return tuple(groups)


def _resolve_anchor(
    root: Scope,
    *,
    declaration: OutputGroupDeclaration,
    sources: Mapping[str, PlannedSource],
) -> SourcePort:
    resolution = root.resolve_data(
        declaration.anchor,
        location=f"{declaration.location}.anchor",
        step_path=(),
        field_path=("anchor",),
    )
    anchor = resolution.origin
    if anchor.output not in sources[anchor.source].outputs:
        raise SelectorError(
            f"{declaration.location}.anchor: $sources.{anchor.source} has no output "
            f"{anchor.output!r}; its outputs are "
            f"{list(sources[anchor.source].outputs)}",
            field_path=("anchor",),
        )

    return anchor


def _plan_selections(
    scope: Scope,
    declarations: Tuple[WorkflowOutputDeclaration, ...],
    *,
    planned: Mapping[StepPath, PlannedStep],
    boundaries: "_Boundaries",
) -> Tuple[PlannedWorkflowOutput, ...]:
    """Plan flat output selections, of the workflow or of one output group."""
    outputs: List[PlannedWorkflowOutput] = []
    for declaration in declarations:
        resolution = _resolve_output(
            scope, declaration=declaration, boundaries=boundaries
        )
        boundaries.layout_of(resolution, location=declaration.location)
        outputs.append(
            PlannedWorkflowOutput(
                name=declaration.name,
                selector=declaration.selector,
                source=resolution.port,
                options=declaration.options,
            )
        )

    return tuple(outputs)


def _child_gates(
    sites: Mapping[StepPath, _Site],
) -> Dict[StepPath, List[Tuple[StepPath, str]]]:
    """Controllers targeting each whole child, with the target selector."""
    gates: Dict[StepPath, List[Tuple[StepPath, str]]] = {}
    for controller in sites.values():
        for target, scope in controller.child_targets.items():
            gates.setdefault(scope, []).append((controller.path, target))

    return gates


def _add_child_output_dependencies(
    sites: Mapping[StepPath, _Site],
    *,
    child_gates: Mapping[StepPath, List[Tuple[StepPath, str]]],
) -> None:
    # A value leaving a gated child through a child output is ready only once
    # every controller of that child has decided (decision 026).
    for site in sites.values():
        for _, resolution in site.data:
            for hop in resolution.hops:
                if isinstance(hop.port, ChildOutputPort):
                    site.dependencies.update(
                        controller
                        for controller, _ in child_gates.get(hop.port.scope, ())
                    )


class _Boundaries:
    """Plan records of the child inputs and outputs that values pass.

    Each record is built once, in dependency order: when a consumer is
    planned, the producers and controllers behind its boundaries already are.
    """

    def __init__(
        self,
        *,
        inputs: Mapping[str, PlannedInput],
        sources: Mapping[str, PlannedSource],
        planned: Mapping[StepPath, PlannedStep],
        child_gates: Mapping[StepPath, List[Tuple[StepPath, str]]],
    ):
        self._inputs = inputs
        self._sources = sources
        self.planned = planned
        self._child_gates = child_gates
        self.records: Dict[
            BoundaryPort, Union[PlannedChildInput, PlannedChildOutput]
        ] = {}

    def kinds_of(self, origin: Source) -> Optional[Tuple[str, ...]]:
        """Kinds of an input, source port or step output; ``None`` for a missing port."""
        if isinstance(origin, InputPort):
            return self._inputs[origin.name].kinds
        if isinstance(origin, SourcePort):
            output = self._sources[origin.source].outputs.get(origin.output)
            return output.kinds if output is not None else None
        if origin.output == "*":
            return (WILDCARD_KIND_NAME,)

        output = self.planned[origin.step].outputs.get(origin.output)
        kinds = output.kinds if output is not None else None

        return kinds

    def output_names(self, origin: Union[SourcePort, StepPort]) -> List[str]:
        """Declared output names of a source or step, for error messages."""
        if isinstance(origin, SourcePort):
            return list(self._sources[origin.source].outputs)

        return list(self.planned[origin.step].outputs)

    def child_inputs(self) -> Tuple[PlannedChildInput, ...]:
        """Child input records, outer before inner."""
        return tuple(
            record
            for record in self.records.values()
            if isinstance(record, PlannedChildInput)
        )

    def child_outputs(self) -> Tuple[PlannedChildOutput, ...]:
        """Child output records."""
        return tuple(
            record
            for record in self.records.values()
            if isinstance(record, PlannedChildOutput)
        )

    def layout_of(self, resolution: Resolution, *, location: str = "") -> EntryLayout:
        """Record every boundary of ``resolution`` and return its port's layout.

        Raises:
            SelectorError: When a boundary forwards every output of a step.
            LineageError: When a child's gates and a forwarded value's axes
                are unrelated.
        """
        layout = self._origin_layout(resolution.origin)
        if resolution.hops and layout is None:
            raise SelectorError(
                f"{location}: forwards a child input bound to "
                f"{resolution.origin.describe()}; a child input cannot forward every "
                "output of a step",
            )
        for hop, source in resolution.boundaries():
            if hop.port not in self.records:
                self.records[hop.port] = self._record(hop, source=source, layout=layout)
            layout = self.records[hop.port].layout

        return layout

    def _record(
        self, hop: Hop, *, source: Source, layout: EntryLayout
    ) -> Union[PlannedChildInput, PlannedChildOutput]:
        port = hop.port
        if isinstance(port, ChildInputPort):
            record = PlannedChildInput(
                scope=port.scope,
                name=port.name,
                kinds=hop.declaration.kinds,
                layout=layout,
                source=source,
            )
            return record

        gates = tuple(
            Gate(
                controller=controller,
                target=target,
                controller_layout=self.planned[controller].invocation_layout,
            )
            for controller, target in self._child_gates.get(port.scope, ())
        )
        layouts = [layout] + [gate.controller_layout for gate in gates]
        effective = max(layouts, key=_depth)
        if not all(_is_prefix(item, of=effective) for item in layouts):
            described = "; ".join(
                f"{format_step_path(gate.controller)} over "
                f"{list(gate.controller_layout.axis_ids)}"
                for gate in gates
            )
            raise LineageError(
                f"{format_step_path(port.scope)} output {port.name!r} forwards a value "
                f"over {list(layout.axis_ids)}, but the child's controls decide over "
                f"unrelated axes: {described}",
                step_path=port.scope,
                field_path=("outputs", port.name),
            )
        record = PlannedChildOutput(
            scope=port.scope,
            name=port.name,
            layout=effective,
            source=source,
            gates=gates,
        )

        return record

    def _origin_layout(self, origin: Source) -> Optional[EntryLayout]:
        """Layout of an origin; ``None`` for a wildcard over every step output."""
        if isinstance(origin, Constant):
            return EntryLayout()
        if isinstance(origin, InputPort):
            return self._inputs[origin.name].layout
        if isinstance(origin, SourcePort):
            port = self._sources[origin.source].outputs.get(origin.output)
            if port is None:
                raise SelectorError(
                    f"$sources.{origin.source} has no output {origin.output!r}; its "
                    f"outputs are {self.output_names(origin)}"
                )
            return port.layout
        if origin.output == "*":
            return None

        layout = self.planned[origin.step].outputs[origin.output].layout

        return layout


def _check_child_outputs(scope: Scope, *, boundaries: "_Boundaries") -> None:
    # Child outputs are resolved lazily when the parent uses them; checking all
    # of them here reports a broken child definition even when it is unused.
    for child in scope.children.values():
        for declaration in child.workflow.outputs:
            _resolve_output(child, declaration=declaration, boundaries=boundaries)
        _check_child_outputs(child, boundaries=boundaries)


def _resolve_output(
    scope: Scope,
    *,
    declaration: WorkflowOutputDeclaration,
    boundaries: "_Boundaries",
) -> Resolution:
    field_path = ("outputs", declaration.name)
    resolution = scope.resolve_data(
        declaration.selector,
        location=declaration.location,
        step_path=scope.path,
        field_path=field_path,
    )
    origin = resolution.origin
    if isinstance(origin, (StepPort, SourcePort)) and origin.output != "*":
        if boundaries.kinds_of(origin) is None:
            raise SelectorError(
                f"{declaration.location}: {_producer_text(origin)} has no output "
                f"{origin.output!r}; its outputs are {boundaries.output_names(origin)}",
                step_path=scope.path,
                field_path=field_path,
            )

    return resolution


def _check_mutations(
    steps: Tuple[PlannedStep, ...],
    *,
    boundaries: Mapping[BoundaryPort, Union[PlannedChildInput, PlannedChildOutput]],
    options: CompileOptions,
) -> Tuple[str, ...]:
    by_path = {step.path: step for step in steps}
    ancestors = _ancestor_closure(steps)
    leaves = [
        (
            step,
            binding,
            _payload_origins(binding.source, steps=by_path, boundaries=boundaries),
        )
        for step in steps
        for binding in step.bindings
    ]
    warnings: List[str] = []
    reported: Set[frozenset] = set()
    for mutator, mutated, origins in leaves:
        if mutated.field not in mutator.spec.mutates:
            continue
        for reader, read, reader_origins in leaves:
            if reader.path == mutator.path or not origins & reader_origins:
                continue
            if (
                reader.path in ancestors[mutator.path]
                or mutator.path in ancestors[reader.path]
            ):
                continue
            pair = frozenset(
                {(mutator.path, mutated.field_path), (reader.path, read.field_path)}
            )
            if pair in reported:
                continue
            reported.add(pair)
            message = (
                f"{format_step_path(mutator.path)} {_field_text(mutated.field_path)} "
                f"({mutated.selector}) is mutated in place, and "
                f"{format_step_path(reader.path)} {_field_text(read.field_path)} "
                f"({read.selector}) may see the same payload without an ordering "
                "dependency between the two steps"
            )
            if options.mutation_conflicts == "error":
                raise MutationConflictError(
                    message, step_path=mutator.path, field_path=mutated.field_path
                )
            warnings.append(message)

    return tuple(warnings)


def _ancestor_closure(steps: Tuple[PlannedStep, ...]) -> Dict[StepPath, Set[StepPath]]:
    closure: Dict[StepPath, Set[StepPath]] = {}
    for step in steps:
        reached: Set[StepPath] = set()
        for dependency in step.dependencies:
            reached |= {dependency} | closure[dependency]
        closure[step.path] = reached

    return closure


def _payload_origins(
    source: Source,
    *,
    steps: Mapping[StepPath, PlannedStep],
    boundaries: Mapping[BoundaryPort, Union[PlannedChildInput, PlannedChildOutput]],
) -> Set[Any]:
    if isinstance(source, (ChildInputPort, ChildOutputPort)):
        # A child input over a constant materializes its own value once per
        # run, shared by its consumers; any other boundary (every child
        # output included) passes its source's payload on.
        boundary = boundaries[source]
        if isinstance(source, ChildInputPort) and isinstance(boundary.source, Constant):
            return {source}
        return _payload_origins(boundary.source, steps=steps, boundaries=boundaries)
    if isinstance(source, Constant):
        if isinstance(source.value, (str, bytes, int, float, bool, type(None))):
            return set()
        return {("constant", id(source.value))}

    origins: Set[Any] = {source}
    if isinstance(source, StepPort):
        producer = steps[source.step]
        source_field = producer.outputs[source.output].source_field
        for binding in producer.bindings_for(source_field) if source_field else ():
            origins |= _payload_origins(
                binding.source, steps=steps, boundaries=boundaries
            )

    return origins


def _is_prefix(layout: EntryLayout, *, of: EntryLayout) -> bool:
    return of.axis_ids[: layout.depth] == layout.axis_ids


def _depth(layout: EntryLayout) -> int:
    return layout.depth


def _field_text(field_path: Tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in field_path)
