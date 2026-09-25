"""Workload introspection builder for Execution Engine v1.

Turns a `StructuralCompilationResult` (see `compiler.core.compile_workflow_structure`)
into the public `WorkflowIntrospection` response: the connectivity graph, per-step
compile-time facts, declared operations / restrictions / resources, the model
inventory and whole-graph summaries.

Nothing here executes: no block is initialised, no model is loaded, no custom
Python is evaluated. The only outbound call is the optional, host-owned
`ModelMetadataProvider`, invoked once per unique literal `(provider, model_id)`
and never with a `$`-prefixed selector.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Type

from networkx import DiGraph
from roboflow_workflows.errors import AssumptionError
from roboflow_workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from roboflow_workflows.execution_engine.entities.workload import (
    DeclarationDomain,
    Discovery,
    DiscoveryProblem,
    DiscoveryProblemCode,
    ModelMetadataLookup,
    ModelMetadataProvider,
    RestrictionMetadata,
    WorkOperation,
    declaration_failed_problem,
    declaration_unavailable_problem,
    invalid_resource_identifier_problem,
    normalize_declaration,
    opaque_remote_workflow_problem,
    restriction_metadata_of,
    unresolved_selector_problem,
    with_block_type,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    _cached_model_json_schema,
    get_manifest_type_identifiers,
)
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    GraphEdge,
    GraphNode,
    ModelSummary,
    StepMetadata,
    WorkflowIntrospection,
    WorkflowSummary,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    ExecutionGraphNode,
    StepNode,
    StructuralCompilationResult,
)
from roboflow_workflows.execution_engine.v1.compiler.graph_constructor import (
    STEP_INPUT_SELECTORS_PROPERTY,
)
from roboflow_workflows.execution_engine.v1.compiler.utils import (
    construct_input_selector,
    construct_output_selector,
    construct_step_selector,
    is_input_node,
    is_output_node,
    is_step_node,
    node_as,
)
from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import (
    BLOCK_SOURCE as DYNAMIC_BLOCKS_SOURCE,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.constants import (
    INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH,
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from roboflow_workflows.prototypes.block import (
    DependentResource,
    DependentResourceType,
    WorkflowBlockManifest,
    is_workflow_selector,
)

ROBOFLOW_MODEL_PROVIDER = "roboflow"

RESOURCES_HOOK = "discover_dependent_resources"
OPERATIONS_HOOK = "discover_work_operations"

ModelReference = Tuple[str, str]


def build_workflow_introspection(
    compilation_result: StructuralCompilationResult,
    model_metadata_provider: Optional[ModelMetadataProvider] = None,
) -> WorkflowIntrospection:
    """Build a fresh `WorkflowIntrospection` from a structural compilation."""
    execution_graph = compilation_result.execution_graph
    definition = compilation_result.parsed_workflow_definition
    nodes = collect_graph_nodes(compilation_result=compilation_result)
    edges = collect_graph_edges(execution_graph=execution_graph)
    specification_by_manifest_class = {
        specification.manifest_class: specification
        for specification in compilation_result.available_blocks
    }
    steps = [
        describe_step(
            step_manifest=step_manifest,
            execution_graph=execution_graph,
            specification_by_manifest_class=specification_by_manifest_class,
        )
        for step_manifest in definition.steps
    ]
    models = build_models_inventory(
        steps=steps,
        model_metadata_provider=model_metadata_provider,
    )
    summary = WorkflowSummary(
        models=models,
        max_dimensionality=compute_max_dimensionality(
            execution_graph=execution_graph,
            steps=steps,
        ),
        steps_by_dimensionality=dict(
            Counter(step.input_dimensionality for step in steps)
        ),
    )
    return WorkflowIntrospection(
        execution_engine_version=str(EXECUTION_ENGINE_V1_VERSION),
        nodes=nodes,
        edges=edges,
        steps=steps,
        summary=summary,
    )


def collect_graph_nodes(
    compilation_result: StructuralCompilationResult,
) -> List[GraphNode]:
    """Inputs, steps and outputs in definition order, identified by their
    canonical selectors. Synthetic compiler nodes (the `<super-input>` used
    during lineage denotation) are removed by the compiler before the graph is
    returned; this is asserted rather than assumed."""
    definition = compilation_result.parsed_workflow_definition
    nodes = [
        GraphNode(id=construct_input_selector(input_name=element.name), kind="input")
        for element in definition.inputs
    ]
    nodes.extend(
        GraphNode(id=construct_step_selector(step_name=step.name), kind="step")
        for step in definition.steps
    )
    nodes.extend(
        GraphNode(id=construct_output_selector(name=output.name), kind="output")
        for output in definition.outputs
    )
    declared_ids = {node.id for node in nodes}
    graph_ids = set(compilation_result.execution_graph.nodes)
    if declared_ids != graph_ids:
        raise AssumptionError(
            public_message="Workflow introspection expected the execution graph nodes to match the "
            f"parsed definition, but found graph-only nodes {sorted(graph_ids - declared_ids)} "
            f"and definition-only nodes {sorted(declared_ids - graph_ids)}. This is most "
            "likely the bug. Contact Roboflow team through github issues "
            "(https://github.com/roboflow/inference/issues) providing full context of the "
            "problem - including workflow definition you use.",
            context="workflow_introspection | graph_nodes_collection",
        )
    return nodes


def collect_graph_edges(execution_graph: DiGraph) -> List[GraphEdge]:
    """Deduplicated `(source, target, kind)` records sorted by that triple.

    The compiled graph carries no single authoritative edge kind: data edges
    into a step carry `STEP_INPUT_SELECTORS_PROPERTY`, edges from inputs and
    into outputs are data by construction, and control edges are the ones the
    source step registered in `StepNode.child_execution_branches`. One node
    pair may yield both records.
    """
    records: Set[Tuple[str, str, str]] = set()
    for source, target, edge_data in execution_graph.edges(data=True):
        matched = False
        if (
            STEP_INPUT_SELECTORS_PROPERTY in edge_data
            or is_input_node(execution_graph=execution_graph, node=source)
            or is_output_node(execution_graph=execution_graph, node=target)
        ):
            records.add((source, target, "data"))
            matched = True
        if is_step_node(execution_graph=execution_graph, node=source):
            source_node = node_as(
                execution_graph=execution_graph,
                node=source,
                expected_type=StepNode,
            )
            if target in source_node.child_execution_branches:
                records.add((source, target, "control"))
                matched = True
        if not matched:
            raise AssumptionError(
                public_message=f"Workflow introspection could not classify execution graph edge "
                f"{source} -> {target} as data or control. This is most likely the bug. "
                "Contact Roboflow team through github issues "
                "(https://github.com/roboflow/inference/issues) providing full context of the "
                "problem - including workflow definition you use.",
                context="workflow_introspection | graph_edges_collection",
            )
    return [
        GraphEdge(source=source, target=target, kind=kind)
        for source, target, kind in sorted(records)
    ]


def describe_step(
    step_manifest: WorkflowBlockManifest,
    execution_graph: DiGraph,
    specification_by_manifest_class: Dict[
        Type[WorkflowBlockManifest], BlockSpecification
    ],
) -> StepMetadata:
    step_selector = construct_step_selector(step_name=step_manifest.name)
    step_node = node_as(
        execution_graph=execution_graph,
        node=step_selector,
        expected_type=StepNode,
    )
    specification = specification_by_manifest_class.get(type(step_manifest))
    if specification is None:
        raise AssumptionError(
            public_message=f"Workflow introspection could not match step `{step_selector}` manifest "
            f"of type {type(step_manifest)} to any loaded block. This is most likely the bug. "
            "Contact Roboflow team through github issues "
            "(https://github.com/roboflow/inference/issues) providing full context of the "
            "problem - including workflow definition you use.",
            context="workflow_introspection | step_description",
        )
    block_type = canonical_block_type(specification=specification)
    opaque = is_dispatched_inner_workflow(
        block_type=block_type, step_manifest=step_manifest
    )
    declared_resources = collect_declaration(
        step_manifest=step_manifest,
        hook_name=RESOURCES_HOOK,
        discovery_type=Discovery[DependentResource],
        declaration="resources",
        step_selector=step_selector,
        block_type=block_type,
        opaque=opaque,
    )
    resources = add_resource_identity_problems(
        discovery=declared_resources,
        step_selector=step_selector,
    )
    return StepMetadata(
        node_id=step_selector,
        block_type=block_type,
        input_dimensionality=step_node.reference_dimensionality,
        output_dimensionality=step_node.output_dimensionality,
        accepts_batch_input=bool(step_manifest.accepts_batch_input()),
        resources=resources,
        restrictions=collect_restrictions(
            step_manifest=step_manifest,
            step_selector=step_selector,
            block_type=block_type,
            opaque=opaque,
        ),
        operations=collect_declaration(
            step_manifest=step_manifest,
            hook_name=OPERATIONS_HOOK,
            discovery_type=Discovery[WorkOperation],
            declaration="operations",
            step_selector=step_selector,
            block_type=block_type,
            opaque=opaque,
        ),
    )


def canonical_block_type(specification: BlockSpecification) -> str:
    """The canonical manifest type identifier (first `type` literal), not the
    alias a definition may have used."""
    manifest_class = specification.manifest_class
    if specification.block_source == DYNAMIC_BLOCKS_SOURCE:
        # Dynamic manifests are unique per compilation - keep them out of the
        # process-wide schema cache.
        block_schema = manifest_class.model_json_schema()
    else:
        block_schema = _cached_model_json_schema(manifest_class)
    return get_manifest_type_identifiers(
        block_schema=block_schema,
        block_source=specification.block_source,
        block_identifier=specification.identifier,
    )[0]


def is_dispatched_inner_workflow(
    block_type: str, step_manifest: WorkflowBlockManifest
) -> bool:
    """A `remote_dispatch` inner workflow stays an opaque step: its child is
    compiled by the target server, so nothing about the child's resources,
    operations or restrictions is known here."""
    return (
        block_type == USE_INNER_WORKFLOW_BLOCK_TYPE
        and getattr(step_manifest, "execution_mode", None)
        == INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH
    )


def collect_declaration(
    step_manifest: WorkflowBlockManifest,
    hook_name: str,
    discovery_type: Type[Discovery],
    declaration: DeclarationDomain,
    step_selector: str,
    block_type: str,
    opaque: bool = False,
) -> Discovery:
    """Call one manifest hook and normalise its answer.

    `None` -> unknown (incomplete with a `declaration_unavailable` problem), a
    list -> complete, a `Discovery` -> itself. A hook that raises, or returns
    something that does not validate as the expected discovery, yields an
    incomplete discovery with a `declaration_failed` problem - compilation and
    schema errors were raised earlier, and a broken declaration must not hide
    the whole graph. The exception itself is never reported: it may carry
    anything the block put in it.
    """
    try:
        declared = getattr(step_manifest, hook_name)()
        normalised = normalize_declaration(
            declared,
            declaration_unavailable_problem(
                node_id=step_selector,
                declaration=declaration,
                block_type=block_type,
            ),
        )
        discovery = discovery_type(
            items=list(normalised.items),
            complete=normalised.complete,
            unknown_reasons=list(normalised.unknown_reasons),
        )
    except Exception:
        discovery = discovery_type(
            items=[],
            complete=False,
            unknown_reasons=[
                declaration_failed_problem(
                    node_id=step_selector,
                    declaration=declaration,
                    block_type=block_type,
                )
            ],
        )
    return add_opaque_problem(
        discovery=discovery,
        discovery_type=discovery_type,
        declaration=declaration,
        step_selector=step_selector,
        opaque=opaque,
    )


def collect_restrictions(
    step_manifest: WorkflowBlockManifest,
    step_selector: str,
    block_type: str,
    opaque: bool = False,
) -> Discovery[RestrictionMetadata]:
    """The step's restrictions, as the PORTABLE view.

    The builder asks for `ignore_environment_restrictions=True` explicitly: a
    workload document describes the definition, not the server that answered,
    so every conditional declaration is reported with its condition intact and
    nothing is filtered against this host's configuration. The authored
    `RuntimeRestriction` entities are then projected onto the wire DTO, which
    drops their human notes.

    A hook that raises, or answers with something that does not validate,
    yields `declaration_failed` exactly as for the other declarations. The
    problems a manifest raises know the step but not the canonical block
    identifier, so the ones that carry `block_type` by convention get it here.
    """
    try:
        declared = step_manifest.get_actual_restrictions(
            ignore_environment_restrictions=True
        )
        discovery = Discovery[RestrictionMetadata](
            items=[restriction_metadata_of(item) for item in declared.items],
            complete=declared.complete,
            unknown_reasons=[
                with_block_type(problem=reason, block_type=block_type)
                for reason in declared.unknown_reasons
            ],
        )
    except Exception:
        discovery = Discovery[RestrictionMetadata](
            items=[],
            complete=False,
            unknown_reasons=[
                declaration_failed_problem(
                    node_id=step_selector,
                    declaration="restrictions",
                    block_type=block_type,
                )
            ],
        )
    return add_opaque_problem(
        discovery=discovery,
        discovery_type=Discovery[RestrictionMetadata],
        declaration="restrictions",
        step_selector=step_selector,
        opaque=opaque,
    )


def add_opaque_problem(
    discovery: Discovery,
    discovery_type: Type[Discovery],
    declaration: DeclarationDomain,
    step_selector: str,
    opaque: bool,
) -> Discovery:
    """A `remote_dispatch` inner workflow can never be complete: whatever the
    step itself declared, the child it dispatches is compiled elsewhere."""
    if not opaque:
        return discovery
    return discovery_type(
        items=list(discovery.items),
        complete=False,
        unknown_reasons=list(discovery.unknown_reasons)
        + [
            opaque_remote_workflow_problem(
                node_id=step_selector, declaration=declaration
            )
        ],
    )


def add_resource_identity_problems(
    discovery: Discovery[DependentResource],
    step_selector: str,
) -> Discovery[DependentResource]:
    """Mark a step's resources incomplete when a resource identity is unknown.

    A legacy hook returns a plain list, which `collect_declaration` reads as
    complete. The list's shape may be known while a resource identity is not:
    an identity field fed by a workflow selector is only known at run time,
    and a blank literal names nothing. Every declared item is kept; each such
    field adds one `unresolved_selector` or `invalid_resource_identifier`
    problem, next to the reasons already reported. Selectors are never
    resolved, `model_id_resolver` never runs, and no default id is guessed.

    Args:
        discovery: The step's collected resources.
        step_selector: Canonical `$steps.<name>` id of the step.

    Returns:
        The same discovery when every identity is a non-blank literal,
        otherwise an incomplete copy carrying the identity problems.
    """
    problems: List[DiscoveryProblem] = []
    for resource in discovery.items:
        resource_type = resource.resource_type.value
        for field, value in resource_identity_fields(resource=resource):
            if is_workflow_selector(value):
                problems.append(
                    unresolved_selector_problem(
                        node_id=step_selector,
                        declaration="resources",
                        field=field,
                        selector=value,
                        resource_type=resource_type,
                    )
                )
            elif not str(value).strip():
                problems.append(
                    invalid_resource_identifier_problem(
                        node_id=step_selector,
                        declaration="resources",
                        field=field,
                        resource_type=resource_type,
                    )
                )
    if not problems:
        return discovery

    return Discovery[DependentResource](
        items=list(discovery.items),
        complete=False,
        unknown_reasons=list(discovery.unknown_reasons) + problems,
    )


def _is_project_identity_problem(problem: DiscoveryProblem) -> bool:
    """A selector / blank project identity says nothing about models."""
    return (
        problem.code
        in (
            DiscoveryProblemCode.UNRESOLVED_SELECTOR,
            DiscoveryProblemCode.INVALID_RESOURCE_IDENTIFIER,
        )
        and problem.details.get("declaration") == "resources"
        and problem.details.get("resource_type")
        == DependentResourceType.ROBOFLOW_PLATFORM_PROJECT.value
    )


def build_models_inventory(
    steps: List[StepMetadata],
    model_metadata_provider: Optional[ModelMetadataProvider],
) -> Discovery[ModelSummary]:
    """Inventory of literal model references across all steps.

    * platform models -> provider `roboflow`; third-party models keep their
      declared provider; projects are not models;
    * selector-valued references are never inventory entries - they stay in
      the per-step resources and add an `unresolved_selector` problem naming
      the resource field and its selector, one per selector-valued field;
    * blank (empty / whitespace-only) literal ids or providers are declared
      as-is per step but never inventoried, never sent to the metadata
      provider, and add an `invalid_resource_identifier` problem naming the
      blank field (never its value) - an id is never fabricated;
    * a step with unknown / incomplete resources propagates its problems
      unchanged, with all the context they carry - except selector / blank
      problems about a project identity: projects are not models, so they
      keep the step's resources incomplete but not the model inventory;
    * entries are unique per `(provider, model_id)` with sorted referring
      steps; metadata lookup outcomes never change `complete`;
    * `steps_by_dimensionality` is derived from that same deduplicated set of
      referring steps (one count per step id, mapped to the step's compiled
      input depth), so it always sums to `len(used_by_steps)`.
    """
    used_by_steps: Dict[ModelReference, Set[str]] = defaultdict(set)
    input_dimensionality_by_step = {
        step.node_id: step.input_dimensionality for step in steps
    }
    problems: List[DiscoveryProblem] = []
    for step in steps:
        if not step.resources.complete:
            problems.extend(
                reason
                for reason in step.resources.unknown_reasons
                if not _is_project_identity_problem(problem=reason)
            )
        for resource in step.resources.items:
            reference = model_reference_of(resource=resource)
            if reference is None:
                continue
            if resource.metadata.requires_runtime_resolution():
                problems.extend(
                    unresolved_selector_problem(
                        node_id=step.node_id,
                        declaration="resources",
                        field=field,
                        selector=selector,
                        resource_type=resource.resource_type.value,
                    )
                    for field, selector in selector_valued_identity_fields(
                        resource=resource
                    )
                )
                continue
            blank_fields = blank_identity_fields(resource=resource)
            if blank_fields:
                problems.extend(
                    invalid_resource_identifier_problem(
                        node_id=step.node_id,
                        declaration="resources",
                        field=field,
                        resource_type=resource.resource_type.value,
                    )
                    for field in blank_fields
                )
                continue
            used_by_steps[reference].add(step.node_id)
    references = sorted(used_by_steps.keys())
    lookups = resolve_models_metadata(
        references=references,
        model_metadata_provider=model_metadata_provider,
    )
    items = [
        ModelSummary(
            provider=provider,
            model_id=model_id,
            used_by_steps=sorted(used_by_steps[(provider, model_id)]),
            steps_by_dimensionality=dict(
                Counter(
                    input_dimensionality_by_step[step_id]
                    for step_id in used_by_steps[(provider, model_id)]
                )
            ),
            metadata=lookups[(provider, model_id)].metadata,
            metadata_status=lookups[(provider, model_id)].status,
        )
        for provider, model_id in references
    ]
    if problems:
        # `Discovery` deduplicates by (code, details) and orders by that key,
        # so the aggregate keeps every distinct step / field / selector context
        # and never depends on the order the steps were visited in.
        return Discovery[ModelSummary](
            items=items,
            complete=False,
            unknown_reasons=problems,
        )
    return Discovery[ModelSummary](items=items, complete=True, unknown_reasons=[])


def resource_identity_fields(resource: DependentResource) -> List[Tuple[str, str]]:
    """List the metadata fields that identify a declared resource.

    For models these are the fields `model_reference_of()` reads; for every
    resource they are exactly the ones `requires_runtime_resolution()`
    inspects. A blank (empty / whitespace-only) value names nothing:
    `ModelSummary` rejects it and no metadata lookup could resolve it.

    Args:
        resource: One declared resource.

    Returns:
        `(metadata field, value)` pairs, in a fixed order per resource type;
        empty for any other resource type.
    """
    if resource.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_MODEL:
        return [("model_id", resource.metadata.model_id)]
    if resource.resource_type is DependentResourceType.THIRD_PARTY_MODEL:
        return [
            ("provider", resource.metadata.provider),
            ("model_id", resource.metadata.model_id),
        ]
    if resource.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_PROJECT:
        return [("project_url", resource.metadata.project_url)]
    return []


def selector_valued_identity_fields(
    resource: DependentResource,
) -> List[Tuple[str, str]]:
    """Identity fields fed by a workflow selector - one unresolved-selector
    problem per field, so two different selectors never collapse into one."""
    return [
        (field, value)
        for field, value in resource_identity_fields(resource=resource)
        if is_workflow_selector(value)
    ]


def blank_identity_fields(resource: DependentResource) -> List[str]:
    """Identity fields whose literal value names nothing (empty/whitespace).
    Only the field names are reported - the invalid value never is."""
    return [
        field
        for field, value in resource_identity_fields(resource=resource)
        if not str(value).strip()
    ]


def model_reference_of(resource: DependentResource) -> Optional[ModelReference]:
    if resource.resource_type is DependentResourceType.ROBOFLOW_PLATFORM_MODEL:
        return ROBOFLOW_MODEL_PROVIDER, resource.metadata.model_id
    if resource.resource_type is DependentResourceType.THIRD_PARTY_MODEL:
        return resource.metadata.provider, resource.metadata.model_id
    return None


def resolve_models_metadata(
    references: List[ModelReference],
    model_metadata_provider: Optional[ModelMetadataProvider],
) -> Dict[ModelReference, ModelMetadataLookup]:
    """One lookup per unique literal reference; no provider or a failing
    provider means `unavailable` (never a dropped inventory entry)."""
    result: Dict[ModelReference, ModelMetadataLookup] = {}
    for provider, model_id in references:
        if model_metadata_provider is None:
            result[(provider, model_id)] = ModelMetadataLookup(status="unavailable")
            continue
        try:
            lookup = model_metadata_provider.resolve_model_metadata(
                provider=provider, model_id=model_id
            )
            if not isinstance(lookup, ModelMetadataLookup):
                lookup = ModelMetadataLookup.model_validate(lookup)
        except Exception:
            lookup = ModelMetadataLookup(status="unavailable")
        result[(provider, model_id)] = lookup
    return result


def compute_max_dimensionality(
    execution_graph: DiGraph, steps: List[StepMetadata]
) -> int:
    """Max over input node depths, step input / output depths and output node
    depths - 0 when there is nothing (a no-step input->output workflow still
    reports its input depth)."""
    depths: List[int] = []
    for _, node_data in execution_graph.nodes(data=True):
        compilation_output: ExecutionGraphNode = node_data[
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        if isinstance(compilation_output, StepNode):
            continue
        depths.append(len(compilation_output.data_lineage))
    for step in steps:
        depths.append(step.input_dimensionality)
        depths.append(step.output_dimensionality)
    return max(depths, default=0)
