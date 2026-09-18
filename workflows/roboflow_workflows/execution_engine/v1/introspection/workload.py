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
    Discovery,
    ModelMetadataLookup,
    ModelMetadataProvider,
    RestrictionMetadata,
    WorkOperation,
    normalize_declaration,
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
)

ROBOFLOW_MODEL_PROVIDER = "roboflow"

RESOURCES_HOOK = "discover_dependent_resources"
OPERATIONS_HOOK = "discover_work_operations"
RESTRICTIONS_HOOK = "discover_portable_restrictions"

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
    opaque_reason = (
        f"remote_dispatch_child_opaque:{step_selector}"
        if is_dispatched_inner_workflow(
            block_type=block_type, step_manifest=step_manifest
        )
        else None
    )
    return StepMetadata(
        node_id=step_selector,
        block_type=block_type,
        input_dimensionality=step_node.reference_dimensionality,
        output_dimensionality=step_node.output_dimensionality,
        accepts_batch_input=bool(step_manifest.accepts_batch_input()),
        resources=collect_declaration(
            step_manifest=step_manifest,
            hook_name=RESOURCES_HOOK,
            discovery_type=Discovery[DependentResource],
            unknown_reason=f"step_resources_unknown:{step_selector}",
            step_selector=step_selector,
            opaque_reason=opaque_reason,
        ),
        restrictions=collect_declaration(
            step_manifest=step_manifest,
            hook_name=RESTRICTIONS_HOOK,
            discovery_type=Discovery[RestrictionMetadata],
            unknown_reason=f"step_restrictions_unknown:{step_selector}",
            step_selector=step_selector,
            opaque_reason=opaque_reason,
        ),
        operations=collect_declaration(
            step_manifest=step_manifest,
            hook_name=OPERATIONS_HOOK,
            discovery_type=Discovery[WorkOperation],
            unknown_reason=f"step_operations_unknown:{step_selector}",
            step_selector=step_selector,
            opaque_reason=opaque_reason,
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
    unknown_reason: str,
    step_selector: str,
    opaque_reason: Optional[str] = None,
) -> Discovery:
    """Call one manifest hook and normalise its answer.

    `None` -> unknown (incomplete with `unknown_reason`), a list -> complete,
    a `Discovery` -> itself. A hook that raises, or returns something that does
    not validate as the expected discovery, yields an incomplete discovery with
    reason `<hook>_failed:$steps.<name>` - compilation and schema errors were
    raised earlier, a broken declaration must not hide the whole graph.
    """
    try:
        declared = getattr(step_manifest, hook_name)()
        normalised = normalize_declaration(declared, unknown_reason)
        discovery = discovery_type(
            items=list(normalised.items),
            complete=normalised.complete,
            unknown_reasons=list(normalised.unknown_reasons),
        )
    except Exception:
        discovery = discovery_type(
            items=[],
            complete=False,
            unknown_reasons=[f"{hook_name}_failed:{step_selector}"],
        )
    if opaque_reason is None:
        return discovery
    return discovery_type(
        items=list(discovery.items),
        complete=False,
        unknown_reasons=list(discovery.unknown_reasons) + [opaque_reason],
    )


def build_models_inventory(
    steps: List[StepMetadata],
    model_metadata_provider: Optional[ModelMetadataProvider],
) -> Discovery[ModelSummary]:
    """Inventory of literal model references across all steps.

    * platform models -> provider `roboflow`; third-party models keep their
      declared provider; projects are not models;
    * selector-valued references are never inventory entries - they stay in
      the per-step resources and add `unresolved_model_selector:<step>`;
    * blank (empty / whitespace-only) literal ids or providers are declared
      as-is per step but never inventoried, never sent to the metadata
      provider, and add `blank_model_identifier:<step>` - an id is never
      fabricated;
    * a step with unknown / incomplete resources propagates its reasons;
    * entries are unique per `(provider, model_id)` with sorted referring
      steps; metadata lookup outcomes never change `complete`.
    """
    used_by_steps: Dict[ModelReference, Set[str]] = defaultdict(set)
    unknown_reasons: Set[str] = set()
    for step in steps:
        if not step.resources.complete:
            unknown_reasons.update(step.resources.unknown_reasons)
        for resource in step.resources.items:
            reference = model_reference_of(resource=resource)
            if reference is None:
                continue
            if resource.metadata.requires_runtime_resolution():
                unknown_reasons.add(f"unresolved_model_selector:{step.node_id}")
                continue
            if is_blank_model_reference(reference=reference):
                unknown_reasons.add(f"blank_model_identifier:{step.node_id}")
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
            metadata=lookups[(provider, model_id)].metadata,
            metadata_status=lookups[(provider, model_id)].status,
        )
        for provider, model_id in references
    ]
    if unknown_reasons:
        return Discovery[ModelSummary](
            items=items,
            complete=False,
            unknown_reasons=sorted(unknown_reasons),
        )
    return Discovery[ModelSummary](items=items, complete=True, unknown_reasons=[])


def is_blank_model_reference(reference: ModelReference) -> bool:
    """A literal reference whose provider or model id is empty / whitespace
    (e.g. `model_id=""`, `base_url=""`) names nothing - `ModelSummary`
    rejects it and no metadata lookup could resolve it."""
    provider, model_id = reference
    return not str(provider).strip() or not str(model_id).strip()


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
