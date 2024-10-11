import itertools
from collections import defaultdict
from copy import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from networkx import DiGraph

from inference.core import logger
from inference.core.workflows.errors import (
    AssumptionError,
    BlockInterfaceError,
    ControlFlowDefinitionError,
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
    StepInputDimensionalityError,
    StepInputLineageError,
    StepOutputLineageError,
)
from inference.core.workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
    WORKFLOW_INPUT_BATCH_LINEAGE_ID,
)
from inference.core.workflows.execution_engine.entities.base import (
    InputType,
    JsonField,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    STEP_AS_SELECTED_ELEMENT,
    Kind,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    ParsedSelector,
)
from inference.core.workflows.execution_engine.introspection.selectors_parser import (
    get_step_selectors,
)
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompoundStepInputDefinition,
    DictOfStepInputDefinitions,
    DynamicStepInputDefinition,
    ExecutionGraphNode,
    InputDimensionalitySpecification,
    InputNode,
    ListOfStepInputDefinitions,
    NodeCategory,
    NodeInputCategory,
    OutputNode,
    ParameterSpecification,
    ParsedWorkflowDefinition,
    PropertyPredecessorDefinition,
    StaticStepInputDefinition,
    StepInputData,
    StepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.v1.compiler.graph_traversal import (
    traverse_graph_ensuring_parents_are_reached_first,
)
from inference.core.workflows.execution_engine.v1.compiler.reference_type_checker import (
    validate_reference_kinds,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    construct_input_selector,
    construct_output_selector,
    construct_step_selector,
    get_last_chunk_of_selector,
    get_nodes_of_specific_category,
    get_step_selector_from_its_output,
    identify_lineage,
    is_flow_control_step,
    is_input_node,
    is_input_selector,
    is_output_node,
    is_step_node,
    is_step_output_selector,
    is_step_selector,
    node_as,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

NODE_DEFINITION_KEY = "definition"
STEP_INPUT_SELECTORS_PROPERTY = "step_input_selectors"
EXCLUDED_FIELDS = {"type", "name"}


@execution_phase(
    name="execution_graph_creation",
    categories=["execution_engine_operation"],
)
def prepare_execution_graph(
    workflow_definition: ParsedWorkflowDefinition,
    profiler: Optional[WorkflowsProfiler] = None,
) -> DiGraph:
    execution_graph = construct_graph(
        workflow_definition=workflow_definition,
    )
    if not nx.is_directed_acyclic_graph(execution_graph):
        raise ExecutionGraphStructureError(
            public_message=f"Detected cycle in execution graph. This means that there is output from one "
            f"step that is connected to input of other step creating loop in the graph that would "
            f"never end if executed.",
            context="workflow_compilation | execution_graph_construction",
        )
    return denote_data_flow_in_workflow(
        execution_graph=execution_graph,
        parsed_workflow_definition=workflow_definition,
    )


def construct_graph(
    workflow_definition: ParsedWorkflowDefinition,
) -> DiGraph:
    execution_graph = nx.DiGraph()
    execution_graph = add_input_nodes_for_graph(
        inputs=workflow_definition.inputs,
        execution_graph=execution_graph,
    )
    execution_graph = add_steps_nodes_for_graph(
        steps=workflow_definition.steps,
        execution_graph=execution_graph,
    )
    execution_graph = add_output_nodes_for_graph(
        outputs=workflow_definition.outputs,
        execution_graph=execution_graph,
    )
    execution_graph = add_steps_edges(
        workflow_definition=workflow_definition,
        execution_graph=execution_graph,
    )
    return add_edges_for_outputs(
        workflow_definition=workflow_definition,
        execution_graph=execution_graph,
    )


def add_input_nodes_for_graph(
    inputs: List[InputType],
    execution_graph: DiGraph,
) -> DiGraph:
    for input_spec in inputs:
        input_selector = construct_input_selector(input_name=input_spec.name)
        data_lineage = (
            []
            if not input_spec.is_batch_oriented()
            else [WORKFLOW_INPUT_BATCH_LINEAGE_ID]
        )
        compilation_output = InputNode(
            node_category=NodeCategory.INPUT_NODE,
            name=input_spec.name,
            selector=input_selector,
            data_lineage=data_lineage,
            input_manifest=input_spec,
        )
        execution_graph.add_node(
            input_selector,
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: compilation_output,
            },
        )
    return execution_graph


def add_steps_nodes_for_graph(
    steps: List[WorkflowBlockManifest],
    execution_graph: DiGraph,
) -> DiGraph:
    for step in steps:
        step_selector = construct_step_selector(step_name=step.name)
        compilation_output = StepNode(
            node_category=NodeCategory.STEP_NODE,
            name=step.name,
            selector=step_selector,
            data_lineage=[],
            step_manifest=step,
        )
        execution_graph.add_node(
            step_selector,
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: compilation_output,
            },
        )
    return execution_graph


def add_output_nodes_for_graph(
    outputs: List[JsonField],
    execution_graph: DiGraph,
) -> DiGraph:
    for output_spec in outputs:
        output_selector = construct_output_selector(name=output_spec.name)
        compilation_output = OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=output_spec.name,
            selector=output_selector,
            data_lineage=[],
            output_manifest=output_spec,
        )
        execution_graph.add_node(
            output_selector,
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: compilation_output,
            },
        )
    return execution_graph


def add_steps_edges(
    workflow_definition: ParsedWorkflowDefinition,
    execution_graph: DiGraph,
) -> DiGraph:
    for step in workflow_definition.steps:
        step_selectors = get_step_selectors(step_manifest=step)
        execution_graph = add_edges_for_step(
            execution_graph=execution_graph,
            step_name=step.name,
            target_step_parsed_selectors=step_selectors,
        )
    return execution_graph


def add_edges_for_step(
    execution_graph: DiGraph,
    step_name: str,
    target_step_parsed_selectors: List[ParsedSelector],
) -> DiGraph:
    source_step_selector = construct_step_selector(step_name=step_name)
    for target_step_parsed_selector in target_step_parsed_selectors:
        execution_graph = add_edge_for_step(
            execution_graph=execution_graph,
            source_step_selector=source_step_selector,
            target_step_parsed_selector=target_step_parsed_selector,
        )
    return execution_graph


def add_edge_for_step(
    execution_graph: DiGraph,
    source_step_selector: str,
    target_step_parsed_selector: ParsedSelector,
) -> DiGraph:
    target_step_selector = get_step_selector_from_its_output(
        step_output_selector=target_step_parsed_selector.value
    )
    verify_edge_is_created_between_existing_nodes(
        execution_graph=execution_graph,
        start=source_step_selector,
        end=target_step_selector,
    )
    if is_step_selector(target_step_parsed_selector.value):
        return establish_flow_control_edge(
            source_step_selector=source_step_selector,
            target_step_parsed_selector=target_step_parsed_selector,
            execution_graph=execution_graph,
        )
    if is_input_selector(target_step_parsed_selector.value):
        input_node_compilation_data = node_as(
            execution_graph=execution_graph,
            node=target_step_parsed_selector.value,
            expected_type=InputNode,
        )
        actual_input_kind = input_node_compilation_data.input_manifest.kind
    else:
        other_step_compilation_data = node_as(
            execution_graph=execution_graph,
            node=target_step_selector,
            expected_type=StepNode,
        )
        actual_input_kind = get_kind_of_value_provided_in_step_output(
            step_manifest=other_step_compilation_data.step_manifest,
            step_property=get_last_chunk_of_selector(
                selector=target_step_parsed_selector.value
            ),
        )
    expected_input_kind = list(
        itertools.chain.from_iterable(
            ref.kind
            for ref in target_step_parsed_selector.definition.allowed_references
        )
    )
    error_message = (
        f"Failed to validate reference provided for step: {source_step_selector} regarding property: "
        f"{target_step_parsed_selector.definition.property_name} with value: {target_step_parsed_selector.value}. "
        f"Allowed kinds of references for this property: {list(set(e.name for e in expected_input_kind))}. "
        f"Types of output for referred property: {list(set(a.name for a in actual_input_kind))}"
    )
    validate_reference_kinds(
        expected=expected_input_kind,
        actual=actual_input_kind,
        error_message=error_message,
    )
    if execution_graph.has_edge(target_step_selector, source_step_selector):
        edge_data = execution_graph.edges[(target_step_selector, source_step_selector)]
        if STEP_INPUT_SELECTORS_PROPERTY not in edge_data:
            edge_data[STEP_INPUT_SELECTORS_PROPERTY] = []
        edge_data[STEP_INPUT_SELECTORS_PROPERTY].append(
            target_step_parsed_selector,
        )
    else:
        execution_graph.add_edge(
            target_step_selector,
            source_step_selector,
            **{STEP_INPUT_SELECTORS_PROPERTY: [target_step_parsed_selector]},
        )
    return execution_graph


def establish_flow_control_edge(
    source_step_selector: str,
    target_step_parsed_selector: ParsedSelector,
    execution_graph: DiGraph,
) -> DiGraph:
    if not step_definition_allows_flow_control_references(
        parsed_selector=target_step_parsed_selector
    ):
        raise ExecutionGraphStructureError(
            public_message=f"Detected reference to other step in manifest of step {source_step_selector}. "
            f"This is not allowed, as step manifest does not define any properties of type `StepSelector` "
            f"allowing for defining connections between steps that represent flow control in `workflows`.",
            context="workflow_compilation | execution_graph_construction",
        )
    target_step_selector = get_step_selector_from_its_output(
        step_output_selector=target_step_parsed_selector.value
    )
    source_compilation_data = node_as(
        execution_graph=execution_graph,
        node=source_step_selector,
        expected_type=StepNode,
    )
    target_compilation_data = node_as(
        execution_graph=execution_graph,
        node=target_step_selector,
        expected_type=StepNode,
    )
    nodes_categories = {
        source_compilation_data.node_category,
        target_compilation_data.node_category,
    }
    if nodes_categories != {NodeCategory.STEP_NODE}:
        raise ExecutionGraphStructureError(
            public_message=f"Attempted to create flow-control connection between workflow nodes: "
            f"{source_step_selector} and {target_step_selector}, one of which is not denoted"
            f"as step node.",
            context="workflow_compilation | execution_graph_construction",
        )
    execution_graph.add_edge(
        source_step_selector,
        target_step_selector,
    )
    if target_step_selector in source_compilation_data.child_execution_branches:
        raise ExecutionGraphStructureError(
            public_message=f"While establishing flow-control connection between source `{source_step_selector}` "
            f"and target `{target_step_selector}` it was discovered that source step defines"
            f"control flow transition into target one more than once, which is not allowed as "
            f"that situation creates ambiguity in selecting execution branch.",
            context="workflow_compilation | execution_graph_construction",
        )
    execution_branch_name = f"Branch[{source_step_selector} -> {target_step_parsed_selector.definition.property_name}]"
    source_compilation_data.child_execution_branches[target_step_selector] = (
        execution_branch_name
    )
    target_compilation_data.execution_branches_impacting_inputs.add(
        execution_branch_name
    )
    return execution_graph


def step_definition_allows_flow_control_references(
    parsed_selector: ParsedSelector,
) -> bool:
    return any(
        definition.selected_element == STEP_AS_SELECTED_ELEMENT
        for definition in parsed_selector.definition.allowed_references
    )


def get_kind_of_value_provided_in_step_output(
    step_manifest: WorkflowBlockManifest,
    step_property: str,
) -> List[Kind]:
    referred_node_outputs = step_manifest.get_actual_outputs()
    actual_kind = []
    matched_property = False
    for output in referred_node_outputs:
        if output.name != step_property:
            continue
        matched_property = True
        actual_kind.extend(output.kind)
    if not matched_property:
        raise ExecutionGraphStructureError(
            public_message=f"Found reference to non-existing property `{step_property}` of step `{step_manifest.name}`.",
            context="workflow_compilation | execution_graph_construction",
        )
    return actual_kind


def add_edges_for_step_inputs(
    execution_graph: DiGraph,
    input_selectors: Set[str],
    step_selector: str,
) -> DiGraph:
    for input_selector in input_selectors:
        if is_step_output_selector(selector_or_value=input_selector):
            input_selector = get_step_selector_from_its_output(
                step_output_selector=input_selector
            )
        verify_edge_is_created_between_existing_nodes(
            execution_graph=execution_graph,
            start=input_selector,
            end=step_selector,
        )
        execution_graph.add_edge(input_selector, step_selector)
    return execution_graph


def add_edges_for_outputs(
    workflow_definition: ParsedWorkflowDefinition,
    execution_graph: DiGraph,
) -> DiGraph:
    for output in workflow_definition.outputs:
        node_selector = output.selector
        if is_step_output_selector(selector_or_value=node_selector):
            node_selector = get_step_selector_from_its_output(
                step_output_selector=node_selector
            )
        output_name = construct_output_selector(name=output.name)
        verify_edge_is_created_between_existing_nodes(
            execution_graph=execution_graph,
            start=node_selector,
            end=output_name,
        )
        if is_step_output_selector(selector_or_value=output.selector):
            step_manifest = execution_graph.nodes[node_selector][
                NODE_COMPILATION_OUTPUT_PROPERTY
            ].step_manifest
            step_outputs = step_manifest.get_actual_outputs()
            verify_output_selector_points_to_valid_output(
                output_selector=output.selector,
                step_outputs=step_outputs,
            )
        execution_graph.add_edge(node_selector, output_name)
    return execution_graph


def verify_edge_is_created_between_existing_nodes(
    execution_graph: DiGraph,
    start: str,
    end: str,
) -> None:
    if not execution_graph.has_node(start):
        raise InvalidReferenceTargetError(
            public_message=f"Graph definition contains selector {start} that points to not defined element.",
            context="workflow_compilation | execution_graph_construction",
        )
    if not execution_graph.has_node(end):
        raise InvalidReferenceTargetError(
            public_message=f"Graph definition contains selector {end} that points to not defined element.",
            context="workflow_compilation | execution_graph_construction",
        )


def verify_output_selector_points_to_valid_output(
    output_selector: str,
    step_outputs: List[OutputDefinition],
) -> None:
    selected_output_name = get_last_chunk_of_selector(selector=output_selector)
    if selected_output_name == "*":
        return None
    defined_output_names = {output.name for output in step_outputs}
    if selected_output_name not in defined_output_names:
        raise InvalidReferenceTargetError(
            public_message=f"Graph definition contains selector {output_selector} that points to output of step "
            f"that is not defined in workflow block used to create step.",
            context="workflow_compilation | execution_graph_construction",
        )


def denote_data_flow_in_workflow(
    execution_graph: DiGraph,
    parsed_workflow_definition: ParsedWorkflowDefinition,
) -> nx.DiGraph:
    block_manifest_by_step_name = {
        step.name: step for step in parsed_workflow_definition.steps
    }
    super_input_node = "<super-input>"
    execution_graph = add_super_input_node_in_execution_graph(
        execution_graph=execution_graph,
        super_input_node=super_input_node,
    )
    execution_graph.nodes[super_input_node][NODE_COMPILATION_OUTPUT_PROPERTY] = (
        InputNode(
            node_category=NodeCategory.INPUT_NODE,
            name=super_input_node,
            selector=f"$inputs.{super_input_node}",
            data_lineage=[],
            input_manifest=None,  # this is expected never to be reached
        )
    )
    for node in traverse_graph_ensuring_parents_are_reached_first(
        graph=execution_graph,
        start_node=super_input_node,
    ):
        execution_graph = denote_data_flow_for_node(
            execution_graph=execution_graph,
            node=node,
            block_manifest_by_step_name=block_manifest_by_step_name,
        )
    execution_graph.remove_node(super_input_node)
    return execution_graph


def denote_data_flow_for_node(
    execution_graph: DiGraph,
    node: str,
    block_manifest_by_step_name: Dict[str, WorkflowBlockManifest],
) -> DiGraph:
    if is_input_node(execution_graph=execution_graph, node=node):
        # everything already set there, in the previous stage of compilation
        return execution_graph
    if is_step_node(execution_graph=execution_graph, node=node):
        step_name = get_last_chunk_of_selector(selector=node)
        if step_name not in block_manifest_by_step_name:
            raise AssumptionError(
                public_message=f"Workflow Compiler expected manifest for the step: {step_name} to be registered "
                f"but this condition is not met. This is most likely the bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full "
                f"context of the problem - including workflow definition you use.",
                context="workflow_compilation | execution_graph_construction | retrieving_predecessors_for_output",
            )
        manifest = block_manifest_by_step_name[step_name]
        return denote_data_flow_for_step(
            execution_graph=execution_graph,
            node=node,
            manifest=manifest,
        )
    if is_output_node(execution_graph=execution_graph, node=node):
        # output is allowed to have exactly one predecessor
        output_predecessors = list(execution_graph.predecessors(node))
        if len(output_predecessors) != 1:
            raise AssumptionError(
                public_message=f"Workflow Compiler expected each output in compiled graph to have one predecessor, "
                f"but this condition is not met for node {node}. This is most likely the bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full "
                f"context of the problem - including workflow definition you use.",
                context="workflow_compilation | execution_graph_construction | retrieving_predecessors_for_output",
            )
        predecessor_node = output_predecessors[0]
        predecessor_node_data = node_as(
            execution_graph=execution_graph,
            node=predecessor_node,
            expected_type=ExecutionGraphNode,
        )
        predecessor_node_lineage = predecessor_node_data.data_lineage
        output_node_data = node_as(
            execution_graph=execution_graph,
            node=node,
            expected_type=OutputNode,
        )
        output_node_data.data_lineage = predecessor_node_lineage
        return execution_graph
    raise AssumptionError(
        f"Workflow Compiler encountered node: {node} which cannot be classified as known node type. "
        f"This is most likely the bug. Contact Roboflow team through github issues "
        f"(https://github.com/roboflow/inference/issues) providing full "
        f"context of the problem - including workflow definition you use.",
        context="workflow_compilation | execution_graph_construction | denoting_data_flow_for_step",
    )


def denote_data_flow_for_step(
    execution_graph: DiGraph,
    node: str,
    manifest: WorkflowBlockManifest,
) -> DiGraph:
    all_control_flow_predecessors, all_non_control_flow_predecessors = (
        separate_flow_control_predecessors_from_data_providers(
            execution_graph=execution_graph, node=node
        )
    )
    input_data = build_input_data_for_step(
        manifest=manifest,
        step_node=node,
        data_providing_predecessors=all_non_control_flow_predecessors,
        execution_graph=execution_graph,
    )
    step_name = get_last_chunk_of_selector(node)
    step_node_data = node_as(
        execution_graph=execution_graph,
        node=node,
        expected_type=StepNode,
    )
    inputs_dimensionalities = get_inputs_dimensionalities(
        step_name=step_name,
        input_data=input_data,
    )
    logger.debug(
        f"For step: {node}, detected the following input dimensionalities: {inputs_dimensionalities}"
    )
    parameters_with_batch_inputs = grab_parameters_defining_batch_inputs(
        inputs_dimensionalities=inputs_dimensionalities,
    )
    dimensionality_reference_property = manifest.get_dimensionality_reference_property()
    input_dimensionality_offsets = manifest.get_input_dimensionality_offsets()
    output_dimensionality_offset = manifest.get_output_dimensionality_offset()
    verify_step_input_dimensionality_offsets(
        step_name=step_name,
        input_dimensionality_offsets=input_dimensionality_offsets,
    )
    verify_output_offset(
        step_name=step_name,
        parameters_with_batch_inputs=parameters_with_batch_inputs,
        dimensionality_reference_property=dimensionality_reference_property,
        input_dimensionality_offsets=input_dimensionality_offsets,
        output_dimensionality_offset=output_dimensionality_offset,
    )
    verify_input_data_dimensionality(
        step_name=step_name,
        dimensionality_reference_property=dimensionality_reference_property,
        inputs_dimensionalities=inputs_dimensionalities,
        dimensionality_offstes=input_dimensionality_offsets,
    )
    all_lineages = get_input_data_lineage(step_name=step_name, input_data=input_data)
    verify_compatibility_of_input_data_lineage_with_control_flow_lineage(
        step_name=step_name,
        inputs_lineage=all_lineages,
        flow_control_steps_selectors=all_control_flow_predecessors,
        execution_graph=execution_graph,
    )
    step_node_data.input_data = input_data
    step_node_data.dimensionality_reference_property = dimensionality_reference_property
    step_node_data.batch_oriented_parameters = parameters_with_batch_inputs
    step_node_data.step_execution_dimensionality = (
        establish_step_execution_dimensionality(
            inputs_dimensionalities=inputs_dimensionalities,
            output_dimensionality_offset=output_dimensionality_offset,
        )
    )
    if not parameters_with_batch_inputs:
        data_lineage = []
    else:
        data_lineage = establish_batch_oriented_step_lineage(
            step_selector=node,
            all_lineages=all_lineages,
            input_data=input_data,
            dimensionality_reference_property=dimensionality_reference_property,
            output_dimensionality_offset=output_dimensionality_offset,
        )
    step_node_data.data_lineage = data_lineage
    return execution_graph


def separate_flow_control_predecessors_from_data_providers(
    execution_graph: DiGraph,
    node: str,
) -> Tuple[List[str], List[str]]:
    all_predecessors = list(execution_graph.predecessors(node))
    all_control_flow_predecessors = [
        predecessor
        for predecessor in all_predecessors
        if is_flow_control_step(execution_graph=execution_graph, node=predecessor)
    ]
    all_non_control_flow_predecessors = [
        predecessor
        for predecessor in all_predecessors
        if not is_flow_control_step(execution_graph=execution_graph, node=predecessor)
    ]
    return all_control_flow_predecessors, all_non_control_flow_predecessors


def build_input_data_for_step(
    manifest: WorkflowBlockManifest,
    step_node: str,
    data_providing_predecessors: List[str],
    execution_graph: DiGraph,
) -> StepInputData:
    predecessors_by_property_name = get_predecessors_for_step_manifest_properties(
        execution_graph=execution_graph,
        step_node=step_node,
        data_providing_predecessors=data_providing_predecessors,
    )
    manifest_fields_values = get_manifest_fields_values(step_manifest=manifest)
    result = {}
    for name, value in manifest_fields_values.items():
        result[name] = build_input_property(
            step_node=step_node,
            manifest_property_name=name,
            manifest_property_value=value,
            predecessors_by_property_name=predecessors_by_property_name,
            execution_graph=execution_graph,
        )
    return result


def get_predecessors_for_step_manifest_properties(
    execution_graph: DiGraph, step_node: str, data_providing_predecessors: List[str]
) -> Dict[str, List[PropertyPredecessorDefinition]]:
    predecessors_by_property_name = defaultdict(list)
    for predecessor in data_providing_predecessors:
        edge_data = execution_graph.edges[(predecessor, step_node)]
        # STEP_INPUT_SELECTORS_PROPERTY is optional, as special <super-input> node may be directly
        # connected to step
        selectors_associated_to_edge: List[ParsedSelector] = edge_data.get(
            STEP_INPUT_SELECTORS_PROPERTY, []
        )
        for selector_associated_to_edge in selectors_associated_to_edge:
            definition = PropertyPredecessorDefinition(
                predecessor_selector=predecessor,
                parsed_selector=selector_associated_to_edge,
            )
            predecessors_by_property_name[
                selector_associated_to_edge.definition.property_name
            ].append(definition)
    return predecessors_by_property_name


def build_input_property(
    step_node: str,
    manifest_property_name: str,
    manifest_property_value: Any,
    predecessors_by_property_name: Dict[str, List[PropertyPredecessorDefinition]],
    execution_graph: DiGraph,
) -> Union[StepInputDefinition, CompoundStepInputDefinition]:
    if manifest_property_name not in predecessors_by_property_name:
        return StaticStepInputDefinition(
            parameter_specification=ParameterSpecification(
                parameter_name=manifest_property_name,
            ),
            category=NodeInputCategory.STATIC_VALUE,
            value=manifest_property_value,
        )
    if isinstance(manifest_property_value, dict):
        return build_nested_dictionary_for_input_property(
            step_node=step_node,
            manifest_property_name=manifest_property_name,
            manifest_property_value=manifest_property_value,
            predecessors_definitions=predecessors_by_property_name[
                manifest_property_name
            ],
            execution_graph=execution_graph,
        )
    if isinstance(manifest_property_value, list):
        return build_nested_list_for_input_property(
            step_node=step_node,
            manifest_property_name=manifest_property_name,
            manifest_property_value=manifest_property_value,
            predecessors_definitions=predecessors_by_property_name[
                manifest_property_name
            ],
            execution_graph=execution_graph,
        )
    matching_predecessors_data = predecessors_by_property_name[manifest_property_name]
    if len(matching_predecessors_data) != 1:
        raise AssumptionError(
            public_message=f"Workflow Compiler deduced that property `{manifest_property_name}` of step `{step_node}` "
            f"should be fed with data coming from exactly one execution graph node, but found "
            f"{len(matching_predecessors_data)} data sources. This is most likely the bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
        )
    predecessor_selector, predecessor_parsed_selector = (
        matching_predecessors_data[0].predecessor_selector,
        matching_predecessors_data[0].parsed_selector,
    )
    if is_input_node(execution_graph=execution_graph, node=predecessor_selector):
        predecessor_node_data = node_as(
            execution_graph=execution_graph,
            node=matching_predecessors_data[0].predecessor_selector,
            expected_type=InputNode,
        )
        category = (
            NodeInputCategory.BATCH_INPUT_PARAMETER
            if predecessor_node_data.is_batch_oriented()
            else NodeInputCategory.NON_BATCH_INPUT_PARAMETER
        )
        return DynamicStepInputDefinition(
            parameter_specification=ParameterSpecification(
                parameter_name=manifest_property_name,
            ),
            category=category,
            data_lineage=predecessor_node_data.data_lineage,
            selector=predecessor_node_data.selector,
        )
    if is_step_node(execution_graph=execution_graph, node=predecessor_selector):
        predecessor_node_data = node_as(
            execution_graph=execution_graph,
            node=matching_predecessors_data[0].predecessor_selector,
            expected_type=StepNode,
        )
        category = (
            NodeInputCategory.BATCH_STEP_OUTPUT
            if predecessor_node_data.output_dimensionality > 0
            else NodeInputCategory.NON_BATCH_STEP_OUTPUT
        )
        return DynamicStepInputDefinition(
            parameter_specification=ParameterSpecification(
                parameter_name=manifest_property_name,
            ),
            category=category,
            data_lineage=predecessor_node_data.data_lineage,
            selector=predecessor_parsed_selector.value,
        )
    raise AssumptionError(
        public_message=f"Workflow Compiler for property `{manifest_property_name}` of step `{step_node}` "
        f"found data providing predecessor `{predecessor_selector}` which has unsupported step type."
        f"This is most likely the bug. "
        f"Contact Roboflow team through github issues "
        f"(https://github.com/roboflow/inference/issues) providing full "
        f"context of the problem - including workflow definition you use.",
        context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
    )


def build_nested_dictionary_for_input_property(
    step_node: str,
    manifest_property_name: str,
    manifest_property_value: dict,
    predecessors_definitions: List[PropertyPredecessorDefinition],
    execution_graph: DiGraph,
) -> DictOfStepInputDefinitions:
    nested_property_name2data: Dict[Optional[str], PropertyPredecessorDefinition] = {
        predecessor_definition.parsed_selector.key: predecessor_definition
        for predecessor_definition in predecessors_definitions
    }
    result = {}
    for nested_dict_key, nested_dict_value in manifest_property_value.items():
        if nested_dict_key not in nested_property_name2data:
            result[nested_dict_key] = StaticStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name=manifest_property_name,
                    nested_element_key=nested_dict_key,
                ),
                category=NodeInputCategory.STATIC_VALUE,
                value=nested_dict_value,
            )
            continue
        referred_node_selector = nested_property_name2data[
            nested_dict_key
        ].predecessor_selector
        referred_node_parsed_selector = nested_property_name2data[
            nested_dict_key
        ].parsed_selector
        if is_input_node(execution_graph=execution_graph, node=referred_node_selector):
            predecessor_node_data = node_as(
                execution_graph=execution_graph,
                node=referred_node_selector,
                expected_type=InputNode,
            )
            category = (
                NodeInputCategory.BATCH_INPUT_PARAMETER
                if predecessor_node_data.is_batch_oriented()
                else NodeInputCategory.NON_BATCH_INPUT_PARAMETER
            )
            result[nested_dict_key] = DynamicStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name=manifest_property_name,
                    nested_element_key=nested_dict_key,
                ),
                category=category,
                data_lineage=predecessor_node_data.data_lineage,
                selector=predecessor_node_data.selector,
            )
            continue
        if is_step_node(execution_graph=execution_graph, node=referred_node_selector):
            predecessor_node_data = node_as(
                execution_graph=execution_graph,
                node=referred_node_selector,
                expected_type=StepNode,
            )
            category = (
                NodeInputCategory.BATCH_STEP_OUTPUT
                if predecessor_node_data.output_dimensionality > 0
                else NodeInputCategory.NON_BATCH_STEP_OUTPUT
            )
            result[nested_dict_key] = DynamicStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name=manifest_property_name,
                    nested_element_key=nested_dict_key,
                ),
                category=category,
                data_lineage=predecessor_node_data.data_lineage,
                selector=referred_node_parsed_selector.value,
            )
            continue
        raise AssumptionError(
            public_message=f"Workflow Compiler for property `{manifest_property_name}` of step `{step_node}` "
            f"found data providing predecessor `{referred_node_selector}` which has "
            f"unsupported step type (key: {nested_dict_key} of nested data dictionary). "
            f"This is most likely the bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
        )
    return DictOfStepInputDefinitions(
        name=manifest_property_name,
        nested_definitions=result,
    )


def get_manifest_fields_values(step_manifest: WorkflowBlockManifest) -> Dict[str, Any]:
    result = {}
    for field in step_manifest.model_fields:
        if field in EXCLUDED_FIELDS:
            continue
        result[field] = getattr(step_manifest, field)
    return result


def build_nested_list_for_input_property(
    step_node: str,
    manifest_property_name: str,
    manifest_property_value: list,
    predecessors_definitions: List[PropertyPredecessorDefinition],
    execution_graph: DiGraph,
) -> ListOfStepInputDefinitions:
    nested_index2data: Dict[Optional[int], PropertyPredecessorDefinition] = {
        e.parsed_selector.index: e for e in predecessors_definitions
    }
    result = []
    for index, element in enumerate(manifest_property_value):
        if index not in nested_index2data:
            result.append(
                StaticStepInputDefinition(
                    parameter_specification=ParameterSpecification(
                        parameter_name=manifest_property_name,
                        nested_element_index=index,
                    ),
                    category=NodeInputCategory.STATIC_VALUE,
                    value=element,
                )
            )
            continue
        referred_node_selector = nested_index2data[index].predecessor_selector
        referred_node_parsed_selector = nested_index2data[index].parsed_selector
        if is_input_node(execution_graph=execution_graph, node=referred_node_selector):
            predecessor_node_data = node_as(
                execution_graph=execution_graph,
                node=referred_node_selector,
                expected_type=InputNode,
            )
            category = (
                NodeInputCategory.BATCH_INPUT_PARAMETER
                if predecessor_node_data.is_batch_oriented()
                else NodeInputCategory.NON_BATCH_INPUT_PARAMETER
            )
            result.append(
                DynamicStepInputDefinition(
                    parameter_specification=ParameterSpecification(
                        parameter_name=manifest_property_name,
                        nested_element_index=index,
                    ),
                    category=category,
                    data_lineage=predecessor_node_data.data_lineage,
                    selector=predecessor_node_data.selector,
                )
            )
            continue
        if is_step_node(execution_graph=execution_graph, node=referred_node_selector):
            predecessor_node_data = node_as(
                execution_graph=execution_graph,
                node=referred_node_selector,
                expected_type=StepNode,
            )
            category = (
                NodeInputCategory.BATCH_STEP_OUTPUT
                if predecessor_node_data.output_dimensionality > 0
                else NodeInputCategory.NON_BATCH_STEP_OUTPUT
            )
            result.append(
                DynamicStepInputDefinition(
                    parameter_specification=ParameterSpecification(
                        parameter_name=manifest_property_name,
                        nested_element_index=index,
                    ),
                    category=category,
                    data_lineage=predecessor_node_data.data_lineage,
                    selector=referred_node_parsed_selector.value,
                )
            )
            continue
        raise AssumptionError(
            public_message=f"Workflow Compiler for property `{manifest_property_name}` of step `{step_node}` "
            f"found data providing predecessor `{referred_node_selector}` which has "
            f"unsupported step type (index: {index} of nested data list). "
            f"This is most likely the bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
        )
    return ListOfStepInputDefinitions(
        name=manifest_property_name,
        nested_definitions=result,
    )


def verify_step_input_dimensionality_offsets(
    step_name: str,
    input_dimensionality_offsets: Dict[str, int],
) -> None:
    min_offset, max_offset = None, None
    for offset in input_dimensionality_offsets.values():
        if min_offset is None or offset < min_offset:
            min_offset = offset
        if max_offset is None or offset > max_offset:
            max_offset = offset
    if min_offset is None or max_offset is None:
        return None
    if min_offset < 0 or max_offset < 0:
        raise BlockInterfaceError(
            public_message=f"Offsets could not be negative, but block defining step: {step_name} defines that values.",
            context="workflow_compilation | execution_graph_construction | verification_of_input_offset_definitions",
        )
    if abs(max_offset - min_offset) > 1:
        raise BlockInterfaceError(
            public_message="Offsets of input parameters could not differ more than 1, but block defining step {step_name} "
            f"violates that rule.",
            context="workflow_compilation | execution_graph_construction | verification_of_input_offset_definitions",
        )


def verify_output_offset(
    step_name: str,
    parameters_with_batch_inputs: Set[str],
    input_dimensionality_offsets: Dict[str, int],
    dimensionality_reference_property: Optional[str],
    output_dimensionality_offset: int,
) -> None:
    if not parameters_with_batch_inputs and output_dimensionality_offset != 0:
        raise BlockInterfaceError(
            public_message=f"Block defining step {step_name} defines dimensionality offset different "
            f"than zero while taking only non-batch parameters, which is not allowed.",
            context="workflow_compilation | execution_graph_construction | verification_of_output_offset",
        )
    if (
        dimensionality_reference_property is not None
        and dimensionality_reference_property not in parameters_with_batch_inputs
    ):
        raise BlockInterfaceError(
            public_message=f"Block defining step {step_name} defines dimensionality reference property "
            f"which is not in scope of parameters bring batch-oriented input, which makes it impossible "
            f"to use as reference for output dimensionality.",
            context="workflow_compilation | execution_graph_construction | verification_of_output_offset",
        )
    if output_dimensionality_offset not in {-1, 0, 1}:
        raise BlockInterfaceError(
            public_message=f"Block defining step {step_name} defines output dimensionality offset "
            f"being {output_dimensionality_offset}, whereas it is only possible for that offset being "
            f"in set [-1, 0, 1].",
            context="workflow_compilation | execution_graph_construction | verification_of_output_offset",
        )
    different_offsets = {o for o in input_dimensionality_offsets.values()}
    if len(parameters_with_batch_inputs) != len(input_dimensionality_offsets):
        # assumption of default offset being zero - and this one does not need to be manifested
        # by block
        different_offsets.add(0)
    if 0 not in different_offsets and parameters_with_batch_inputs:
        raise BlockInterfaceError(
            public_message=f"Block defining step {step_name} explicitly defines input dimensionalities offsets"
            f"with {input_dimensionality_offsets}, but the definition lack 0-level input, which is "
            f"not allowed, as in this scenario offsets could be adjusted to include 0",
            context="workflow_compilation | execution_graph_construction | verification_of_output_offset",
        )
    if len(different_offsets) > 1 and dimensionality_reference_property is None:
        raise BlockInterfaceError(
            public_message=f"Block defining step {step_name} explicitly defines input dimensionality "
            f"offsets {input_dimensionality_offsets}. In this scenario it is required to provide dimensionality "
            f"reference property.",
            context="workflow_compilation | execution_graph_construction | verification_of_output_offset",
        )
    if len(different_offsets) > 1 and output_dimensionality_offset != 0:
        raise BlockInterfaceError(
            public_message=f"Block defining step {step_name} explicitly defines input dimensionality "
            f"offsets {input_dimensionality_offsets} and output dimensionality offset {output_dimensionality_offset} "
            f"where the latter is not 0, but for inputs differing with dimensionality it is only possible to keep "
            f"output dimensionality the same and point reference parameter.",
            context="workflow_compilation | execution_graph_construction | verification_of_output_offset",
        )


def verify_input_data_dimensionality(
    step_name: str,
    dimensionality_reference_property: Optional[str],
    inputs_dimensionalities: Dict[str, Set[int]],
    dimensionality_offstes: Dict[str, int],
) -> None:
    parameter_name2dimensionality_specification = (
        grab_input_data_dimensionality_specifications(
            step_name=step_name,
            inputs_dimensionalities=inputs_dimensionalities,
            dimensionality_offstes=dimensionality_offstes,
        )
    )
    if not parameter_name2dimensionality_specification:
        # nothing to check if no batch-oriented inputs
        return None
    different_dimensionalities_of_parameters = {
        specification.actual_dimensionality
        for specification in parameter_name2dimensionality_specification.values()
    }
    if dimensionality_reference_property is None:
        if len(different_dimensionalities_of_parameters) > 1:
            parameter2dimensionality = {
                k: e.actual_dimensionality
                for k, e in parameter_name2dimensionality_specification.items()
            }
            raise StepInputDimensionalityError(
                public_message=f"Block defining step {step_name} does not define dimensionality reference property, "
                f"which means that all batch-oriented parameters must be at the same dimensionality level, "
                f"but detected the following dimensionalities for parameters {parameter2dimensionality}",
                context="workflow_compilation | execution_graph_construction | denoting_step_inputs_dimensionality",
            )
        return None
    reference_specification = parameter_name2dimensionality_specification[
        dimensionality_reference_property
    ]
    expected_dimensionalities = {
        property_name: (e.expected_offset - reference_specification.expected_offset)
        + reference_specification.actual_dimensionality
        for property_name, e in parameter_name2dimensionality_specification.items()
    }
    if any(dim <= 0 for dim in expected_dimensionalities.values()):
        raise StepInputDimensionalityError(
            public_message=f"Given the definition of block defining step {step_name} and data provided, "
            f"the block would expect batch input dimensionality to be 0 or below, which is invalid.",
            context="workflow_compilation | execution_graph_construction | denoting_step_inputs_dimensionality",
        )
    for property_name, expected_dimensionality in expected_dimensionalities.items():
        actual_dimensionality = parameter_name2dimensionality_specification[
            property_name
        ].actual_dimensionality
        if actual_dimensionality != expected_dimensionality:
            raise StepInputDimensionalityError(
                public_message=f"Data fed into step `{step_name}` property `{property_name}` has "
                f"actual dimensionality {actual_dimensionality}, "
                f"when expected was {expected_dimensionality}",
                context="workflow_compilation | execution_graph_construction | denoting_step_inputs_dimensionality",
            )
    return None


def grab_input_data_dimensionality_specifications(
    step_name: str,
    inputs_dimensionalities: Dict[str, Set[int]],
    dimensionality_offstes: Dict[str, int],
) -> Dict[str, InputDimensionalitySpecification]:
    result = {}
    for parameter_name, dimensionality in inputs_dimensionalities.items():
        parameter_offset = dimensionality_offstes.get(parameter_name, 0)
        non_zero_dimensionalities_for_parameter = {d for d in dimensionality if d > 0}
        if len(non_zero_dimensionalities_for_parameter) > 1:
            raise AssumptionError(
                public_message=f"Workflow Compiler for step: `{step_name}` and parameter: {parameter_name}"
                f"found multiple different values of actual input dimensionalities: "
                f"`{non_zero_dimensionalities_for_parameter}` which should be detected and addresses "
                f"at earlier stages of compilation."
                f"This is most likely the bug. Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full "
                f"context of the problem - including workflow definition you use.",
                context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
            )
        if non_zero_dimensionalities_for_parameter:
            actual_dimensionality = next(iter(non_zero_dimensionalities_for_parameter))
            result[parameter_name] = InputDimensionalitySpecification(
                actual_dimensionality=actual_dimensionality,
                expected_offset=parameter_offset,
            )
    return result


def verify_compatibility_of_input_data_lineage_with_control_flow_lineage(
    step_name: str,
    inputs_lineage: List[List[str]],
    flow_control_steps_selectors: List[str],
    execution_graph: DiGraph,
) -> None:
    already_spotted_input_lineages = set()
    lineage_id2control_flow_steps = defaultdict(list)
    batch_oriented_control_flow_lineages = []
    for flow_control_steps_selector in flow_control_steps_selectors:
        flow_control_step_data = node_as(
            execution_graph=execution_graph,
            node=flow_control_steps_selector,
            expected_type=StepNode,
        )
        lineage = flow_control_step_data.data_lineage
        lineage_id = identify_lineage(lineage=lineage)
        if lineage_id not in already_spotted_input_lineages and lineage:
            already_spotted_input_lineages.add(lineage_id)
            batch_oriented_control_flow_lineages.append(lineage)
        lineage_id2control_flow_steps[lineage_id].append(flow_control_steps_selector)
    all_input_lineage_prefixes = get_all_batch_lineage_prefixes(lineages=inputs_lineage)
    all_input_lineage_prefixes_hashes = {
        identify_lineage(lineage=lineage) for lineage in all_input_lineage_prefixes
    }
    for control_flow_lineage in batch_oriented_control_flow_lineages:
        control_flow_lineage_id = identify_lineage(lineage=control_flow_lineage)
        if control_flow_lineage_id not in all_input_lineage_prefixes_hashes:
            problematic_flow_control_steps = lineage_id2control_flow_steps[
                control_flow_lineage_id
            ]
            raise ControlFlowDefinitionError(
                public_message=f"Step {step_name} execution is impacted by control flow outcome of the following "
                f"steps {problematic_flow_control_steps} which make decision based on data that is "
                f"not compatible with data fed to the step {step_name} - which would cause the step "
                f"to never execute. This behaviour is invalid and prevented upfront by Workflows compiler.",
                context="workflow_compilation | execution_graph_construction | verification_of_flow_control_lineage",
            )


def get_all_batch_lineage_prefixes(lineages: List[List[str]]) -> List[List[str]]:
    result = []
    already_spotted = set()
    for lineage in lineages:
        lineage_prefixes = get_batch_lineage_prefixes(lineage=lineage)
        for lineage_prefix in lineage_prefixes:
            lineage_prefix_id = identify_lineage(lineage=lineage_prefix)
            if lineage_prefix_id not in already_spotted:
                already_spotted.add(lineage_prefix_id)
                result.append(lineage_prefix)
    return result


def get_batch_lineage_prefixes(lineage: List[str]) -> List[List[str]]:
    if not lineage:
        return []
    result = []
    prefix = []
    for i in range(len(lineage)):
        prefix.append(lineage[i])
        result.append(copy(prefix))
    return result


def get_inputs_dimensionalities(
    step_name: str, input_data: StepInputData
) -> Dict[str, Set[int]]:
    result = defaultdict(set)
    dimensionalities_spotted = set()
    for property_name, input_definition in input_data.items():
        if input_definition.is_compound_input():
            result[property_name] = get_compound_input_dimensionality(
                step_name=step_name,
                property_name=property_name,
                input_definition=input_definition,
            )
        else:
            result[property_name] = {input_definition.get_dimensionality()}
        dimensionalities_spotted.update(result[property_name])
    non_zero_dimensionalities_spotted = {d for d in dimensionalities_spotted if d != 0}
    if len(non_zero_dimensionalities_spotted) > 0:
        min_dim, max_dim = min(non_zero_dimensionalities_spotted), max(
            non_zero_dimensionalities_spotted
        )
        if abs(max_dim - min_dim) > 1:
            raise StepInputDimensionalityError(
                public_message=f"For step {step_name} attempted to plug input data differing in dimensionality more than 1",
                context="workflow_compilation | execution_graph_construction | collecting_step_input_data",
            )
    return result


def get_compound_input_dimensionality(
    step_name: str,
    property_name: str,
    input_definition: CompoundStepInputDefinition,
) -> Set[int]:
    dimensionalities_spotted = set()
    for definition in input_definition.iterate_through_definitions():
        dimensionalities_spotted.add(definition.get_dimensionality())
    non_zero_dimensionalities = {e for e in dimensionalities_spotted if e != 0}
    if len(non_zero_dimensionalities) > 1:
        raise StepInputDimensionalityError(
            public_message=f"While evaluating compound property {property_name} of step {step_name}, "
            f"detected multiple inputs of differing batch dimensionalities: {non_zero_dimensionalities}",
            context="workflow_compilation | execution_graph_construction | collecting_step_input_data",
        )
    return dimensionalities_spotted


def grab_parameters_defining_batch_inputs(
    inputs_dimensionalities: Dict[str, Set[int]],
) -> Set[str]:
    result = set()
    for paremeter, dimensionalities in inputs_dimensionalities.items():
        batch_dimensionalities = {d for d in dimensionalities if d > 0}
        if len(batch_dimensionalities):
            result.add(paremeter)
    return result


def get_input_data_lineage(
    step_name: str,
    input_data: StepInputData,
) -> List[List[str]]:
    lineage_deduplication_set = set()
    lineages = []
    for property_name, input_definition in input_data.items():
        new_lineages_detected_within_property_data = get_lineage_for_input_property(
            step_name=step_name,
            property_name=property_name,
            input_definition=input_definition,
            lineage_deduplication_set=lineage_deduplication_set,
        )
        lineages.extend(new_lineages_detected_within_property_data)
    if not lineages:
        return lineages
    verify_lineages(step_name=step_name, detected_lineages=lineages)
    return lineages


def get_lineage_for_input_property(
    step_name: str,
    property_name: str,
    input_definition: Union[StepInputDefinition, CompoundStepInputDefinition],
    lineage_deduplication_set: Set[int],
) -> List[List[str]]:
    if input_definition.is_compound_input():
        return get_lineage_from_compound_input(
            step_name=step_name,
            property_name=property_name,
            input_definition=input_definition,
            lineage_deduplication_set=lineage_deduplication_set,
        )
    lineages = []
    if input_definition.is_batch_oriented():
        lineage = input_definition.data_lineage
        lineage_id = identify_lineage(lineage=lineage)
        if lineage_id not in lineage_deduplication_set:
            lineage_deduplication_set.add(lineage_id)
            lineages.append(lineage)
    return lineages


def get_lineage_from_compound_input(
    step_name: str,
    property_name: str,
    input_definition: Union[StepInputDefinition, CompoundStepInputDefinition],
    lineage_deduplication_set: Set[int],
) -> List[List[str]]:
    lineages = []
    for nested_element in input_definition.iterate_through_definitions():
        if nested_element.is_batch_oriented():
            lineage = nested_element.data_lineage
            lineage_id = identify_lineage(lineage=lineage)
            if lineage_id not in lineage_deduplication_set:
                lineage_deduplication_set.add(lineage_id)
                lineages.append(lineage)
    if len(lineages) > 1:
        raise StepInputLineageError(
            public_message=f"Input data provided for step: `{step_name}` comes with multiple different lineages "
            f"{lineages} for property `{property_name}` accepting "
            f"multiple selectors, which is not allowed.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs_lineage",
        )
    return lineages


def verify_lineages(step_name: str, detected_lineages: List[List[str]]) -> None:
    lineages_by_length = defaultdict(list)
    for lineage in detected_lineages:
        lineages_by_length[len(lineage)].append(lineage)
    if len(lineages_by_length) > 2:
        raise StepInputLineageError(
            public_message=f"Input data provided for step: `{step_name}` comes with lineages at more than two "
            f"dimensionality levels, which should not be possible.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs_lineage",
        )
    lineage_lengths = sorted(lineages_by_length.keys())
    for lineage_length in lineage_lengths:
        if len(lineages_by_length[lineage_length]) > 1:
            raise StepInputLineageError(
                public_message=f"Among step `{step_name}` inputs found different lineages at the same  "
                f"dimensionality levels dimensionality.",
                context="workflow_compilation | execution_graph_construction | collecting_step_inputs_lineage",
            )
    if len(lineages_by_length[lineage_lengths[-1]]) > 1 and len(lineage_lengths) < 2:
        raise StepInputLineageError(
            public_message=f"If lineage differ at the last level, then Execution Engine requires at least one "
            f"higher-dimension  input that will come with common lineage, but this is not the "
            f"case for step {step_name}.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs_lineage",
        )
    if len(lineage_lengths) == 2:
        reference_lineage = lineages_by_length[lineage_lengths[0]][0]
        if reference_lineage != lineages_by_length[lineage_lengths[1]][0][:-1]:
            raise StepInputLineageError(
                public_message=f"Step `{step_name}` inputs does not share common lineage. Differing element: "
                f"{lineages_by_length[lineage_lengths[1]][0]}, "
                f"reference lineage prefix: {reference_lineage}",
                context="workflow_compilation | execution_graph_construction | collecting_step_inputs_lineage",
            )


def establish_step_execution_dimensionality(
    inputs_dimensionalities: Dict[str, Set[int]],
    output_dimensionality_offset: int,
) -> int:
    step_execution_dimensionality = 0
    non_zero_dimensionalities = {
        dimensionality
        for dimensionalities in inputs_dimensionalities.values()
        for dimensionality in dimensionalities
        if dimensionality > 0
    }
    if len(non_zero_dimensionalities) > 0:
        step_execution_dimensionality = min(non_zero_dimensionalities)
        if output_dimensionality_offset < 0:
            step_execution_dimensionality -= 1
    return step_execution_dimensionality


def establish_batch_oriented_step_lineage(
    step_selector: str,
    all_lineages: List[List[str]],
    input_data: StepInputData,
    dimensionality_reference_property: Optional[str],
    output_dimensionality_offset: int,
) -> List[str]:
    reference_lineage = get_reference_lineage(
        step_selector=step_selector,
        all_lineages=all_lineages,
        input_data=input_data,
        dimensionality_reference_property=dimensionality_reference_property,
    )
    if output_dimensionality_offset < 0:
        result_dimensionality = reference_lineage[:output_dimensionality_offset]
        if len(result_dimensionality) == 0:
            raise StepOutputLineageError(
                public_message=f"Step {step_selector} is to decrease dimensionality, but it is not possible if "
                f"input dimensionality is not greater or equal 2, otherwise output would not "
                f"be batch-oriented.",
                context="workflow_compilation | execution_graph_construction | establishing_step_output_lineage",
            )
        return result_dimensionality
    if output_dimensionality_offset == 0:
        return reference_lineage
    reference_lineage.append(step_selector)
    return reference_lineage


def get_reference_lineage(
    step_selector: str,
    all_lineages: List[List[str]],
    input_data: StepInputData,
    dimensionality_reference_property: Optional[str],
) -> List[str]:
    if len(all_lineages) == 1:
        return copy(all_lineages[0])
    if dimensionality_reference_property not in input_data:
        raise AssumptionError(
            public_message=f"Workflow Compiler for step: `{step_selector}` expected dimensionality_reference_property "
            f"presence to be verified at earlier stages, which did not happen as expected. "
            f"This is most likely the bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
        )
    property_data = input_data[dimensionality_reference_property]
    if property_data.is_compound_input():
        lineage = None
        for nested_element in property_data.iterate_through_definitions():
            if nested_element.is_batch_oriented():
                lineage = copy(nested_element.data_lineage)
                return lineage
        if lineage is None:
            raise AssumptionError(
                public_message=f"Workflow Compiler for step: `{step_selector}` cannot establish output lineage. "
                f"At this stage it is expected to succeed - lack of success indicates bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full "
                f"context of the problem - including workflow definition you use.",
                context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
            )
    if not property_data.is_batch_oriented():
        raise AssumptionError(
            public_message=f"Workflow Compiler for step: `{step_selector}` cannot establish output lineage. "
            f"At this stage it is expected to succeed - lack of success indicates bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | collecting_step_inputs",
        )
    return copy(property_data.data_lineage)


def add_super_input_node_in_execution_graph(
    execution_graph: DiGraph,
    super_input_node: str,
) -> DiGraph:
    nodes_to_attach_super_input_into = get_nodes_of_specific_category(
        execution_graph=execution_graph,
        category=NodeCategory.INPUT_NODE,
    )
    step_nodes_without_predecessors = [
        node
        for node in execution_graph.nodes
        if not list(execution_graph.predecessors(node))
    ]
    nodes_to_attach_super_input_into.update(step_nodes_without_predecessors)
    execution_graph.add_node(super_input_node)
    for node in nodes_to_attach_super_input_into:
        execution_graph.add_edge(super_input_node, node)
    return execution_graph
