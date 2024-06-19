import itertools
from collections import defaultdict
from copy import copy
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from networkx import DiGraph

from inference.core.workflows.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
    WORKFLOW_INPUT_BATCH_LINEAGE_ID,
)
from inference.core.workflows.entities.base import (
    InputType,
    JsonField,
    OutputDefinition,
)
from inference.core.workflows.entities.types import STEP_AS_SELECTED_ELEMENT, Kind
from inference.core.workflows.errors import (
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
)
from inference.core.workflows.execution_engine.compiler.entities import (
    CompoundStepInputDefinition,
    DictOfStepInputDefinitions,
    DynamicStepInputDefinition,
    InputNode,
    ListOfStepInputDefinitions,
    NodeCategory,
    NodeInputCategory,
    OutputNode,
    ParameterSpecification,
    ParsedWorkflowDefinition,
    StaticStepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.compiler.reference_type_checker import (
    validate_reference_kinds,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    FLOW_CONTROL_NODE_KEY,
    construct_input_selector,
    construct_output_selector,
    construct_step_selector,
    get_last_chunk_of_selector,
    get_nodes_of_specific_category,
    get_step_selector_from_its_output,
    is_flow_control_step,
    is_input_node,
    is_input_selector,
    is_output_node,
    is_step_node,
    is_step_output_selector,
    is_step_selector,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    ParsedSelector,
)
from inference.core.workflows.execution_engine.introspection.selectors_parser import (
    get_step_selectors,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

NODE_DEFINITION_KEY = "definition"
STEP_INPUT_SELECTOR_PROPERTY = "step_input_selector"
EXCLUDED_FIELDS = {"type", "name"}


def prepare_execution_graph(
    workflow_definition: ParsedWorkflowDefinition,
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
    return execution_graph


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
        input_node_compilation_data: InputNode = execution_graph.nodes[
            target_step_parsed_selector.value
        ][NODE_COMPILATION_OUTPUT_PROPERTY]
        actual_input_kind = input_node_compilation_data.input_manifest.kind
    else:
        other_step_compilation_data: StepNode = execution_graph.nodes[
            target_step_selector
        ][NODE_COMPILATION_OUTPUT_PROPERTY]
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
    execution_graph.add_edge(
        target_step_selector,
        source_step_selector,
        **{STEP_INPUT_SELECTOR_PROPERTY: target_step_parsed_selector},
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
    source_compilation_data: StepNode = execution_graph.nodes[source_step_selector][
        NODE_COMPILATION_OUTPUT_PROPERTY
    ]
    target_compilation_data: StepNode = execution_graph.nodes[target_step_selector][
        NODE_COMPILATION_OUTPUT_PROPERTY
    ]
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


def get_nodes_that_are_reachable_from_pointed_ones_in_reversed_graph(
    execution_graph: DiGraph,
    pointed_nodes: Set[str],
) -> Set[str]:
    result = set()
    reversed_graph = execution_graph.reverse(copy=True)
    for pointed_node in pointed_nodes:
        nodes_reaching_pointed_one = list(
            nx.dfs_postorder_nodes(reversed_graph, pointed_node)
        )
        result.update(nodes_reaching_pointed_one)
    return result


def denote_workflow_dimensionality(
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
        if is_input_node(execution_graph=execution_graph, node=node):
            # everything already set there
            continue
        elif is_step_node(execution_graph=execution_graph, node=node):
            execution_graph = set_dimensionality_for_step(
                execution_graph=execution_graph,
                node=node,
                block_manifest_by_step_name=block_manifest_by_step_name,
            )
        elif is_output_node(execution_graph=execution_graph, node=node):
            # output is allowed to have exactly one predecessor
            predecessor_node = list(execution_graph.predecessors(node))[0]
            predecessor_node_data: Union[StepNode, InputNode] = execution_graph.nodes[
                predecessor_node
            ][NODE_COMPILATION_OUTPUT_PROPERTY]
            predecessor_node_lineage = predecessor_node_data.data_lineage
            output_node_data: OutputNode = execution_graph.nodes[node][
                NODE_COMPILATION_OUTPUT_PROPERTY
            ]
            output_node_data.data_lineage = predecessor_node_lineage
    execution_graph.remove_node(super_input_node)
    return execution_graph


def set_dimensionality_for_step(
    execution_graph: DiGraph,
    node: str,
    block_manifest_by_step_name: Dict[str, WorkflowBlockManifest],
) -> DiGraph:
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
    input_data = collect_input_data(
        manifest=block_manifest_by_step_name[get_last_chunk_of_selector(selector=node)],
        step_node=node,
        all_non_control_flow_predecessors=all_non_control_flow_predecessors,
        execution_graph=execution_graph,
    )
    step_name = get_last_chunk_of_selector(node)
    step_node_data: StepNode = execution_graph.nodes[node][
        NODE_COMPILATION_OUTPUT_PROPERTY
    ]
    print(f"input_data - {node}", input_data)
    inputs_dimensionalities = get_inputs_dimensionalities(
        step_name=step_name,
        input_data=input_data,
    )
    print("inputs_dimensionalities", inputs_dimensionalities)
    parameters_with_batch_inputs = grab_parameters_defining_batch_inputs(
        inputs_dimensionalities=inputs_dimensionalities,
    )
    manifest = block_manifest_by_step_name[step_name]
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
    all_lineages = get_all_data_lineage(step_name=step_name, input_data=input_data)
    verify_lineage_of_flow_control_steps_impacting_inputs(
        step_name=step_name,
        inputs_lineage=all_lineages,
        flow_control_steps_selectors=all_control_flow_predecessors,
        execution_graph=execution_graph,
    )
    step_node_data.input_data = input_data
    step_node_data.dimensionality_reference_property = dimensionality_reference_property
    step_node_data.batch_oriented_parameters = parameters_with_batch_inputs
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
        step_node_data.step_execution_dimensionality = step_execution_dimensionality
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


def collect_input_data(
    manifest: WorkflowBlockManifest,
    step_node: str,
    all_non_control_flow_predecessors: List[str],
    execution_graph: DiGraph,
) -> Dict[
    str,
    Union[
        DynamicStepInputDefinition,
        StaticStepInputDefinition,
        CompoundStepInputDefinition,
    ],
]:
    print("manifest.name", manifest.name, all_non_control_flow_predecessors)
    predecessors_by_property_name = defaultdict(list)
    for predecessor in all_non_control_flow_predecessors:
        edge_data = execution_graph.edges[(predecessor, step_node)]
        if STEP_INPUT_SELECTOR_PROPERTY not in edge_data:
            continue
        selector_associated_to_edge: ParsedSelector = edge_data[
            STEP_INPUT_SELECTOR_PROPERTY
        ]
        predecessors_by_property_name[
            selector_associated_to_edge.definition.property_name
        ].append((predecessor, selector_associated_to_edge))
    manifest_fields_values = get_manifest_fields_values(step_manifest=manifest)
    result = {}
    for name, value in manifest_fields_values.items():
        if isinstance(value, dict) and name in predecessors_by_property_name:
            result[name] = build_nested_dict_of_input_data(
                property_name=name,
                value=value,
                predecessors_by_property_name=predecessors_by_property_name,
                execution_graph=execution_graph,
            )
        elif isinstance(value, list) and name in predecessors_by_property_name:
            result[name] = build_nested_list_of_input_data(
                property_name=name,
                value=value,
                predecessors_by_property_name=predecessors_by_property_name,
                execution_graph=execution_graph,
            )
        else:
            if name not in predecessors_by_property_name:
                result[name] = StaticStepInputDefinition(
                    parameter_specification=ParameterSpecification(
                        parameter_name=name,
                    ),
                    category=NodeInputCategory.STATIC_VALUE,
                    value=value,
                )
            else:
                if len(predecessors_by_property_name[name]) != 1:
                    raise ValueError("Should not meet more than one predecessor here")
                predecessor_selector, predecessor_parsed_selector = (
                    predecessors_by_property_name[name][0]
                )
                predecessor_node_data = execution_graph.nodes[predecessor_selector][
                    NODE_COMPILATION_OUTPUT_PROPERTY
                ]
                if is_input_node(
                    execution_graph=execution_graph, node=predecessor_selector
                ):
                    category = (
                        NodeInputCategory.BATCH_INPUT_PARAMETER
                        if predecessor_node_data.is_batch_oriented()
                        else NodeInputCategory.NON_BATCH_INPUT_PARAMETER
                    )
                    result[name] = DynamicStepInputDefinition(
                        parameter_specification=ParameterSpecification(
                            parameter_name=name,
                        ),
                        category=category,
                        data_lineage=predecessor_node_data.data_lineage,
                        selector=predecessor_node_data.selector,
                    )
                elif is_step_node(
                    execution_graph=execution_graph, node=predecessor_selector
                ):
                    category = (
                        NodeInputCategory.BATCH_STEP_OUTPUT
                        if predecessor_node_data.output_dimensionality > 0
                        else NodeInputCategory.NON_BATCH_STEP_OUTPUT
                    )
                    result[name] = DynamicStepInputDefinition(
                        parameter_specification=ParameterSpecification(
                            parameter_name=name,
                        ),
                        category=category,
                        data_lineage=predecessor_node_data.data_lineage,
                        selector=predecessor_parsed_selector.value,
                    )
                else:
                    raise ValueError("Should not reach here")
    return result


def build_nested_dict_of_input_data(
    property_name: str,
    value: dict,
    predecessors_by_property_name: Dict[str, List[Tuple[str, ParsedSelector]]],
    execution_graph: DiGraph,
) -> DictOfStepInputDefinitions:
    nested_property_name2data = {
        e[1].key: e for e in predecessors_by_property_name[property_name]
    }
    result = {}
    for k, v in value.items():
        if k not in nested_property_name2data:
            result[k] = StaticStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name=property_name,
                    nested_element_key=k,
                ),
                category=NodeInputCategory.STATIC_VALUE,
                value=v,
            )
            continue
        referred_node_selector = nested_property_name2data[k][0]
        referred_node_parsed_selector = nested_property_name2data[k][1]
        referred_node_data = execution_graph.nodes[referred_node_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        if is_input_node(execution_graph=execution_graph, node=referred_node_selector):
            category = (
                NodeInputCategory.BATCH_INPUT_PARAMETER
                if referred_node_data.is_batch_oriented()
                else NodeInputCategory.NON_BATCH_INPUT_PARAMETER
            )
            result[k] = DynamicStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name=property_name,
                    nested_element_key=k,
                ),
                category=category,
                data_lineage=referred_node_data.data_lineage,
                selector=referred_node_data.selector,
            )
            continue
        category = (
            NodeInputCategory.BATCH_STEP_OUTPUT
            if referred_node_data.output_dimensionality > 0
            else NodeInputCategory.NON_BATCH_STEP_OUTPUT
        )
        result[k] = DynamicStepInputDefinition(
            parameter_specification=ParameterSpecification(
                parameter_name=property_name,
                nested_element_key=k,
            ),
            category=category,
            data_lineage=referred_node_data.data_lineage,
            selector=referred_node_parsed_selector.value,
        )
    return DictOfStepInputDefinitions(
        name=property_name,
        nested_definitions=result,
    )


def get_manifest_fields_values(step_manifest: WorkflowBlockManifest) -> Dict[str, Any]:
    result = {}
    for field in step_manifest.model_fields:
        if field in EXCLUDED_FIELDS:
            continue
        result[field] = getattr(step_manifest, field)
    return result


def build_nested_list_of_input_data(
    property_name: str,
    value: list,
    predecessors_by_property_name: Dict[str, List[Tuple[str, ParsedSelector]]],
    execution_graph: DiGraph,
) -> ListOfStepInputDefinitions:
    nested_index2data = {
        e[1].index: e for e in predecessors_by_property_name[property_name]
    }
    result = []
    for index, element in enumerate(value):
        if index not in nested_index2data:
            result.append(
                StaticStepInputDefinition(
                    parameter_specification=ParameterSpecification(
                        parameter_name=property_name,
                        nested_element_index=index,
                    ),
                    category=NodeInputCategory.STATIC_VALUE,
                    value=element,
                )
            )
            continue
        referred_node_selector = nested_index2data[index][0]
        referred_node_parsed_selector = nested_index2data[index][1]
        referred_node_data = execution_graph.nodes[referred_node_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        if is_input_node(execution_graph=execution_graph, node=referred_node_selector):
            category = (
                NodeInputCategory.BATCH_INPUT_PARAMETER
                if referred_node_data.is_batch_oriented()
                else NodeInputCategory.NON_BATCH_INPUT_PARAMETER
            )
            result.append(
                DynamicStepInputDefinition(
                    parameter_specification=ParameterSpecification(
                        parameter_name=property_name,
                        nested_element_index=index,
                    ),
                    category=category,
                    data_lineage=referred_node_data.data_lineage,
                    selector=referred_node_data.selector,
                )
            )
            continue
        category = (
            NodeInputCategory.BATCH_STEP_OUTPUT
            if referred_node_data.output_dimensionality > 0
            else NodeInputCategory.NON_BATCH_STEP_OUTPUT
        )
        result.append(
            DynamicStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name=property_name,
                    nested_element_index=index,
                ),
                category=category,
                data_lineage=referred_node_data.data_lineage,
                selector=referred_node_parsed_selector.value,
            )
        )
    return ListOfStepInputDefinitions(
        name=property_name,
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
        raise ValueError(
            f"Offsets could not be negative, but block defining step: {step_name} defines that values."
        )
    if abs(max_offset - min_offset) > 1:
        raise ValueError(
            f"Offsets of input parameters could not differ more than 1, but block defining step {step_name} "
            f"violates that rule."
        )


def verify_output_offset(
    step_name: str,
    parameters_with_batch_inputs: Set[str],
    input_dimensionality_offsets: Dict[str, int],
    dimensionality_reference_property: Optional[str],
    output_dimensionality_offset: int,
) -> None:
    if not parameters_with_batch_inputs and output_dimensionality_offset != 0:
        raise ValueError(
            f"Block defining step {step_name} defines dimensionality offset different than zero while taking "
            f"only non-batch parameters, which is not allowed."
        )
    if (
        dimensionality_reference_property is not None
        and dimensionality_reference_property not in parameters_with_batch_inputs
    ):
        raise ValueError(
            f"Block defining step {step_name} defines dimensionality reference property which is not in scope of "
            f"parameters bring batch-oriented input, which makes it impossible to use as reference for output "
            f"dimensionality."
        )
    if output_dimensionality_offset not in {-1, 0, 1}:
        raise ValueError(
            f"Block defining step {step_name} defines output dimensionality offset "
            f"being {output_dimensionality_offset}, whereas it is only possible for that offset being "
            f"in set [-1, 0, 1]."
        )
    different_offsets = {o for o in input_dimensionality_offsets.values()}
    if len(parameters_with_batch_inputs) != len(input_dimensionality_offsets):
        different_offsets.add(0)
    if 0 not in different_offsets and parameters_with_batch_inputs:
        raise ValueError(
            f"Block defining step {step_name} explicitly defines input dimensionalities offset s"
            f"with {input_dimensionality_offsets}, but the definition lack 0-level input, which is "
            f"not allowed, as in this scenario offsets could be adjusted to include 0"
        )
    if len(different_offsets) > 1 and dimensionality_reference_property is None:
        raise ValueError(
            f"Block defining step {step_name} explicitly defines input dimensionality "
            f"offsets {input_dimensionality_offsets}. In this scenario it is required to provide dimensionality "
            f"reference property."
        )
    if len(different_offsets) > 1 and output_dimensionality_offset != 0:
        raise ValueError(
            f"Block defining step {step_name} explicitly defines input dimensionality "
            f"offsets {input_dimensionality_offsets} and output dimensionality offset {output_dimensionality_offset} "
            f"where the latter is not 0, but for inputs differing with dimensionality it is only possible to keep "
            f"output dimensionality the same and point reference parameter."
        )


def verify_input_data_dimensionality(
    step_name: str,
    dimensionality_reference_property: Optional[str],
    inputs_dimensionalities: Dict[str, Set[int]],
    dimensionality_offstes: Dict[str, int],
) -> None:
    print("inputs_dimensionalities", inputs_dimensionalities)
    parameter2offset_and_non_zero_dimensionality = {}
    for parameter_name, dimensionality in inputs_dimensionalities.items():
        parameter_offset = dimensionality_offstes.get(parameter_name, 0)
        print("offset for parameter", parameter_name, parameter_offset)
        non_zero_dims = {d for d in dimensionality if d > 0}
        if len(non_zero_dims) > 1:
            raise ValueError("Should not be possible here")
        if non_zero_dims:
            dim = next(iter(non_zero_dims))
            parameter2offset_and_non_zero_dimensionality[parameter_name] = (
                parameter_offset,
                dim,
            )
    if not parameter2offset_and_non_zero_dimensionality:
        return None
    if dimensionality_reference_property is None:
        different_dims = {
            e[1] for e in parameter2offset_and_non_zero_dimensionality.values()
        }
        if len(different_dims) != 1:
            param2dim = {
                k: e[1] for k, e in parameter2offset_and_non_zero_dimensionality.items()
            }
            raise ValueError(
                f"Block defining step {step_name} does not define dimensionality reference property, "
                f"which means that all batch-oriented parameters must be at the same dimensionality level, "
                f"but detected the following dimensionalities for parameters {param2dim}"
            )
        return None
    reference_offset, reference_property_dim = (
        parameter2offset_and_non_zero_dimensionality[dimensionality_reference_property]
    )
    expected_dimensionalities = {
        property_name: (e[0] - reference_offset) + reference_property_dim
        for property_name, e in parameter2offset_and_non_zero_dimensionality.items()
    }
    if any(v <= 0 for v in expected_dimensionalities.values()):
        raise ValueError(
            f"Given the definition of block defining step {step_name} and data provided, "
            f"the block would expect batch input dimensionality to be 0 or below, which is invalid."
        )

    print("expected_dimensionalities", expected_dimensionalities)
    print(
        "parameter2offset_and_non_zero_dimensionality",
        parameter2offset_and_non_zero_dimensionality,
    )
    for property_name, expected_dimensionality in expected_dimensionalities.items():
        actual_dimensionality = parameter2offset_and_non_zero_dimensionality[
            property_name
        ][1]
        if actual_dimensionality != expected_dimensionality:
            raise ValueError(
                f"Data fed into step `{step_name}` property `{property_name}` has "
                f"actual dimensionality offset {actual_dimensionality}, "
                f"when expected was {expected_dimensionality}"
            )
    return None


def verify_lineage_of_flow_control_steps_impacting_inputs(
    step_name: str,
    inputs_lineage: List[List[str]],
    flow_control_steps_selectors: List[str],
    execution_graph: DiGraph,
) -> None:
    already_spotted_input_lineages = set()
    lineage_id2control_flow_steps = defaultdict(list)
    batch_oriented_control_flow_lineages = []
    for flow_control_steps_selector in flow_control_steps_selectors:
        flow_control_step_data = execution_graph.nodes[flow_control_steps_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        lineage = flow_control_step_data.data_lineage
        lineage_id = identify_lineage(lineage=lineage)
        if lineage_id not in already_spotted_input_lineages and lineage:
            already_spotted_input_lineages.add(lineage_id)
            batch_oriented_control_flow_lineages.append(lineage)
        lineage_id2control_flow_steps[lineage_id].append(flow_control_steps_selector)
    all_input_lineage_prefixes = get_all_lineage_prefixes(lineages=inputs_lineage)
    all_input_lineage_prefixes_hashes = {
        identify_lineage(lineage=lineage) for lineage in all_input_lineage_prefixes
    }
    for control_flow_lineage in batch_oriented_control_flow_lineages:
        control_flow_lineage_id = identify_lineage(lineage=control_flow_lineage)
        if control_flow_lineage_id not in all_input_lineage_prefixes_hashes:
            problematic_flow_control_steps = lineage_id2control_flow_steps[
                control_flow_lineage_id
            ]
            raise ValueError(
                f"Step {step_name} execution is impacted by control flow outcome of the following "
                f"steps {problematic_flow_control_steps} which make decision based on data that is "
                f"not compatible with data fed to the step {step_name} - which would cause the step "
                f"to never execute. This behaviour is invalid and prevented upfront by Workflows compiler."
            )


def get_all_lineage_prefixes(lineages: List[List[str]]) -> List[List[str]]:
    result = []
    already_spotted = set()
    for lineage in lineages:
        lineage_prefixes = get_lineage_prefixes(lineage=lineage)
        for lineage_prefix in lineage_prefixes:
            lineage_prefix_id = identify_lineage(lineage=lineage_prefix)
            if lineage_prefix_id not in already_spotted:
                already_spotted.add(lineage_prefix_id)
                result.append(lineage_prefix)
    return result


def get_lineage_prefixes(lineage: List[str]) -> List[List[str]]:
    if not lineage:
        return []
    result = []
    prefix = []
    for i in range(len(lineage)):
        prefix.append(lineage[i])
        result.append(copy(prefix))
    return result


def get_inputs_dimensionalities(
    step_name: str,
    input_data: Dict[
        str,
        Union[
            DynamicStepInputDefinition,
            StaticStepInputDefinition,
            CompoundStepInputDefinition,
        ],
    ],
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
            raise ValueError(
                f"For step {step_name} attempted to plug input data differing in dimensionality more than 1"
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
        raise ValueError(
            f"While evaluating compound property {property_name} of step {step_name}, "
            f"detected multiple inputs of differing batch dimensionalities: {non_zero_dimensionalities}"
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


def get_all_data_lineage(
    step_name: str,
    input_data: Dict[
        str,
        Union[
            DynamicStepInputDefinition,
            StaticStepInputDefinition,
            CompoundStepInputDefinition,
        ],
    ],
) -> List[List[str]]:
    lineage_id2lineage = set()
    lineages = []
    for property_name, input_definition in input_data.items():
        if input_definition.is_compound_input():
            lineages_detected_in_compound_input = []
            for nested_element in input_definition.iterate_through_definitions():
                if nested_element.is_batch_oriented():
                    lineage = nested_element.data_lineage
                    lineage_id = identify_lineage(lineage=lineage)
                    if lineage_id not in lineage_id2lineage:
                        lineage_id2lineage.add(lineage_id)
                        lineages.append(lineage)
                        lineages_detected_in_compound_input.append(
                            lineages_detected_in_compound_input
                        )
            if len(lineages_detected_in_compound_input) > 1:
                raise ValueError(
                    f"Input data provided for step: `{step_name}` comes with multiple different lineages "
                    f"{lineages_detected_in_compound_input} for property `{property_name}` accepting "
                    f"multiple selectors"
                )
        else:
            if input_definition.is_batch_oriented():
                lineage = input_definition.data_lineage
                lineage_id = identify_lineage(lineage=lineage)
                if lineage_id not in lineage_id2lineage:
                    lineage_id2lineage.add(lineage_id)
                    lineages.append(lineage)
    if not lineages:
        return lineages
    lineages_by_length = defaultdict(list)
    for lineage in lineages:
        lineages_by_length[len(lineage)].append(lineage)
    if len(lineages_by_length) > 2:
        raise ValueError(
            f"Input data provided for step: `{step_name}` comes with lineages at more than two "
            f"dimensionality levels, which should not be possible."
        )
    lineage_lengths = sorted(lineages_by_length.keys())
    for lineage_length in lineage_lengths:
        if len(lineages_by_length[lineage_length]) > 1:
            raise ValueError(
                f"Among step `{step_name}` inputs found different lineages at the same  "
                f"dimensionality levels dimensionality."
            )
    if len(lineages_by_length[lineage_lengths[-1]]) > 1 and len(lineage_lengths) < 2:
        raise ValueError(
            f"If lineage differ at the last level, then Execution Engine requires at least one higher-dimension "
            f"input that will come with common lineage, but this is not the case for step {step_name}."
        )
    print("lineages_by_length", lineages_by_length)
    if len(lineage_lengths) == 2:
        reference_lineage = lineages_by_length[lineage_lengths[0]][0]
        print("diff", reference_lineage, lineages_by_length[lineage_lengths[1]][0][:-1])
        if reference_lineage != lineages_by_length[lineage_lengths[1]][0][:-1]:
            raise ValueError(
                f"Step `{step_name}` inputs does not share common lineage. Differing element: "
                f"{lineages_by_length[lineage_lengths[1]][0]}, "
                f"reference lineage prefix: {reference_lineage}"
            )
    return lineages


def identify_lineage(lineage: List[str]) -> int:
    return sum(hash(e) for e in lineage)


def establish_batch_oriented_step_lineage(
    step_selector: str,
    all_lineages: List[List[str]],
    input_data: Dict[
        str,
        Union[
            DynamicStepInputDefinition,
            StaticStepInputDefinition,
            CompoundStepInputDefinition,
        ],
    ],
    dimensionality_reference_property: Optional[str],
    output_dimensionality_offset: int,
) -> List[str]:
    reference_lineage = get_reference_lineage(
        all_lineages=all_lineages,
        input_data=input_data,
        dimensionality_reference_property=dimensionality_reference_property,
    )
    if output_dimensionality_offset < 0:
        result_dimensionality = reference_lineage[:output_dimensionality_offset]
        if len(result_dimensionality) == 0:
            raise ValueError(
                f"Step {step_selector} is to decrease dimensionality, but it is not possible if "
                f"input dimensionality is not greater or equal 2, otherwise output is not batch-oriented."
            )
        return result_dimensionality
    if output_dimensionality_offset == 0:
        return reference_lineage
    reference_lineage.append(step_selector)
    return reference_lineage


def get_reference_lineage(
    all_lineages: List[List[str]],
    input_data: Dict[
        str,
        Union[
            DynamicStepInputDefinition,
            StaticStepInputDefinition,
            CompoundStepInputDefinition,
        ],
    ],
    dimensionality_reference_property: Optional[str],
) -> List[str]:
    if len(all_lineages) == 1:
        return copy(all_lineages[0])
    if dimensionality_reference_property not in input_data:
        raise ValueError(
            "Dimensionality reference property not set and that should be picked up earlier"
        )
    property_data = input_data[dimensionality_reference_property]
    if property_data.is_compound_input():
        lineage = None
        for nested_element in property_data.iterate_through_definitions():
            if nested_element.is_batch_oriented():
                lineage = copy(nested_element.data_lineage)
                return lineage
        if lineage is None:
            raise ValueError(
                "Property cannot be taken as linage reference and that should be picked up earlier"
            )
    else:
        if not property_data.is_batch_oriented():
            raise ValueError(
                "Property cannot be taken as linage reference and that should be picked up earlier"
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
    execution_graph.add_node(super_input_node, kind="SUPER_INPUT")
    for node in nodes_to_attach_super_input_into:
        execution_graph.add_edge(super_input_node, node)
    return execution_graph


def traverse_graph_ensuring_parents_are_reached_first(
    graph: DiGraph,
    start_node: str,
) -> List[str]:
    graph_copy = graph.copy()
    distance_key = "distance"
    graph_copy = assign_max_distances_from_start(
        graph=graph_copy,
        start_node=start_node,
        distance_key=distance_key,
    )
    nodes_groups = group_nodes_by_sorted_key_value(graph=graph_copy, key=distance_key)
    return [node for node_group in nodes_groups for node in node_group]


def assign_max_distances_from_start(
    graph: nx.DiGraph, start_node: str, distance_key: str = "distance"
) -> nx.DiGraph:
    nodes_to_consider = Queue()
    nodes_to_consider.put(start_node)
    while nodes_to_consider.qsize() > 0:
        node_to_consider = nodes_to_consider.get()
        predecessors = list(graph.predecessors(node_to_consider))
        if not all(graph.nodes[p].get(distance_key) is not None for p in predecessors):
            # we can proceed to establish distance, only if all parents have distances established
            continue
        if len(predecessors) == 0:
            distance_from_start = 0
        else:
            distance_from_start = (
                max(graph.nodes[p][distance_key] for p in predecessors) + 1
            )
        graph.nodes[node_to_consider][distance_key] = distance_from_start
        for neighbour in graph.successors(node_to_consider):
            nodes_to_consider.put(neighbour)
    return graph


def group_nodes_by_sorted_key_value(
    graph: nx.DiGraph,
    key: str,
    excluded_nodes: Optional[Set[str]] = None,
) -> List[List[str]]:
    if excluded_nodes is None:
        excluded_nodes = set()
    key2nodes = defaultdict(list)
    for node_name, node_data in graph.nodes(data=True):
        if node_name in excluded_nodes:
            continue
        key2nodes[node_data[key]].append(node_name)
    sorted_key_values = sorted(list(key2nodes.keys()))
    return [key2nodes[d] for d in sorted_key_values]
