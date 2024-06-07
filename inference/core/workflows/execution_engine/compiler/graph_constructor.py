import itertools
from collections import defaultdict
from copy import copy
from queue import Queue
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type

import networkx as nx
from networkx import DiGraph

from inference.core.workflows.constants import (
    DIMENSIONALITY_PROPERTY,
    EXECUTION_BRANCHES_STACK_PROPERTY,
    INPUT_NODE_KIND,
    OUTPUT_NODE_KIND,
    STEP_INPUT_PROPERTY,
    STEP_NODE_KIND,
)
from inference.core.workflows.entities.base import (
    InputType,
    JsonField,
    OutputDefinition,
)
from inference.core.workflows.entities.types import STEP_AS_SELECTED_ELEMENT, Kind
from inference.core.workflows.errors import (
    ConditionalBranchesCollapseError,
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
)
from inference.core.workflows.execution_engine.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.compiler.reference_type_checker import (
    validate_reference_kinds,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    FLOW_CONTROL_NODE_KEY,
    construct_input_selector,
    construct_output_name,
    construct_step_selector,
    get_last_chunk_of_selector,
    get_nodes_of_specific_kind,
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
    execution_graph = add_edges_for_outputs(
        workflow_definition=workflow_definition,
        execution_graph=execution_graph,
    )
    return denote_execution_branches(execution_graph=execution_graph)


def add_input_nodes_for_graph(
    inputs: List[InputType],
    execution_graph: DiGraph,
) -> DiGraph:
    for input_spec in inputs:
        input_selector = construct_input_selector(input_name=input_spec.name)
        execution_graph.add_node(
            input_selector,
            kind=INPUT_NODE_KIND,
            definition=input_spec,
            batch_oriented=input_spec.is_batch_oriented(),
        )
    return execution_graph


def add_steps_nodes_for_graph(
    steps: List[WorkflowBlockManifest],
    execution_graph: DiGraph,
) -> DiGraph:
    for step in steps:
        step_selector = construct_step_selector(step_name=step.name)
        execution_graph.add_node(
            step_selector,
            kind=STEP_NODE_KIND,
            definition=step,
        )
    return execution_graph


def add_output_nodes_for_graph(
    outputs: List[JsonField],
    execution_graph: DiGraph,
) -> DiGraph:
    for output_spec in outputs:
        execution_graph.add_node(
            construct_output_name(name=output_spec.name),
            kind=OUTPUT_NODE_KIND,
            definition=output_spec,
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
            parsed_selectors=step_selectors,
        )
    return execution_graph


def add_edges_for_step(
    execution_graph: DiGraph,
    step_name: str,
    parsed_selectors: List[ParsedSelector],
) -> DiGraph:
    step_selector = construct_step_selector(step_name=step_name)
    for parsed_selector in parsed_selectors:
        execution_graph = add_edge_for_step(
            execution_graph=execution_graph,
            step_selector=step_selector,
            parsed_selector=parsed_selector,
        )
    return execution_graph


def add_edge_for_step(
    execution_graph: DiGraph,
    step_selector: str,
    parsed_selector: ParsedSelector,
) -> DiGraph:
    other_step_selector = get_step_selector_from_its_output(
        step_output_selector=parsed_selector.value
    )
    verify_edge_is_created_between_existing_nodes(
        execution_graph=execution_graph,
        start=step_selector,
        end=other_step_selector,
    )
    if is_step_selector(parsed_selector.value):
        return establish_flow_control_edge(
            step_selector=step_selector,
            parsed_selector=parsed_selector,
            execution_graph=execution_graph,
        )
    if is_input_selector(parsed_selector.value):
        actual_input_kind = execution_graph.nodes[parsed_selector.value][
            NODE_DEFINITION_KEY
        ].kind
    else:
        other_step_manifest: WorkflowBlockManifest = execution_graph.nodes[
            other_step_selector
        ][NODE_DEFINITION_KEY]
        actual_input_kind = get_kind_of_value_provided_in_step_output(
            step_manifest=other_step_manifest,
            step_property=get_last_chunk_of_selector(selector=parsed_selector.value),
        )
    expected_input_kind = list(
        itertools.chain.from_iterable(
            ref.kind for ref in parsed_selector.definition.allowed_references
        )
    )
    error_message = (
        f"Failed to validate reference provided for step: {step_selector} regarding property: "
        f"{parsed_selector.definition.property_name} with value: {parsed_selector.value}. "
        f"Allowed kinds of references for this property: {list(set(e.name for e in expected_input_kind))}. "
        f"Types of output for referred property: {list(set(a.name for a in actual_input_kind))}"
    )
    validate_reference_kinds(
        expected=expected_input_kind,
        actual=actual_input_kind,
        error_message=error_message,
    )
    execution_graph.add_edge(
        other_step_selector,
        step_selector,
        **{STEP_INPUT_PROPERTY: parsed_selector.definition.property_name},
    )
    return execution_graph


def establish_flow_control_edge(
    step_selector: str,
    parsed_selector: ParsedSelector,
    execution_graph: DiGraph,
) -> DiGraph:
    if not step_definition_allows_flow_control_references(
        parsed_selector=parsed_selector
    ):
        raise ExecutionGraphStructureError(
            public_message=f"Detected reference to other step in manifest of step {step_selector}. "
            f"This is not allowed, as step manifest does not define any properties of type `StepSelector` "
            f"allowing for defining connections between steps that represent flow control in `workflows`.",
            context="workflow_compilation | execution_graph_construction",
        )
    other_node_selector = get_step_selector_from_its_output(
        step_output_selector=parsed_selector.value
    )
    execution_graph.add_edge(step_selector, other_node_selector)
    execution_graph.nodes[step_selector][FLOW_CONTROL_NODE_KEY] = True
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
        output_name = construct_output_name(name=output.name)
        verify_edge_is_created_between_existing_nodes(
            execution_graph=execution_graph,
            start=node_selector,
            end=output_name,
        )
        if is_step_output_selector(selector_or_value=output.selector):
            step_manifest = execution_graph.nodes[node_selector][NODE_DEFINITION_KEY]
            step_outputs = step_manifest.get_actual_outputs()
            verify_output_selector_points_to_valid_output(
                output_selector=output.selector,
                step_outputs=step_outputs,
            )
        execution_graph.add_edge(node_selector, output_name)
    return execution_graph


def denote_execution_branches(execution_graph: DiGraph) -> DiGraph:
    super_input_node = "<super-input>"
    execution_graph = add_super_input_node_in_execution_graph(
        execution_graph=execution_graph,
        super_input_node=super_input_node,
    )
    execution_graph = denote_execution_branches_for_node(
        node_name=super_input_node,
        branches=["Branch[root]"],
        execution_graph=execution_graph,
    )
    for node in traverse_graph_ensuring_parents_are_reached_first(
        graph=execution_graph,
        start_node=super_input_node,
    ):
        if is_flow_control_step(execution_graph=execution_graph, node=node):
            execution_graph = start_execution_branches_for_node_successors(
                execution_graph=execution_graph,
                node=node,
            )
        elif node_merges_execution_paths(execution_graph=execution_graph, node=node):
            execution_graph = denote_common_execution_branch_for_merged_branches(
                execution_graph=execution_graph,
                node=node,
            )
        else:
            execution_graph = denote_execution_branches_for_nodes(
                node_names=execution_graph.successors(node),
                branches=execution_graph.nodes[node][EXECUTION_BRANCHES_STACK_PROPERTY],
                execution_graph=execution_graph,
            )
    execution_graph.remove_node(super_input_node)
    ensure_all_nodes_have_execution_branch_associated(execution_graph=execution_graph)
    return execution_graph


def start_execution_branches_for_node_successors(
    execution_graph: DiGraph, node: str
) -> DiGraph:
    node_execution_branches_stack = execution_graph.nodes[node][
        EXECUTION_BRANCHES_STACK_PROPERTY
    ]
    for successor in execution_graph.successors(node):
        successor_stack = copy(node_execution_branches_stack)
        successor_stack.append(f"Branch[{node} -> {successor}]")
        execution_graph = denote_execution_branches_for_node(
            node_name=successor,
            branches=successor_stack,
            execution_graph=execution_graph,
        )
    return execution_graph


def node_merges_execution_paths(execution_graph: DiGraph, node: str) -> bool:
    node_predecessors = list(execution_graph.predecessors(node))
    if len(node_predecessors) < 2:
        return False
    reference_execution_branches_stack = execution_graph.nodes[node_predecessors[0]][
        EXECUTION_BRANCHES_STACK_PROPERTY
    ]
    for predecessor in node_predecessors[1:]:
        predecessor_execution_branches_stack = execution_graph.nodes[predecessor][
            EXECUTION_BRANCHES_STACK_PROPERTY
        ]
        if reference_execution_branches_stack != predecessor_execution_branches_stack:
            return True
    return False


def denote_common_execution_branch_for_merged_branches(
    execution_graph: DiGraph, node: str
) -> DiGraph:
    node_predecessors = list(execution_graph.predecessors(node))
    execution_branches_stacks_to_merge = [
        execution_graph.nodes[node_predecessor][EXECUTION_BRANCHES_STACK_PROPERTY]
        for node_predecessor in node_predecessors
    ]
    merged_stack = find_longest_common_array_elements_prefix(
        arrays=execution_branches_stacks_to_merge,
    )
    if len(merged_stack) == 0:
        raise ValueError(f"Could not merge execution branches defined in step: {node}")
    execution_graph = denote_execution_branches_for_node(
        node_name=node, branches=merged_stack, execution_graph=execution_graph
    )
    return denote_execution_branches_for_nodes(
        node_names=execution_graph.successors(node),
        branches=merged_stack,
        execution_graph=execution_graph,
    )


def find_longest_common_array_elements_prefix(arrays: List[List[str]]) -> List[str]:
    if len(arrays) == 0:
        return []
    if len(arrays) == 1:
        return copy(arrays[0])
    longest_common_prefix = []
    shortest_array = min(len(array) for array in arrays)
    for i in range(shortest_array):
        reference_element = arrays[0][i]
        for j in range(1, len(arrays)):
            if arrays[j][i] != reference_element:
                return longest_common_prefix
        longest_common_prefix.append(reference_element)
    return longest_common_prefix


def denote_execution_branches_for_nodes(
    node_names: Iterable[str], branches: List[str], execution_graph: DiGraph
) -> DiGraph:
    for node_name in node_names:
        execution_graph = denote_execution_branches_for_node(
            node_name=node_name,
            branches=branches,
            execution_graph=execution_graph,
        )
    return execution_graph


def denote_execution_branches_for_node(
    node_name: str, branches: List[str], execution_graph: DiGraph
) -> DiGraph:
    execution_graph.nodes[node_name][EXECUTION_BRANCHES_STACK_PROPERTY] = branches
    return execution_graph


def ensure_all_nodes_have_execution_branch_associated(execution_graph: DiGraph) -> None:
    for node in execution_graph.nodes:
        if EXECUTION_BRANCHES_STACK_PROPERTY not in execution_graph.nodes[node]:
            raise ValueError(f"Could not associate execution branch for {node}")


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
    available_bocks: List[BlockSpecification],
) -> nx.DiGraph:
    block_class_by_step_name = {
        block.manifest_class.name: block.block_class for block in available_bocks
    }
    super_input_node = "<super-input>"
    execution_graph = add_super_input_node_in_execution_graph(
        execution_graph=execution_graph,
        super_input_node=super_input_node,
    )
    execution_graph.nodes[super_input_node][DIMENSIONALITY_PROPERTY] = 0
    for node in traverse_graph_ensuring_parents_are_reached_first(
        graph=execution_graph,
        start_node=super_input_node,
    ):
        if is_input_node(execution_graph=execution_graph, node=node):
            dimensionality = (
                1
                if execution_graph.nodes[node][NODE_DEFINITION_KEY].is_batch_oriented()
                else 0
            )
            execution_graph = set_dimensionality_for_node(
                execution_graph=execution_graph,
                node=node,
                dimensionality=dimensionality,
            )
        elif is_step_node(execution_graph=execution_graph, node=node):
            execution_graph = set_dimensionality_for_step(
                execution_graph=execution_graph,
                node=node,
                block_class_by_step_name=block_class_by_step_name,
            )
        elif is_output_node(execution_graph=execution_graph, node=node):
            # output is allowed to have exactly one predecessors
            predecessor_node = list(execution_graph.predecessors(node))[0]
            execution_graph = set_dimensionality_for_node(
                execution_graph=execution_graph,
                node=node,
                dimensionality=execution_graph.nodes[predecessor_node][
                    DIMENSIONALITY_PROPERTY
                ],
            )
    execution_graph.remove_node(super_input_node)
    ensure_all_nodes_have_dimensionality_associated(execution_graph=execution_graph)
    return execution_graph


DIMENSIONALITY_DELTAS_BY_MODE = {
    "reduces": -1,
    "keeps_the_same": 0,
    "increases": 1,
}


def set_dimensionality_for_step(
    execution_graph: DiGraph,
    node: str,
    block_class_by_step_name: Dict[str, Type[WorkflowBlock]],
) -> DiGraph:
    step_name = get_last_chunk_of_selector(selector=node)
    if not block_class_by_step_name[step_name].produces_batch_output():
        print(f"Step {node} does not produce batches")
        return set_dimensionality_for_node(
            execution_graph=execution_graph,
            node=node,
            dimensionality=0,
        )
    predecessors_dimensionalities = defaultdict(list)
    all_encountered_batch_dimensionalities = set()
    for predecessor in execution_graph.predecessors(node):
        predecessor_dimensionality = execution_graph.nodes[predecessor][
            DIMENSIONALITY_PROPERTY
        ]
        print(f"Predecessor: {predecessor} - dim: {predecessor_dimensionality}")
        edge_properties = execution_graph.edges[(predecessor, node)]
        input_property = edge_properties.get(STEP_INPUT_PROPERTY)
        predecessors_dimensionalities[input_property].append(predecessor_dimensionality)
        if predecessor_dimensionality == 0:
            continue
        all_encountered_batch_dimensionalities.add(predecessor_dimensionality)
    print(
        f"all_encountered_batch_dimensionalities for {node}: {all_encountered_batch_dimensionalities}"
    )
    if len(all_encountered_batch_dimensionalities) == 0:
        # no inputs found, step will produce singular output
        print(f"No batch inputs found for {node}")
        return set_dimensionality_for_node(
            execution_graph=execution_graph,
            node=node,
            dimensionality=0,
        )
    dimensionality_change = DIMENSIONALITY_DELTAS_BY_MODE[
        block_class_by_step_name[step_name].get_impact_on_data_dimensionality()
    ]
    dimensionality_reference_property = block_class_by_step_name[
        step_name
    ].get_data_dimensionality_property()
    if dimensionality_reference_property is None:
        if len(all_encountered_batch_dimensionalities) > 1:
            dimensionality_summary = ""
            for prop, dims in predecessors_dimensionalities.items():
                dims_stringified = [str(e) for e in dims]
                dimensionality_summary = f"{dimensionality_summary}\t{prop}: [{', '.join(dims_stringified)}]\n"
            raise RuntimeError(
                f"While verifying of workflow definition encountered step {node} that "
                f"requires all batch inputs to be of the same dimensionality, but got: \n"
                f"{dimensionality_summary}"
            )
        inputs_dim = next(iter(all_encountered_batch_dimensionalities))
        return set_dimensionality_for_node(
            execution_graph=execution_graph,
            node=node,
            dimensionality=inputs_dim + dimensionality_change,
        )
    if dimensionality_reference_property not in predecessors_dimensionalities:
        raise RuntimeError(
            f"Block implementing step {step_name} declared reference dimensionality property to be "
            f"{dimensionality_reference_property}, but no such property fed with data according to "
            f"workflow manifest."
        )
    dimensionalities_for_reference_property = set(
        predecessors_dimensionalities[dimensionality_reference_property]
    )
    allowed_different_dimensions = 1
    if 0 in dimensionalities_for_reference_property:
        allowed_different_dimensions += 1
    if len(dimensionalities_for_reference_property) > allowed_different_dimensions:
        raise ValueError(
            f"For step {node}, property: {dimensionality_reference_property} fed with data that declare the"
            f" following different dimensions: {dimensionalities_for_reference_property}, whereas it is allowed to "
            f"combine non-batch inputs and batches with the same number of dimensions."
        )
    return set_dimensionality_for_node(
        execution_graph=execution_graph,
        node=node,
        dimensionality=max(dimensionalities_for_reference_property)
        + dimensionality_change,
    )


def set_dimensionality_for_node(
    execution_graph: DiGraph, node: str, dimensionality: int
) -> DiGraph:
    if dimensionality < 0:
        raise ValueError(
            f"Attempted to set dimensionality {dimensionality} for step: {node}"
        )
    execution_graph.nodes[node][DIMENSIONALITY_PROPERTY] = dimensionality
    return execution_graph


def add_super_input_node_in_execution_graph(
    execution_graph: DiGraph,
    super_input_node: str,
) -> DiGraph:
    input_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=INPUT_NODE_KIND
    )
    execution_graph.add_node(super_input_node, kind="SUPER_INPUT")
    for node in input_nodes:
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


def ensure_all_nodes_have_dimensionality_associated(execution_graph: DiGraph) -> None:
    for node in execution_graph.nodes:
        if DIMENSIONALITY_PROPERTY not in execution_graph.nodes[node]:
            raise ValueError(f"Could not associate dimensionality for {node}")
