import itertools
from collections import defaultdict
from copy import copy
from queue import Queue
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple

import networkx as nx
from networkx import DiGraph

from inference.core.workflows.constants import (
    EXECUTION_BRANCHES_STACK_PROPERTY,
    INPUT_NODE_KIND,
    OUTPUT_NODE_KIND,
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
    is_input_selector,
    is_step_output_selector,
    is_step_selector,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    ParsedSelector,
)
from inference.core.workflows.execution_engine.introspection.selectors_parser import (
    get_step_selectors,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

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
    verify_each_node_step_has_parent_in_the_same_branch(execution_graph=execution_graph)
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
    execution_graph.add_edge(other_step_selector, step_selector)
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
    input_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=INPUT_NODE_KIND
    )
    execution_graph.add_node(super_input_node, kind="SUPER_INPUT")
    execution_graph = denote_execution_branches_for_node(
        node_name=super_input_node,
        branches=["Branch[root]"],
        execution_graph=execution_graph,
    )
    for node in input_nodes:
        execution_graph.add_edge(super_input_node, node)
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


def verify_each_node_step_has_parent_in_the_same_branch(
    execution_graph: DiGraph,
) -> None:
    """
    Conditional branching creates a bit of mess, in terms of determining which
    steps to execute.
    Let's imagine graph:
              / -> B -> C -> D
    A -> IF <             \
              \ -> E -> F -> G -> H
    where node G requires node C even though IF branched the execution. In other
    words - the problem (1) emerges if a node of kind STEP has a parent (node from which
    it can be achieved) of kind STEP and this parent is in a different branch (point out that
    we allow for a single step to have multiple steps as input, but they must be at the same
    execution path - for instance if D requires an output from C and B - this is allowed).
    Additionally, we must prevent situation when outcomes of branches started by two or more
    condition steps merge with each other, as condition eval may result in contradictory
    execution (2).


    We need to detect that situations upfront, such that we can raise error of ambiguous execution path
    rather than run time-consuming computations that will end up in error.

    To detect problem (1), first we detect steps with more than one parent step.
    From those steps we trace what sequence of steps would lead to execution engine reaching them.
    We analyse paths from those parent nodes in reversed topological order (from those nodes towards entry
    nodes of execution graph). While our analysis, on each path we denote control flow steps and
    result of evaluation that must have been observed in runtime (which next step in normal flow must be
    chosen to form the path). If we detect that for any control flow step we would need to output multiple
    values at time (more than one registered next step of control flow step) - we raise error.

    To detect problem (2) - we only let number of all different condition steps spotted in our traversal of
    execution graph (while checking (1)) to be the max number condition steps in a single path from graph start node
    to parent of problematic step with >= 2 parents.

    Beware that the latter part of algorithm has quite bad time complexity in general case.
    Worst part of algorithm runs at O(V^4) - at least taking coarse, worst-case estimations.
    In fact, for reasonable workflows we can expect this is not so bad:
    * The number of step nodes with multiple parents that we loop over in main loop, reduces the number of
    steps we iterate through in inner loops, as we are dealing with DAG (with quite limited amount of edges)
    and for each multi-parent node takes at least two other nodes (to construct a suspicious group) -
    so expected number of iterations in main loop is low - let's say 1-3 for a real graph.
    * for any reasonable execution graph, the complexity should be acceptable.
    """
    steps_with_more_than_one_parent = detect_steps_with_more_than_one_parent_step(
        execution_graph=execution_graph
    )  # O(V+E)
    if not steps_with_more_than_one_parent:
        return None
    reversed_steps_graph = construct_reversed_graph_with_steps_only(
        execution_graph=execution_graph
    )  # O(V+E)
    reversed_topological_order = list(
        nx.topological_sort(reversed_steps_graph)
    )  # O(V+E)
    for step in steps_with_more_than_one_parent:  # O(V)
        verify_multi_parent_step_execution_paths(
            reversed_steps_graph=reversed_steps_graph,
            reversed_topological_order=reversed_topological_order,
            step=step,
        )


def detect_steps_with_more_than_one_parent_step(execution_graph: DiGraph) -> Set[str]:
    steps_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=STEP_NODE_KIND,
    )
    edges_of_steps_nodes = [edge for edge in execution_graph.edges()]
    steps_parents = defaultdict(set)
    for edge in edges_of_steps_nodes:
        parent, child = edge
        if parent not in steps_nodes or child not in steps_nodes:
            continue
        steps_parents[child].add(parent)
    return {key for key, value in steps_parents.items() if len(value) > 1}


def construct_reversed_graph_with_steps_only(execution_graph: DiGraph) -> DiGraph:
    reversed_steps_graph = execution_graph.reverse()
    for node, node_data in list(reversed_steps_graph.nodes(data=True)):
        if node_data.get("kind") != STEP_NODE_KIND:
            reversed_steps_graph.remove_node(node)
    return reversed_steps_graph


def verify_multi_parent_step_execution_paths(
    reversed_steps_graph: nx.DiGraph,
    reversed_topological_order: List[str],
    step: str,
) -> None:
    control_flow_steps_successors = defaultdict(set)
    max_conditions_steps_in_execution_branch = 0
    for parent_of_investigated_step in reversed_steps_graph.successors(step):  # O(V)
        reversed_flow_path = (
            construct_path_to_step_through_selected_parent(  # O(E) -> O(V^2)
                graph=reversed_steps_graph,
                topological_order=reversed_topological_order,
                parent_step=parent_of_investigated_step,
                step=step,
            )
        )
        (
            control_flow_steps_successors,
            condition_steps,
        ) = denote_flow_control_steps_successors_in_normal_flow(  # O(V)
            reversed_steps_graph=reversed_steps_graph,
            reversed_flow_path=reversed_flow_path,
            control_flow_steps_successors=control_flow_steps_successors,
        )
        max_conditions_steps_in_execution_branch = max(
            condition_steps, max_conditions_steps_in_execution_branch
        )
    if len(control_flow_steps_successors) > max_conditions_steps_in_execution_branch:
        raise ConditionalBranchesCollapseError(
            public_message=f"In execution graph, detected collision of branches that originate "
            f"in different flow-control steps. When using flow control step you create "
            f"a sub-workflow, which cannot reach any step from the outside.",
            context="workflow_compilation | execution_graph_construction",
        )
    for condition_step, potential_next_steps in control_flow_steps_successors.items():
        if len(potential_next_steps) > 1:
            raise ConditionalBranchesCollapseError(
                public_message=f"In execution graph, in flow-control step: {condition_step} there are originated "
                f"workflow branches that clashes together.",
                context="workflow_compilation | execution_graph_construction",
            )


def construct_path_to_step_through_selected_parent(
    graph: nx.DiGraph,
    topological_order: List[str],
    parent_step: str,
    step: str,
) -> List[str]:
    normal_flow_path_nodes = nx.descendants(graph, parent_step)
    normal_flow_path_nodes.add(parent_step)
    normal_flow_path_nodes.add(step)
    return [n for n in topological_order if n in normal_flow_path_nodes]


def denote_flow_control_steps_successors_in_normal_flow(
    reversed_steps_graph: nx.DiGraph,
    reversed_flow_path: List[str],
    control_flow_steps_successors: Dict[str, Set[str]],
) -> Tuple[Dict[str, Set[str]], int]:
    conditions_steps = 0
    if not reversed_flow_path:
        return control_flow_steps_successors, conditions_steps
    previous_node = reversed_flow_path[0]
    for node in reversed_flow_path[1:]:
        if is_flow_control_step(execution_graph=reversed_steps_graph, node=node):
            control_flow_steps_successors[node].add(previous_node)
            conditions_steps += 1
        previous_node = node
    return control_flow_steps_successors, conditions_steps
