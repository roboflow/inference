from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx import DiGraph

from inference.enterprise.workflows.complier.utils import (
    construct_input_selector,
    construct_output_name,
    construct_step_selector,
    get_nodes_of_specific_kind,
    get_step_input_selectors,
    get_step_selector_from_its_output,
    is_condition_step,
    is_step_output_selector,
)
from inference.enterprise.workflows.constants import (
    INPUT_NODE_KIND,
    OUTPUT_NODE_KIND,
    STEP_NODE_KIND,
)
from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.validators import is_selector
from inference.enterprise.workflows.entities.workflows_specification import (
    InputType,
    StepType,
    WorkflowSpecificationV1,
)
from inference.enterprise.workflows.errors import (
    AmbiguousPathDetected,
    NodesNotReachingOutputError,
    NotAcyclicGraphError,
    SelectorToUndefinedNodeError,
)


def prepare_execution_graph(workflow_specification: WorkflowSpecificationV1) -> DiGraph:
    execution_graph = construct_graph(workflow_specification=workflow_specification)
    if not nx.is_directed_acyclic_graph(execution_graph):
        raise NotAcyclicGraphError(f"Detected cycle in execution graph.")
    verify_each_node_reach_at_least_one_output(execution_graph=execution_graph)
    verify_each_node_step_has_parent_in_the_same_branch(execution_graph=execution_graph)
    verify_that_steps_are_connected_with_compatible_inputs(
        execution_graph=execution_graph
    )
    return execution_graph


def construct_graph(workflow_specification: WorkflowSpecificationV1) -> DiGraph:
    execution_graph = nx.DiGraph()
    execution_graph = add_input_nodes_for_graph(
        inputs=workflow_specification.inputs, execution_graph=execution_graph
    )
    execution_graph = add_steps_nodes_for_graph(
        steps=workflow_specification.steps, execution_graph=execution_graph
    )
    execution_graph = add_output_nodes_for_graph(
        outputs=workflow_specification.outputs, execution_graph=execution_graph
    )
    execution_graph = add_steps_edges(
        workflow_specification=workflow_specification, execution_graph=execution_graph
    )
    return add_edges_for_outputs(
        workflow_specification=workflow_specification, execution_graph=execution_graph
    )


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
        )
    return execution_graph


def add_steps_nodes_for_graph(
    steps: List[StepType],
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
    workflow_specification: WorkflowSpecificationV1,
    execution_graph: DiGraph,
) -> DiGraph:
    for step in workflow_specification.steps:
        input_selectors = get_step_input_selectors(step=step)
        step_selector = construct_step_selector(step_name=step.name)
        execution_graph = add_edges_for_step_inputs(
            execution_graph=execution_graph,
            input_selectors=input_selectors,
            step_selector=step_selector,
        )
        if step.type == "Condition":
            verify_edge_is_created_between_existing_nodes(
                execution_graph=execution_graph,
                start=step_selector,
                end=step.step_if_true,
            )
            execution_graph.add_edge(step_selector, step.step_if_true)
            verify_edge_is_created_between_existing_nodes(
                execution_graph=execution_graph,
                start=step_selector,
                end=step.step_if_false,
            )
            execution_graph.add_edge(step_selector, step.step_if_false)
    return execution_graph


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
    workflow_specification: WorkflowSpecificationV1,
    execution_graph: DiGraph,
) -> DiGraph:
    for output in workflow_specification.outputs:
        output_selector = output.selector
        if is_step_output_selector(selector_or_value=output_selector):
            output_selector = get_step_selector_from_its_output(
                step_output_selector=output_selector
            )
        output_name = construct_output_name(name=output.name)
        verify_edge_is_created_between_existing_nodes(
            execution_graph=execution_graph,
            start=output_selector,
            end=output_name,
        )
        execution_graph.add_edge(output_selector, output_name)
    return execution_graph


def verify_edge_is_created_between_existing_nodes(
    execution_graph: DiGraph,
    start: str,
    end: str,
) -> None:
    if not execution_graph.has_node(start):
        raise SelectorToUndefinedNodeError(
            f"Graph definition contains selector {start} that points to not defined element."
        )
    if not execution_graph.has_node(end):
        raise SelectorToUndefinedNodeError(
            f"Graph definition contains selector {end} that points to not defined element."
        )


def verify_each_node_reach_at_least_one_output(
    execution_graph: DiGraph,
) -> None:
    all_nodes = set(execution_graph.nodes())
    output_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=OUTPUT_NODE_KIND
    )
    nodes_without_outputs = get_nodes_that_do_not_produce_outputs(
        execution_graph=execution_graph
    )
    nodes_that_must_be_reached = output_nodes.union(nodes_without_outputs)
    nodes_reaching_output = (
        get_nodes_that_are_reachable_from_pointed_ones_in_reversed_graph(
            execution_graph=execution_graph,
            pointed_nodes=nodes_that_must_be_reached,
        )
    )
    nodes_not_reaching_output = all_nodes.difference(nodes_reaching_output)
    if len(nodes_not_reaching_output) > 0:
        raise NodesNotReachingOutputError(
            f"Detected {len(nodes_not_reaching_output)} nodes not reaching any of output node:"
            f"{nodes_not_reaching_output}."
        )


def get_nodes_that_do_not_produce_outputs(execution_graph: DiGraph) -> Set[str]:
    # assumption is that nodes without outputs will produce some side effect and shall be
    # treated as output nodes while checking if there is no dangling steps in graph
    step_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=STEP_NODE_KIND
    )
    return {
        step_node
        for step_node in step_nodes
        if len(execution_graph.nodes[step_node]["definition"].get_output_names()) == 0
    }


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
    words - the problem emerges if a node of kind STEP has a parent (node from which
    it can be achieved) of kind STEP and this parent is in a different branch (point out that
    we allow for a single step to have multiple steps as input, but they must be at the same
    execution path - for instance if D requires an output from C and B - this is allowed).
    Additionally, we must prevent situation when outcomes of branches started by two or more
    condition steps merge with each other, as condition eval may result in contradictory
    execution (2).


    We need to detect that situation upfront, such that we can raise error of ambiguous execution path
    rather than run time-consuming computations that will end up in error.

    To detect problem, first we detect steps with more than one parent step.
    From those steps we trace what sequence of steps would lead to execution of problematic one.
    For each problematic node we take its parent nodes. Then, we analyse paths from
    those parent nodes in reversed topological order (from those nodes towards entry nodes
    of execution graph). While our analysis, on each path we denote `Condition` steps and
    result of condition evaluation that must have been observed in runtime, to reach
    the problematic node while graph execution in normal direction. If we detect that
    for any `Condition` step we would need to output both True and False (more than one registered
    next step of `Condition` step) - we raise error.
    To detect problem (2) - we only let number of different condition steps considered be the number of
    max condition steps in a single path from origin to parent of problematic step.

    Beware that the latter part of algorithm has quite bad time complexity in general case.
    Worst part of algorithm runs at O(V^4) - at least taking coarse, worst-case estimations.
    In fact, there is not so bad:
    * The number of step nodes with multiple parents that we loop over in main loop, reduces the number of
    steps we iterate through in inner loops, as we are dealing with DAG (with quite limited amount of edges)
    and for each multi-parent node takes at least two other nodes (to construct a suspicious group) -
    so expected number of iterations in main loop is low - let's say 1-3 for a real graph.
    * for any reasonable execution graph, the complexity should be acceptable.
    """
    steps_with_more_than_one_parent = detect_steps_with_more_than_one_parent_step(
        execution_graph=execution_graph
    )  # O(V+E)
    if len(steps_with_more_than_one_parent) == 0:
        return None
    reversed_steps_graph = construct_reversed_steps_graph(
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


def construct_reversed_steps_graph(execution_graph: DiGraph) -> DiGraph:
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
    condition_steps_successors = defaultdict(set)
    max_conditions_steps = 0
    for normal_flow_predecessor in reversed_steps_graph.successors(step):  # O(V)
        reversed_flow_path = (
            construct_reversed_path_to_multi_parent_step_parent(  # O(E) -> O(V^2)
                reversed_steps_graph=reversed_steps_graph,
                reversed_topological_order=reversed_topological_order,
                parent_step=normal_flow_predecessor,
                step=step,
            )
        )
        (
            condition_steps_successors,
            condition_steps,
        ) = denote_condition_steps_successors_in_normal_flow(  # O(V)
            reversed_steps_graph=reversed_steps_graph,
            reversed_flow_path=reversed_flow_path,
            condition_steps_successors=condition_steps_successors,
        )
        max_conditions_steps = max(condition_steps, max_conditions_steps)
    if len(condition_steps_successors) > max_conditions_steps:
        raise AmbiguousPathDetected(
            f"In execution graph, detected collision of branches that originate in different condition steps."
        )
    for condition_step, potential_next_steps in condition_steps_successors.items():
        if len(potential_next_steps) > 1:
            raise AmbiguousPathDetected(
                f"In execution graph, condition step: {condition_step} creates ambiguous execution paths."
            )


def construct_reversed_path_to_multi_parent_step_parent(
    reversed_steps_graph: nx.DiGraph,
    reversed_topological_order: List[str],
    parent_step: str,
    step: str,
) -> List[str]:
    normal_flow_path_nodes = nx.descendants(reversed_steps_graph, parent_step)
    normal_flow_path_nodes.add(parent_step)
    normal_flow_path_nodes.add(step)
    return [n for n in reversed_topological_order if n in normal_flow_path_nodes]


def denote_condition_steps_successors_in_normal_flow(
    reversed_steps_graph: nx.DiGraph,
    reversed_flow_path: List[str],
    condition_steps_successors: Dict[str, Set[str]],
) -> Tuple[Dict[str, Set[str]], int]:
    conditions_steps = 0
    if len(reversed_flow_path) == 0:
        return condition_steps_successors, conditions_steps
    previous_node = reversed_flow_path[0]
    for node in reversed_flow_path[1:]:
        if is_condition_step(execution_graph=reversed_steps_graph, node=node):
            condition_steps_successors[node].add(previous_node)
            conditions_steps += 1
        previous_node = node
    return condition_steps_successors, conditions_steps


def verify_that_steps_are_connected_with_compatible_inputs(
    execution_graph: nx.DiGraph,
) -> None:
    steps_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=STEP_NODE_KIND,
    )
    for step in steps_nodes:
        verify_step_inputs_selectors(step=step, execution_graph=execution_graph)


def verify_step_inputs_selectors(step: str, execution_graph: nx.DiGraph) -> None:
    step_definition = execution_graph.nodes[step]["definition"]
    all_inputs = step_definition.get_input_names()
    for input_step in all_inputs:
        input_selector_or_value = getattr(step_definition, input_step)
        if issubclass(type(input_selector_or_value), list):
            for idx, single_selector_or_value in enumerate(input_selector_or_value):
                validate_step_definition_input(
                    step_definition=step_definition,
                    input_name=input_step,
                    execution_graph=execution_graph,
                    input_selector_or_value=single_selector_or_value,
                    index=idx,
                )
        else:
            validate_step_definition_input(
                step_definition=step_definition,
                input_name=input_step,
                execution_graph=execution_graph,
                input_selector_or_value=input_selector_or_value,
            )


def validate_step_definition_input(
    step_definition: StepInterface,
    input_name: str,
    execution_graph: nx.DiGraph,
    input_selector_or_value: Any,
    index: Optional[int] = None,
) -> None:
    if not is_selector(selector_or_value=input_selector_or_value):
        return None
    if is_step_output_selector(selector_or_value=input_selector_or_value):
        input_selector_or_value = get_step_selector_from_its_output(
            step_output_selector=input_selector_or_value
        )
    input_node_definition = execution_graph.nodes[input_selector_or_value]["definition"]
    step_definition.validate_field_selector(
        field_name=input_name,
        input_step=input_node_definition,
        index=index,
    )
