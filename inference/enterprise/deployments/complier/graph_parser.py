from collections import defaultdict
from typing import Set

import networkx as nx
from networkx import DiGraph

from inference.enterprise.deployments.complier.utils import (
    construct_input_selector,
    construct_step_selector,
    get_step_input_selectors,
    get_step_selector_from_its_output,
    is_step_output_selector, get_nodes_of_specific_kind,
)
from inference.enterprise.deployments.constants import INPUT_NODE_KIND, STEP_NODE_KIND, OUTPUT_NODE_KIND
from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1
from inference.enterprise.deployments.errors import (
    NodesNotReachingOutputError,
    NotAcyclicGraphError, AmbiguousPathDetected,
)


def construct_execution_graph(deployment_spec: DeploymentSpecV1) -> DiGraph:
    execution_graph = nx.DiGraph()
    execution_graph = add_input_nodes_for_graph(
        deployment_spec=deployment_spec, execution_graph=execution_graph
    )
    execution_graph = add_steps_nodes_for_graph(
        deployment_spec=deployment_spec, execution_graph=execution_graph
    )
    execution_graph = add_output_nodes_for_graph(
        deployment_spec=deployment_spec, execution_graph=execution_graph
    )
    execution_graph = add_steps_edges(
        deployment_spec=deployment_spec, execution_graph=execution_graph
    )
    execution_graph = add_edges_for_outputs(
        deployment_spec=deployment_spec, execution_graph=execution_graph
    )
    if not nx.is_directed_acyclic_graph(execution_graph):
        raise NotAcyclicGraphError(f"Detected cycle in execution graph.")
    verify_each_node_reachable_from_at_least_one_output(execution_graph=execution_graph)
    verify_each_node_step_has_at_most_one_parent_being_step(execution_graph=execution_graph)
    return execution_graph


def add_input_nodes_for_graph(
    deployment_spec: DeploymentSpecV1,
    execution_graph: DiGraph,
) -> DiGraph:
    for input_spec in deployment_spec.inputs:
        input_selector = construct_input_selector(input_name=input_spec.name)
        execution_graph.add_node(
            input_selector,
            kind=INPUT_NODE_KIND,
            definition=input_spec,
        )
    return execution_graph


def add_steps_nodes_for_graph(
    deployment_spec: DeploymentSpecV1,
    execution_graph: DiGraph,
) -> DiGraph:
    for step in deployment_spec.steps:
        step_selector = construct_step_selector(step_name=step.name)
        execution_graph.add_node(
            step_selector,
            kind=STEP_NODE_KIND,
            definition=step,
        )
    return execution_graph


def add_output_nodes_for_graph(
    deployment_spec: DeploymentSpecV1,
    execution_graph: DiGraph,
) -> DiGraph:
    for output_spec in deployment_spec.outputs:
        execution_graph.add_node(
            output_spec.name,
            kind=OUTPUT_NODE_KIND,
            definition=output_spec,
        )
    return execution_graph


def add_steps_edges(
    deployment_spec: DeploymentSpecV1,
    execution_graph: DiGraph,
) -> DiGraph:
    for step in deployment_spec.steps:
        input_selectors = get_step_input_selectors(step=step)
        step_selector = construct_step_selector(step_name=step.name)
        execution_graph = add_edges_for_step_inputs(
            execution_graph=execution_graph,
            input_selectors=input_selectors,
            step_selector=step_selector,
        )
        if step.type == "Condition":
            execution_graph.add_edge(step_selector, step.step_if_true)
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
        execution_graph.add_edge(input_selector, step_selector)
    return execution_graph


def add_edges_for_outputs(
    deployment_spec: DeploymentSpecV1,
    execution_graph: DiGraph,
) -> DiGraph:
    for output in deployment_spec.outputs:
        output_selector = output.selector
        if is_step_output_selector(selector_or_value=output_selector):
            output_selector = get_step_selector_from_its_output(
                step_output_selector=output_selector
            )
        execution_graph.add_edge(output_selector, output.name)
    return execution_graph


def verify_each_node_reachable_from_at_least_one_output(
    execution_graph: DiGraph,
) -> None:
    all_nodes = set(execution_graph.nodes())
    output_nodes = get_nodes_of_specific_kind(execution_graph=execution_graph, kind=OUTPUT_NODE_KIND)
    nodes_reaching_output = get_nodes_that_reach_pointed_ones(
        execution_graph=execution_graph,
        pointed_nodes=output_nodes,
    )
    nodes_not_reaching_output = all_nodes.difference(nodes_reaching_output)
    if len(nodes_not_reaching_output) > 0:
        raise NodesNotReachingOutputError(
            f"Detected {len(nodes_not_reaching_output)} nodes not reaching any of output node:"
            f"{nodes_not_reaching_output}."
        )


def get_nodes_that_reach_pointed_ones(
    execution_graph: DiGraph,
    pointed_nodes: Set[str],
) -> Set[str]:
    result = set()
    reversed_graph = execution_graph.reverse()
    for pointed_node in pointed_nodes:
        nodes_reaching_pointed_one = list(
            nx.dfs_postorder_nodes(reversed_graph, pointed_node)
        )
        result.update(nodes_reaching_pointed_one)
    return result


def verify_each_node_step_has_at_most_one_parent_being_step(execution_graph: DiGraph) -> None:
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
    execution path - for instance if D requires an output from C and D - this is allowed)
    """
    steps_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=STEP_NODE_KIND,
    )
    edges_of_steps_nodes = [
        edge for edge in execution_graph.edges()
        if edge[0] in steps_nodes or edge[1] in steps_nodes
    ]
    steps_parents = defaultdict(set)
    for edge in edges_of_steps_nodes:
        parent, child = edge
        if parent not in steps_nodes or child not in steps_nodes:
            continue
        steps_parents[child].add(parent)
    steps_with_more_than_one_parent = [key for key, value in steps_parents.items() if len(value) > 1]
    if len(steps_with_more_than_one_parent) == 0:
        return None
    
        # raise AmbiguousPathDetected(
        #     f"Detected steps that require more than one parent: {steps_with_more_than_one_parent}"
        # )


