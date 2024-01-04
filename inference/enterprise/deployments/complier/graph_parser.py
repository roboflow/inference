from typing import Set

import networkx as nx
from networkx import DiGraph

from inference.enterprise.deployments.complier.utils import (
    construct_input_selector,
    construct_step_selector,
    get_step_input_selectors,
    get_step_selector_from_its_output,
    is_step_output_selector,
)
from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1
from inference.enterprise.deployments.errors import (
    NodesNotReachingOutputError,
    NotAcyclicGraphError,
)

INPUT_NODE_KIND = "INPUT_NODE"
STEP_NODE_KIND = "STEP_NODE"
OUTPUT_NODE_KIND = "OUTPUT_NODE"


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
    output_nodes = {
        node[0]
        for node in execution_graph.nodes(data=True)
        if node[1]["kind"] == OUTPUT_NODE_KIND
    }
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
