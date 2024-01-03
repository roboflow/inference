from typing import Set

from networkx import Graph

from inference.enterprise.deployments.complier.utils import (
    construct_input_selector,
    construct_step_selector,
    get_step_input_selectors,
    is_step_output_selector,
    get_step_selector_from_its_output,
)
from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1

INPUT_NODE_KIND = "INPUT_NODE"
STEP_NODE_KIND = "STEP_NODE"
OUTPUT_NODE_KIND = "OUTPUT_NODE"


def construct_execution_graph(deployment_spec: DeploymentSpecV1) -> Graph:
    execution_graph = Graph()
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
    return add_edges_for_outputs(
        deployment_spec=deployment_spec, execution_graph=execution_graph
    )


def add_input_nodes_for_graph(
    deployment_spec: DeploymentSpecV1,
    execution_graph: Graph,
) -> Graph:
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
    execution_graph: Graph,
) -> Graph:
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
    execution_graph: Graph,
) -> Graph:
    for output_spec in deployment_spec.outputs:
        execution_graph.add_node(
            output_spec.name,
            kind=OUTPUT_NODE_KIND,
            definition=output_spec,
        )
    return execution_graph


def add_steps_edges(
    deployment_spec: DeploymentSpecV1,
    execution_graph: Graph,
) -> Graph:
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
    execution_graph: Graph,
    input_selectors: Set[str],
    step_selector: str,
) -> Graph:
    for input_selector in input_selectors:
        if is_step_output_selector(selector_or_value=input_selector):
            input_selector = get_step_selector_from_its_output(
                step_output_selector=input_selector
            )
        execution_graph.add_edge(input_selector, step_selector)
    return execution_graph


def add_edges_for_outputs(
    deployment_spec: DeploymentSpecV1,
    execution_graph: Graph,
) -> Graph:
    for output in deployment_spec.outputs:
        output_selector = output.selector
        if is_step_output_selector(selector_or_value=output_selector):
            output_selector = get_step_selector_from_its_output(
                step_output_selector=output_selector
            )
        execution_graph.add_edge(output_selector, output.name)
    return execution_graph
