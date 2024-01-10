from typing import Any, Set, Tuple

from networkx import DiGraph

from inference.enterprise.deployments.entities.deployment_specs import (
    DeploymentSpecV1,
    StepType,
)
from inference.enterprise.deployments.entities.steps import is_selector


def get_input_parameters_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {
        construct_input_selector(input_name=input_definition.name)
        for input_definition in deployment_spec.inputs
    }


def construct_input_selector(input_name: str) -> str:
    return f"$inputs.{input_name}"


def get_steps_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {
        construct_step_selector(step_name=step.name) for step in deployment_spec.steps
    }


def construct_step_selector(step_name: str) -> str:
    return f"$steps.{step_name}"


def get_steps_input_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    result = set()
    for step in deployment_spec.steps:
        result.update(get_step_input_selectors(step=step))
    return result


def get_step_input_selectors(step: StepType) -> Set[str]:
    return {
        getattr(step, step_input_name)
        for step_input_name in step.get_input_names()
        if is_selector(selector_or_value=getattr(step, step_input_name))
    }


def get_steps_output_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    result = set()
    for step in deployment_spec.steps:
        for output_name in step.get_output_names():
            result.add(f"$steps.{step.name}.{output_name}")
    return result


def get_output_names(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {
        construct_output_name(name=output.name) for output in deployment_spec.outputs
    }


def construct_output_name(name: str) -> str:
    return f"$outputs.{name}"


def get_output_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {output.selector for output in deployment_spec.outputs}


def is_input_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return selector_or_value.startswith("$inputs")


def get_last_selector_chunk(selector: str) -> str:
    return selector.split(".")[-1]


def is_step_output_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return (
        selector_or_value.startswith("$steps.")
        and len(selector_or_value.split(".")) == 3
    )


def get_step_selector_from_its_output(step_output_selector: str) -> str:
    return ".".join(step_output_selector.split(".")[:2])


def get_nodes_of_specific_kind(execution_graph: DiGraph, kind: str) -> Set[str]:
    return {
        node[0]
        for node in execution_graph.nodes(data=True)
        if node[1].get("kind") == kind
    }


def get_selector_chunks(selector: str) -> Tuple[str, ...]:
    return tuple(selector.split(".")[1:])


def is_condition_step(execution_graph: DiGraph, node: str) -> bool:
    return execution_graph.nodes[node]["definition"].type == "Condition"
