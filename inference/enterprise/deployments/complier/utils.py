from typing import Any, Set, Tuple

from networkx import DiGraph

from inference.enterprise.deployments.entities.deployment_specs import (
    DeploymentSpecV1,
    StepType,
)
from inference.enterprise.deployments.entities.steps import ConditionSpecs


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
    return {
        step_input
        for step in deployment_spec.steps
        for step_input in step.inputs.values()
        if is_selector(selector_or_value=step_input)
    }


def get_step_input_selectors(step: StepType) -> Set[str]:
    return {
        step_input
        for step_input in step.inputs.values()
        if is_selector(selector_or_value=step_input)
    }


def get_steps_output_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    result = set()
    for step in deployment_spec.steps:
        if step.type == "CVModel":
            result.add(f"$steps.{step.name}.predictions")
            result.add(f"$steps.{step.name}.top")  # TODO: put more realistic outputs
        if step.type == "Crop":
            result.add(f"$steps.{step.name}.crops")
    return result


def get_output_names(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {output.name for output in deployment_spec.outputs}


def get_output_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {output.selector for output in deployment_spec.outputs}


def get_selectors_from_condition_specs(condition_specs: ConditionSpecs) -> Set[str]:
    result = set()
    if issubclass(type(condition_specs.left), ConditionSpecs):
        result = result | get_selectors_from_condition_specs(
            condition_specs=condition_specs.left
        )
    elif is_selector(condition_specs.left):
        result.add(condition_specs.left)
    if issubclass(type(condition_specs.right), ConditionSpecs):
        result = result | get_selectors_from_condition_specs(
            condition_specs=condition_specs.right
        )
    elif is_selector(condition_specs.right):
        result.add(condition_specs.right)
    return result


def is_step_output_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return (
        selector_or_value.startswith("$steps.")
        and len(selector_or_value.split(".")) == 3
    )


def is_selector(selector_or_value: Any) -> bool:
    if not issubclass(type(selector_or_value), str):
        return False
    return selector_or_value.startswith("$")


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


def is_cv_model_step(execution_graph: DiGraph, node: str) -> bool:
    return execution_graph.nodes[node]["definition"].type == "CVModel"


def is_crop_step(execution_graph: DiGraph, node: str) -> bool:
    return execution_graph.nodes[node]["definition"].type == "Crop"
