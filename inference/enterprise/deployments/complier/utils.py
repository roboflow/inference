from typing import Set

from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1
from inference.enterprise.deployments.entities.steps import ConditionSpecs


def get_input_parameters_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {
        f"$inputs.{input_definition.name}"
        for input_definition in deployment_spec.inputs
    }


def get_steps_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {f"$steps.{step.name}" for step in deployment_spec.steps}


def get_steps_input_selectors(deployment_spec: DeploymentSpecV1) -> Set[str]:
    return {
        step_input
        for step in deployment_spec.steps
        for step_input in step.inputs.values()
        if step_input.startswith("$")
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
    elif str(condition_specs.left).startswith("$"):
        result.add(condition_specs.left)
    if issubclass(type(condition_specs.right), ConditionSpecs):
        result = result | get_selectors_from_condition_specs(
            condition_specs=condition_specs.right
        )
    elif str(condition_specs.right).startswith("$"):
        result.add(condition_specs.right)
    return result
