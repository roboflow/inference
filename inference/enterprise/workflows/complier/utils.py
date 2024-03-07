from typing import Any, List, Set

from networkx import DiGraph

from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.validators import is_selector
from inference.enterprise.workflows.entities.workflows_specification import (
    InputType,
    StepType,
)


def get_input_parameters_selectors(inputs: List[InputType]) -> Set[str]:
    return {
        construct_input_selector(input_name=input_definition.name)
        for input_definition in inputs
    }


def construct_input_selector(input_name: str) -> str:
    return f"$inputs.{input_name}"


def get_steps_selectors(steps: List[StepType]) -> Set[str]:
    return {construct_step_selector(step_name=step.name) for step in steps}


def construct_step_selector(step_name: str) -> str:
    return f"$steps.{step_name}"


def get_steps_input_selectors(steps: List[StepType]) -> Set[str]:
    result = set()
    for step in steps:
        result.update(get_step_input_selectors(step=step))
    return result


def get_step_input_selectors(step: StepType) -> Set[str]:
    result = set()
    for step_input_name in step.get_input_names():
        step_input = getattr(step, step_input_name)
        if not issubclass(type(step_input), list):
            step_input = [step_input]
        for element in step_input:
            if not is_selector(selector_or_value=element):
                continue
            result.add(element)
    return result


def get_steps_output_selectors(steps: List[StepType]) -> Set[str]:
    result = set()
    for step in steps:
        for output_name in step.get_output_names():
            result.add(f"$steps.{step.name}.{output_name}")
            result.add(f"$steps.{step.name}.*")
    return result


def get_output_names(outputs: List[JsonField]) -> Set[str]:
    return {construct_output_name(name=output.name) for output in outputs}


def construct_output_name(name: str) -> str:
    return f"$outputs.{name}"


def get_output_selectors(outputs: List[JsonField]) -> Set[str]:
    return {output.selector for output in outputs}


def is_input_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return selector_or_value.startswith("$inputs")


def construct_selector_pointing_step_output(selector: str, new_output: str) -> str:
    if is_step_output_selector(selector_or_value=selector):
        selector = get_step_selector_from_its_output(step_output_selector=selector)
    return f"{selector}.{new_output}"


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


def is_condition_step(execution_graph: DiGraph, node: str) -> bool:
    return execution_graph.nodes[node]["definition"].type == "Condition"
