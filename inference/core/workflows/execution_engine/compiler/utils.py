from typing import Any, List, Set

from networkx import DiGraph

from inference.core.workflows.entities.base import InputType, JsonField
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

FLOW_CONTROL_NODE_KEY = "flow_control_node"


def get_input_parameters_selectors(inputs: List[InputType]) -> Set[str]:
    return {
        construct_input_selector(input_name=input_definition.name)
        for input_definition in inputs
    }


def construct_input_selector(input_name: str) -> str:
    return f"$inputs.{input_name}"


def get_steps_selectors(steps: List[WorkflowBlockManifest]) -> Set[str]:
    return {construct_step_selector(step_name=step.name) for step in steps}


def construct_step_selector(step_name: str) -> str:
    return f"$steps.{step_name}"


def get_output_names(outputs: List[JsonField]) -> Set[str]:
    return {construct_output_name(name=output.name) for output in outputs}


def construct_output_name(name: str) -> str:
    return f"$outputs.{name}"


def is_input_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return selector_or_value.startswith("$inputs")


def is_step_selector(selector_or_value: Any) -> bool:
    if not is_selector(selector_or_value=selector_or_value):
        return False
    return (
        selector_or_value.startswith("$steps.")
        and len(selector_or_value.split(".")) == 2
    )


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


def get_last_chunk_of_selector(selector: str) -> str:
    return selector.split(".")[-1]


def is_flow_control_step(execution_graph: DiGraph, node: str) -> bool:
    return execution_graph.nodes[node].get(FLOW_CONTROL_NODE_KEY, False)


def is_selector(selector_or_value: Any) -> bool:
    return str(selector_or_value).startswith("$")
