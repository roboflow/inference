from typing import Any, List, Set, Type, TypeVar

from networkx import DiGraph

from inference.core.workflows.errors import AssumptionError
from inference.core.workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from inference.core.workflows.execution_engine.entities.base import InputType, JsonField
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ExecutionGraphNode,
    NodeCategory,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

NodeTypeVar = TypeVar("NodeTypeVar", bound=ExecutionGraphNode)


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


def get_output_selectors(outputs: List[JsonField]) -> Set[str]:
    return {construct_output_selector(name=output.name) for output in outputs}


def construct_output_selector(name: str) -> str:
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


def get_nodes_of_specific_category(
    execution_graph: DiGraph, category: NodeCategory
) -> Set[str]:
    return {
        node[0]
        for node in execution_graph.nodes(data=True)
        if node[1][NODE_COMPILATION_OUTPUT_PROPERTY].node_category is category
    }


def get_last_chunk_of_selector(selector: str) -> str:
    return selector.split(".")[-1]


def is_flow_control_step(execution_graph: DiGraph, node: str) -> bool:
    if not is_step_node(execution_graph=execution_graph, node=node):
        return False
    return execution_graph.nodes[node][NODE_COMPILATION_OUTPUT_PROPERTY].controls_flow()


def is_input_node(execution_graph: DiGraph, node: str) -> bool:
    return (
        execution_graph.nodes[node][NODE_COMPILATION_OUTPUT_PROPERTY].node_category
        is NodeCategory.INPUT_NODE
    )


def is_step_node(execution_graph: DiGraph, node: str) -> bool:
    return (
        execution_graph.nodes[node][NODE_COMPILATION_OUTPUT_PROPERTY].node_category
        is NodeCategory.STEP_NODE
    )


def is_output_node(execution_graph: DiGraph, node: str) -> bool:
    return (
        execution_graph.nodes[node][NODE_COMPILATION_OUTPUT_PROPERTY].node_category
        is NodeCategory.OUTPUT_NODE
    )


def is_selector(selector_or_value: Any) -> bool:
    return str(selector_or_value).startswith("$")


def identify_lineage(lineage: List[str]) -> int:
    return sum(hash(e) for e in lineage)


def node_as(
    execution_graph: DiGraph, node: str, expected_type: Type[NodeTypeVar]
) -> NodeTypeVar:
    if node not in execution_graph.nodes:
        raise AssumptionError(
            public_message=f"Workflow Compiler expected node: {node} to be present in execution graph, "
            f"but condition failed to be met. This is most likely the bug. Contact Roboflow team "
            f"through github issues (https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | retrieving_compiled_node",
        )
    if NODE_COMPILATION_OUTPUT_PROPERTY not in execution_graph.nodes[node]:
        raise AssumptionError(
            public_message=f"Workflow Compiler expected key: {NODE_COMPILATION_OUTPUT_PROPERTY} to be present "
            f"for node {node} in execution graph, "
            f"but condition failed to be met. This is most likely the bug. Contact Roboflow team "
            f"through github issues (https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | retrieving_compiled_node",
        )
    node_data: ExecutionGraphNode = execution_graph.nodes[node][
        NODE_COMPILATION_OUTPUT_PROPERTY
    ]
    if not isinstance(node_data, expected_type):
        node_data_type = type(node_data)
        raise AssumptionError(
            public_message=f"Workflow Compiler expected compilation output for node {node}"
            f"to be {expected_type}, but found: {node_data_type}. "
            f"This is most likely the bug. Contact Roboflow team "
            f"through github issues (https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_compilation | execution_graph_construction | retrieving_compiled_node",
        )
    return node_data
