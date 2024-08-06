from typing import List
from unittest.mock import MagicMock

from networkx import DiGraph

from inference.core.workflows.execution_engine.entities.base import (
    JsonField,
    OutputDefinition,
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    InputNode,
    NodeCategory,
    OutputNode,
    StepNode,
)


def prepare_execution_graph_for_tests(
    steps_names: List[str],
    are_batch_oriented: List[bool],
    steps_outputs: List[List[OutputDefinition]],
) -> DiGraph:
    execution_graph = DiGraph()
    execution_graph.add_node(
        "$inputs.image",
        node_compilation_output=InputNode(
            node_category=NodeCategory.INPUT_NODE,
            name="image",
            selector="$inputs.image",
            data_lineage=["<workflow_input>"],
            input_manifest=WorkflowImage(type="WorkflowImage", name="image"),
        ),
    )
    execution_graph.add_node(
        "$inputs.param",
        node_compilation_output=InputNode(
            node_category=NodeCategory.INPUT_NODE,
            name="param",
            selector="$inputs.param",
            data_lineage=[],
            input_manifest=WorkflowParameter(type="WorkflowParameter", name="param"),
        ),
    )
    for step_name, is_batch_oriented, outputs in zip(
        steps_names, are_batch_oriented, steps_outputs
    ):
        step_selector = f"$steps.{step_name}"
        data_lineage = [] if not is_batch_oriented else ["<workflow_input>"]
        batch_oriented_parameters = set() if not is_batch_oriented else {"some_param"}
        manifest = MagicMock()
        manifest.get_actual_outputs.return_value = outputs
        manifest.name = step_name
        execution_graph.add_node(
            step_selector,
            node_compilation_output=StepNode(
                node_category=NodeCategory.STEP_NODE,
                name=step_name,
                selector=step_selector,
                data_lineage=data_lineage,
                step_manifest=manifest,
                batch_oriented_parameters=batch_oriented_parameters,
            ),
        )
    execution_graph.add_node(
        "$outputs.some",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name="some",
            selector="$outputs.some",
            data_lineage=["<workflow_input>"],
            output_manifest=JsonField(
                type="JsonField", name="some", selector="$steps.dummy"
            ),
        ),
    )
    return execution_graph
