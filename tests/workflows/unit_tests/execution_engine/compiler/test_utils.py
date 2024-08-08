from typing import Any, Set
from unittest.mock import MagicMock

import networkx as nx
import pytest

from inference.core.workflows.core_steps.models.roboflow.object_detection import (
    v1 as object_detection_version_1,
)
from inference.core.workflows.core_steps.transformations.dynamic_crop import (
    v1 as dynamic_crop_version_1,
)
from inference.core.workflows.execution_engine.entities.base import (
    JsonField,
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    InputNode,
    NodeCategory,
    OutputNode,
    StepNode,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    construct_input_selector,
    get_input_parameters_selectors,
    get_last_chunk_of_selector,
    get_nodes_of_specific_category,
    get_output_selectors,
    get_step_selector_from_its_output,
    get_steps_selectors,
    is_flow_control_step,
    is_input_selector,
    is_selector,
    is_step_output_selector,
    is_step_selector,
)


def test_construct_input_selector() -> None:
    # when
    result = construct_input_selector(input_name="some")

    # then
    assert result == "$inputs.some"


@pytest.mark.parametrize(
    "expected_kind, expected_result",
    [
        (NodeCategory.INPUT_NODE, {"some"}),
        (NodeCategory.STEP_NODE, {"step_one", "step_two"}),
        (NodeCategory.OUTPUT_NODE, {"other"}),
    ],
)
def test_get_nodes_of_specific_kind(
    expected_kind: NodeCategory, expected_result: Set[str]
) -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "some",
        node_compilation_output=InputNode(
            node_category=NodeCategory.INPUT_NODE,
            name="some",
            selector=f"$inputs.some",
            data_lineage=[],
            input_manifest=WorkflowParameter(
                type="WorkflowParameter",
                name="some",
            ),
        ),
    )
    graph.add_node(
        "step_one",
        node_compilation_output=StepNode(
            node_category=NodeCategory.STEP_NODE,
            name="step_one",
            selector="$steps.selector",
            data_lineage=["Batch[image]"],
            step_manifest=MagicMock(),
        ),
    )
    graph.add_node(
        "step_two",
        node_compilation_output=StepNode(
            node_category=NodeCategory.STEP_NODE,
            name="step_two",
            selector="$steps.selector",
            data_lineage=["Batch[image]"],
            step_manifest=MagicMock(),
        ),
    )
    graph.add_node(
        "other",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name="step_two",
            selector="$steps.selector",
            data_lineage=["Batch[image]"],
            output_manifest=MagicMock(),
        ),
    )

    # when
    result = get_nodes_of_specific_category(
        execution_graph=graph, category=expected_kind
    )

    # then
    assert result == expected_result, "Only expected steps are to be extracted"


def test_get_step_selector_from_its_output() -> None:
    # when
    result = get_step_selector_from_its_output(
        step_output_selector="$steps.detection.predictions"
    )

    # then
    assert result == "$steps.detection"


def test_is_step_output_selector_when_step_output_provided() -> None:
    # when
    result = is_step_output_selector(selector_or_value="$steps.some.parent_id")

    # then
    assert result is True


def test_is_step_output_selector_when_step_selector_provided() -> None:
    # when
    result = is_step_output_selector(selector_or_value="$steps.some")

    # then
    assert result is False


def test_is_step_output_selector_when_input_selector_provided() -> None:
    # when
    result = is_step_output_selector(selector_or_value="$inputs.some")

    # then
    assert result is False


def test_is_step_output_selector_when_specific_value_provided() -> None:
    # when
    result = is_step_output_selector(selector_or_value=3)

    # then
    assert result is False


def test_is_input_selector_when_specific_value_given() -> None:
    # when
    result = is_input_selector(selector_or_value=4)

    # then
    assert result is False


def test_is_input_selector_when_step_selector_given() -> None:
    # when
    result = is_input_selector(selector_or_value="$steps.some")

    # then
    assert result is False


def test_is_input_selector_when_input_selector_given() -> None:
    # when
    result = is_input_selector(selector_or_value="$inputs.some")

    # then
    assert result is True


def test_get_output_names() -> None:
    # given
    outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
        JsonField(type="JsonField", name="other", selector="$steps.b.predictions"),
    ]

    # when
    result = get_output_selectors(outputs=outputs)

    # then
    assert result == {
        "$outputs.some",
        "$outputs.other",
    }, "$outputs. prefix must be added to each output name"


def test_get_steps_selectors() -> None:
    # given
    steps = [
        dynamic_crop_version_1.BlockManifest(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect_2.predictions",
        ),
        object_detection_version_1.BlockManifest(
            type="ObjectDetectionModel",
            name="my_model",
            image="$inputs.image",
            model_id="some/1",
            confidence=0.3,
        ),
    ]

    # when
    result = get_steps_selectors(steps=steps)

    # then
    assert result == {
        "$steps.my_crop",
        "$steps.my_model",
    }, "Expected to find step selector with name of each step prefixed with $steps."


def test_get_input_parameters_selectors() -> None:
    # given
    inputs = [
        WorkflowImage(type="WorkflowImage", name="image"),
        WorkflowParameter(type="WorkflowParameter", name="x"),
        WorkflowParameter(type="WorkflowParameter", name="y"),
    ]

    # when
    result = get_input_parameters_selectors(inputs=inputs)

    # then
    assert result == {
        "$inputs.image",
        "$inputs.x",
        "$inputs.y",
    }, "Expected that each input will have its own selector with its name starting from $inputs."


def test_is_step_selector_when_not_a_selector_given() -> None:
    # when
    result = is_step_selector(selector_or_value="some")

    # then
    assert result is False


def test_is_step_selector_when_input_selector_given() -> None:
    # when
    result = is_step_selector(selector_or_value="$inputs.some")

    # then
    assert result is False


def test_is_step_selector_when_step_output_selector_given() -> None:
    # when
    result = is_step_selector(selector_or_value="$steps.some.output")

    # then
    assert result is False


def test_is_step_selector_when_step_selector_given() -> None:
    # when
    result = is_step_selector(selector_or_value="$steps.some")

    # then
    assert result is True


def test_get_last_chunk_of_selector() -> None:
    # when
    result = get_last_chunk_of_selector(selector="$some.value")

    # then
    assert result == "value"


def test_is_flow_control_step_when_not_a_step_node_given() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "some",
        node_compilation_output=InputNode(
            node_category=NodeCategory.INPUT_NODE,
            name="some",
            selector=f"$inputs.some",
            data_lineage=[],
            input_manifest=WorkflowParameter(
                type="WorkflowParameter",
                name="some",
            ),
        ),
    )

    # when
    result = is_flow_control_step(execution_graph=graph, node="some")

    # then
    assert result is False


def test_is_flow_control_step_when_step_node_given_but_not_control_flow() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "some",
        node_compilation_output=StepNode(
            node_category=NodeCategory.STEP_NODE,
            name="some",
            selector=f"$inputs.some",
            data_lineage=[],
            step_manifest=MagicMock(),
        ),
    )

    # when
    result = is_flow_control_step(execution_graph=graph, node="some")

    # then
    assert result is False


def test_is_flow_control_step_when_control_flow_step_given() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "some",
        node_compilation_output=StepNode(
            node_category=NodeCategory.STEP_NODE,
            name="some",
            selector=f"$inputs.some",
            data_lineage=[],
            step_manifest=MagicMock(),
            child_execution_branches={"branch_a": "$steps.a"},
        ),
    )

    # when
    result = is_flow_control_step(execution_graph=graph, node="some")

    # then
    assert result is True


@pytest.mark.parametrize("selector", ["$inputs.a", "$steps.step", "$steps.step.output"])
def test_is_selector_when_selector_given(selector: str) -> None:
    # when
    result = is_selector(selector_or_value=selector)

    # then
    assert result is True


@pytest.mark.parametrize("value", ["some", 1, [1, 2, 3]])
def test_is_selector_when_not_a_selector_given(value: Any) -> None:
    # when
    result = is_selector(selector_or_value=value)

    # then
    assert result is False
