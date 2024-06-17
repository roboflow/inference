from typing import Any

import networkx as nx
import pytest

from inference.core.workflows.constants import INPUT_NODE_KIND, STEP_NODE_KIND
from inference.core.workflows.core_steps.common.operators import Operator
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    IsTrue,
    StatementGroup,
    StaticOperand,
    UnaryStatement,
)
from inference.core.workflows.core_steps.flow_control import condition
from inference.core.workflows.core_steps.models.roboflow import object_detection
from inference.core.workflows.core_steps.transformations import crop
from inference.core.workflows.entities.base import (
    JsonField,
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.compiler.utils import (
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


def test_get_nodes_of_specific_kind() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "one",
        kind=STEP_NODE_KIND,
        definition=condition.BlockManifest(
            type="Condition",
            name="one",
            condition_statement=StatementGroup(
                type="StatementGroup",
                statements=[
                    UnaryStatement(
                        type="UnaryStatement",
                        operand=StaticOperand(
                            type="StaticOperand",
                            value=True,
                        ),
                        operator=IsTrue(type="(Boolean) is True"),
                    )
                ],
            ),
            evaluation_parameters={},
            step_if_true="$steps.a",
            step_if_false="$steps.b",
        ),
    )
    graph.add_node(
        "two",
        kind=STEP_NODE_KIND,
        definition=crop.BlockManifest(
            type="Crop",
            name="two",
            image="$inputs.image",
            detections="$steps.detection.predictions",
        ),
    )
    graph.add_node(
        "three",
        kind=INPUT_NODE_KIND,
        definition=WorkflowParameter(type="WorkflowParameter", name="three"),
    )

    # when
    result = get_nodes_of_specific_category(execution_graph=graph, category=STEP_NODE_KIND)

    # then
    assert result == {
        "one",
        "two",
    }, "Only nodes `one` and `two` are defined with step kind"


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
        crop.BlockManifest(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect_2.predictions",
        ),
        object_detection.BlockManifest(
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


def test_is_flow_control_step_when_flow_control_step_given() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("some")

    # when
    result = is_flow_control_step(execution_graph=graph, node="some")

    # then
    assert result is False


def test_is_flow_control_step_when_not_a_flow_control_step_given() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("some", flow_control_node=True)

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
