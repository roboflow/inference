import networkx as nx

from inference.enterprise.workflows.execution_engine.compiler.utils import (
    construct_selector_pointing_step_output,
    get_input_parameters_selectors,
    get_nodes_of_specific_kind,
    get_output_names,
    get_output_selectors,
    get_step_selector_from_its_output,
    get_steps_input_selectors,
    get_steps_output_selectors,
    get_steps_selectors,
    is_condition_step,
    is_input_selector,
    is_step_output_selector,
)
from inference.enterprise.workflows.constants import INPUT_NODE_KIND, STEP_NODE_KIND
from inference.enterprise.workflows.entities.inputs import InferenceImage, InferenceParameter
from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.steps import (
    Condition,
    Crop,
    DetectionsConsensus,
    ObjectDetectionModel,
    Operator,
)


def test_is_condition_step_when_node_type_is_condition() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "some",
        kind=STEP_NODE_KIND,
        definition=Condition(
            type="Condition",
            name="some",
            left=3,
            operator=Operator.EQUAL,
            right=3,
            step_if_true="$steps.a",
            step_if_false="$steps.b",
        ),
    )

    # when
    result = is_condition_step(execution_graph=graph, node="some")

    # then
    assert result is True


def test_is_condition_step_when_node_type_is_not_condition() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "some",
        kind=STEP_NODE_KIND,
        definition=Crop(
            type="Crop",
            name="some",
            image="$inputs.image",
            detections="$steps.detection.predictions",
        ),
    )

    # when
    result = is_condition_step(execution_graph=graph, node="some")

    # then
    assert result is False


def test_get_nodes_of_specific_kind() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "one",
        kind=STEP_NODE_KIND,
        definition=Condition(
            type="Condition",
            name="one",
            left=3,
            operator=Operator.EQUAL,
            right=3,
            step_if_true="$steps.a",
            step_if_false="$steps.b",
        ),
    )
    graph.add_node(
        "two",
        kind=STEP_NODE_KIND,
        definition=Crop(
            type="Crop",
            name="two",
            image="$inputs.image",
            detections="$steps.detection.predictions",
        ),
    )
    graph.add_node(
        "three",
        kind=INPUT_NODE_KIND,
        definition=InferenceParameter(type="InferenceParameter", name="three"),
    )

    # when
    result = get_nodes_of_specific_kind(execution_graph=graph, kind=STEP_NODE_KIND)

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


def test_construct_selector_pointing_step_output_when_step_output_selector_provided() -> (
    None
):
    # when
    result = construct_selector_pointing_step_output(
        selector="$steps.some.parent_id", new_output="image"
    )

    # then
    assert result == "$steps.some.image"


def test_construct_selector_pointing_step_output_when_step_selector_provided() -> None:
    # when
    result = construct_selector_pointing_step_output(
        selector="$steps.some", new_output="image"
    )

    # then
    assert result == "$steps.some.image"


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


def test_get_output_selectors() -> None:
    # given
    outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
        JsonField(type="JsonField", name="other", selector="$steps.b.predictions"),
    ]

    # when
    result = get_output_selectors(outputs=outputs)

    # then
    assert result == {"$steps.a.predictions", "$steps.b.predictions"}


def test_get_output_names() -> None:
    # given
    outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
        JsonField(type="JsonField", name="other", selector="$steps.b.predictions"),
    ]

    # when
    result = get_output_names(outputs=outputs)

    # then
    assert result == {
        "$outputs.some",
        "$outputs.other",
    }, "$outputs. prefix must be added to each output name"


def test_get_steps_output_selectors() -> None:
    # given
    steps = [
        Crop(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect_2.predictions",
        ),
        ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="my_model",
            image="$inputs.image",
            model_id="some/1",
        ),
    ]

    # when
    result = get_steps_output_selectors(steps=steps)

    # then
    assert result == {
        "$steps.my_crop.crops",
        "$steps.my_crop.parent_id",
        "$steps.my_crop.*",
        "$steps.my_model.image",
        "$steps.my_model.predictions",
        "$steps.my_model.parent_id",
        "$steps.my_model.prediction_type",
        "$steps.my_model.*",
    }, "Each step output must be prefixed with $steps. and name of step. Crop step defines `crops` and `parent_id` outputs, object detection defines `image`, `predictions` and `parent_id`. Additionally, widlcard output must be registered!"


def test_get_steps_input_selectors() -> None:
    # given
    steps = [
        Crop(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect_2.predictions",
        ),
        ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="my_model",
            image="$inputs.image",
            model_id="some/1",
            confidence=0.3,
        ),
    ]

    # when
    result = get_steps_input_selectors(steps=steps)

    # then
    assert result == {
        "$inputs.image",
        "$steps.detect_2.predictions",
    }, "image reference and predictions reference defined within steps and expected to be found"


def test_get_steps_input_selectors_when_nested_selectors_are_detected() -> None:
    # given
    steps = [
        DetectionsConsensus(
            type="DetectionsConsensus",
            name="some",
            predictions=["$steps.detection.predictions", "$steps.other.predictions"],
            required_votes="$inputs.required_votes",
        )
    ]

    # when
    result = get_steps_input_selectors(steps=steps)

    # then
    assert result == {
        "$inputs.required_votes",
        "$steps.detection.predictions",
        "$steps.other.predictions",
    }, "reference to input parameter with required predictions and 2 detection steps outputs' must be found"


def test_get_steps_selectors() -> None:
    # given
    steps = [
        Crop(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect_2.predictions",
        ),
        ObjectDetectionModel(
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
        InferenceImage(type="InferenceImage", name="image"),
        InferenceParameter(type="InferenceParameter", name="x"),
        InferenceParameter(type="InferenceParameter", name="y"),
    ]

    # when
    result = get_input_parameters_selectors(inputs=inputs)

    # then
    assert result == {
        "$inputs.image",
        "$inputs.x",
        "$inputs.y",
    }, "Expected that each input will have its own selector with its name starting from $inputs."
