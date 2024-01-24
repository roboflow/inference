from typing import Any

import networkx as nx
import numpy as np
import pytest

from inference.enterprise.deployments.complier.runtime_input_validator import (
    assembly_input_images,
    ensure_all_parameters_filled,
    fill_runtime_parameters_with_defaults,
    validate_inputs_binding,
)
from inference.enterprise.deployments.constants import INPUT_NODE_KIND, STEP_NODE_KIND
from inference.enterprise.deployments.entities.inputs import (
    InferenceImage,
    InferenceParameter,
)
from inference.enterprise.deployments.entities.steps import Crop, ObjectDetectionModel
from inference.enterprise.deployments.errors import (
    RuntimeParameterMissingError,
    VariableTypeError,
)


def test_ensure_all_parameters_filled_when_there_is_no_missing_parameters() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$inputs.one",
        kind=INPUT_NODE_KIND,
        definition=InferenceParameter(type="InferenceParameter", name="one"),
    )
    graph.add_node(
        "$inputs.two",
        kind=INPUT_NODE_KIND,
        definition=InferenceParameter(
            type="InferenceParameter", name="two", default_value=3
        ),
    )
    graph.add_node(
        "$inputs.three",
        kind=INPUT_NODE_KIND,
        definition=InferenceImage(type="InferenceImage", name="three"),
    )

    # when
    ensure_all_parameters_filled(
        execution_graph=graph,
        runtime_parameters={
            "three": {"type": "url", "value": "https://some.com/image.jpg"},
            "one": 1,
        },
    )

    # then - no error


def test_ensure_all_parameters_filled_when_there_is_missing_parameters() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$inputs.one",
        kind=INPUT_NODE_KIND,
        definition=InferenceParameter(type="InferenceParameter", name="one"),
    )
    graph.add_node(
        "$inputs.three",
        kind=INPUT_NODE_KIND,
        definition=InferenceImage(type="InferenceImage", name="three"),
    )

    # when
    with pytest.raises(RuntimeParameterMissingError):
        ensure_all_parameters_filled(
            execution_graph=graph,
            runtime_parameters={
                "three": {"type": "url", "value": "https://some.com/image.jpg"},
            },
        )


def test_fill_runtime_parameters_with_defaults() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$inputs.one",
        kind=INPUT_NODE_KIND,
        definition=InferenceParameter(type="InferenceParameter", name="one"),
    )
    graph.add_node(
        "$inputs.two",
        kind=INPUT_NODE_KIND,
        definition=InferenceParameter(
            type="InferenceParameter", name="two", default_value=2
        ),
    )
    graph.add_node(
        "$inputs.three",
        kind=INPUT_NODE_KIND,
        definition=InferenceParameter(
            type="InferenceParameter", name="three", default_value=3
        ),
    )

    # when
    result = fill_runtime_parameters_with_defaults(
        execution_graph=graph,
        runtime_parameters={
            "one": 1,
            "two": 22,
        },
    )

    # then
    assert result == {
        "one": 1,
        "two": 22,
        "three": 3,
    }, "Parameters that are explicitly given must shadow default, but parameter `three` must be returned with default"


def test_assembly_input_images_when_images_provided_as_single_elements() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$inputs.one",
        kind=INPUT_NODE_KIND,
        definition=InferenceImage(type="InferenceImage", name="one"),
    )
    graph.add_node(
        "$inputs.two",
        kind=INPUT_NODE_KIND,
        definition=InferenceImage(type="InferenceImage", name="two"),
    )

    # when
    result = assembly_input_images(
        execution_graph=graph,
        runtime_parameters={
            "one": {"type": "url", "value": "https://some.com/image.jpg"},
            "two": np.zeros((192, 168, 3), dtype=np.uint8),
            "some": "value",
        },
    )

    # then
    assert result["one"] == {
        "type": "url",
        "value": "https://some.com/image.jpg",
        "parent_id": "$inputs.one",
    }, "parent_id expected to be added"
    assert result["some"] == "value", "Value must not be touched by function"
    assert (
        result["two"]["type"] == "numpy_object"
    ), "numpy array must be packed in dict with type definition"
    assert (
        result["two"]["value"] == np.zeros((192, 168, 3), dtype=np.uint8)
    ).all(), "Image cannot be mutated"
    assert (
        result["two"]["parent_id"] == "$inputs.two"
    ), "parent_id expected to be added and match input identifier"


def test_assembly_input_images_when_images_provided_as_list_elements() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$inputs.one",
        kind=INPUT_NODE_KIND,
        definition=InferenceImage(type="InferenceImage", name="one"),
    )
    graph.add_node(
        "$inputs.two",
        kind=INPUT_NODE_KIND,
        definition=InferenceImage(type="InferenceImage", name="two"),
    )

    # when
    result = assembly_input_images(
        execution_graph=graph,
        runtime_parameters={
            "one": [
                {"type": "url", "value": "https://some.com/image.jpg"},
                {"type": "url", "value": "https://some.com/other.jpg"},
            ],
            "two": [
                np.zeros((192, 168, 3), dtype=np.uint8),
                np.ones((192, 168, 3), dtype=np.uint8),
            ],
            "some": "value",
        },
    )

    # then
    assert result["one"] == [
        {
            "type": "url",
            "value": "https://some.com/image.jpg",
            "parent_id": "$inputs.one.[0]",
        },
        {
            "type": "url",
            "value": "https://some.com/other.jpg",
            "parent_id": "$inputs.one.[1]",
        },
    ], "parent_id expected to be added"
    assert result["some"] == "value", "Value must not be touched by function"
    assert (
        result["two"][0]["type"] == "numpy_object"
    ), "numpy array must be packed in dict with type definition"
    assert (
        result["two"][0]["value"] == np.zeros((192, 168, 3), dtype=np.uint8)
    ).all(), "Image cannot be mutated"
    assert (
        result["two"][0]["parent_id"] == "$inputs.two.[0]"
    ), "parent_id expected to be added and match input identifier"
    assert (
        result["two"][1]["type"] == "numpy_object"
    ), "numpy array must be packed in dict with type definition"
    assert (
        result["two"][1]["value"] == np.ones((192, 168, 3), dtype=np.uint8)
    ).all(), "Image cannot be mutated"
    assert (
        result["two"][1]["parent_id"] == "$inputs.two.[1]"
    ), "parent_id expected to be added and match input identifier"


def test_validate_inputs_binding_when_validation_should_succeed() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$steps.one",
        kind=STEP_NODE_KIND,
        definition=Crop(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect.predictions",
        ),
    )
    graph.add_node(
        "$steps.two",
        kind=STEP_NODE_KIND,
        definition=ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="detect",
            image="$inputs.image",
            model_id="$inputs.model_id",
            confidence="$inputs.confidence",
        ),
    )

    # when
    validate_inputs_binding(
        execution_graph=graph,
        runtime_parameters={
            "image": {"type": "url", "value": "https://some.com/image.jpg"},
            "confidence": 0.3,
            "model_id": "some/3",
        },
    )

    # then - no error


@pytest.mark.parametrize(
    "faulty_key, faulty_value",
    [
        ("image", "invalid_image"),
        ("confidence", "invalid_confidence"),
        ("confidence", 1.1),
        ("model_id", None),
    ],
)
def test_validate_inputs_binding_when_validation_should_fail(
    faulty_key: str, faulty_value: Any
) -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$steps.one",
        kind=STEP_NODE_KIND,
        definition=Crop(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect.predictions",
        ),
    )
    graph.add_node(
        "$steps.two",
        kind=STEP_NODE_KIND,
        definition=ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="detect",
            image="$inputs.image",
            model_id="$inputs.model_id",
            confidence="$inputs.confidence",
        ),
    )
    runtime_parameters = {
        "image": {"type": "url", "value": "https://some.com/image.jpg"},
        "confidence": 0.3,
        "model_id": "some/3",
    }
    runtime_parameters[faulty_key] = faulty_value

    # when
    with pytest.raises(VariableTypeError):
        validate_inputs_binding(
            execution_graph=graph, runtime_parameters=runtime_parameters
        )
