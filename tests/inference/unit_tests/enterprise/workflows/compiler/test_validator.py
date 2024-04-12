import pytest

from inference.enterprise.workflows.complier.validator import (
    validate_inputs_names_are_unique,
    validate_outputs_names_are_unique,
    validate_steps_names_are_unique,
    validate_workflow_specification,
)
from inference.enterprise.workflows.entities.inputs import InferenceImage, InferenceParameter
from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.steps import Crop, ObjectDetectionModel
from inference.enterprise.workflows.entities.workflows_specification import (
    WorkflowSpecificationV1,
)
from inference.enterprise.workflows.errors import (
    DuplicatedSymbolError,
    InvalidReferenceError,
)


def test_validate_inputs_names_are_unique_when_input_is_valid() -> None:
    # given
    inputs = [
        InferenceImage(type="InferenceImage", name="image"),
        InferenceParameter(type="InferenceParameter", name="x"),
        InferenceParameter(type="InferenceParameter", name="y"),
    ]

    # when
    validate_inputs_names_are_unique(inputs=inputs)

    # then - no error


def test_validate_inputs_names_are_unique_when_input_is_invalid() -> None:
    # given
    inputs = [
        InferenceImage(type="InferenceImage", name="image"),
        InferenceParameter(type="InferenceParameter", name="x"),
        InferenceParameter(type="InferenceParameter", name="x"),
    ]

    # when
    with pytest.raises(DuplicatedSymbolError):
        validate_inputs_names_are_unique(inputs=inputs)


def test_validate_steps_names_are_unique_when_input_is_valid() -> None:
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
    validate_steps_names_are_unique(steps=steps)

    # then - no error


def test_validate_steps_names_are_unique_when_input_is_invalid() -> None:
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
            name="my_crop",
            image="$inputs.image",
            model_id="some/1",
            confidence=0.3,
        ),
    ]

    # when
    with pytest.raises(DuplicatedSymbolError):
        validate_steps_names_are_unique(steps=steps)


def test_validate_outputs_names_are_unique_when_input_is_valid() -> None:
    # given
    outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
        JsonField(type="JsonField", name="other", selector="$steps.b.predictions"),
    ]

    # when
    validate_outputs_names_are_unique(outputs=outputs)

    # then - no error


def test_validate_outputs_names_are_unique_when_input_is_invalid() -> None:
    # given
    outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
        JsonField(type="JsonField", name="some", selector="$steps.b.predictions"),
    ]

    # when
    with pytest.raises(DuplicatedSymbolError):
        validate_outputs_names_are_unique(outputs=outputs)


def test_validate_workflow_specification_when_there_is_selector_to_missing_element() -> (
    None
):
    # given
    workflow_specification = WorkflowSpecificationV1.parse_obj(
        {
            "version": "1.0",
            "inputs": [
                {"type": "InferenceImage", "name": "image"},
            ],
            "steps": [
                {
                    "type": "ObjectDetectionModel",
                    "name": "step_1",
                    "image": "$inputs.image",
                    "model_id": "vehicle-classification-eapcd/2",
                    "confidence": "$inputs.confidence",
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "predictions",
                    "selector": "$steps.step_1.predictions",
                },
            ],
        }
    )

    # when
    with pytest.raises(InvalidReferenceError):
        validate_workflow_specification(workflow_specification=workflow_specification)
