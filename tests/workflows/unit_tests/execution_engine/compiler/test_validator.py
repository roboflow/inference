import pytest

from inference.enterprise.workflows.core_steps.models.roboflow import object_detection
from inference.enterprise.workflows.core_steps.transformations import crop
from inference.enterprise.workflows.entities.base import (
    InferenceImage,
    InferenceParameter,
    JsonField,
)
from inference.enterprise.workflows.errors import DuplicatedNameError
from inference.enterprise.workflows.execution_engine.compiler.validator import (
    validate_inputs_names_are_unique,
    validate_outputs_names_are_unique,
    validate_steps_names_are_unique,
    validate_workflow_specification,
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
    with pytest.raises(DuplicatedNameError):
        validate_inputs_names_are_unique(inputs=inputs)


def test_validate_steps_names_are_unique_when_input_is_valid() -> None:
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
    validate_steps_names_are_unique(steps=steps)

    # then - no error


def test_validate_steps_names_are_unique_when_input_is_invalid() -> None:
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
            name="my_crop",
            image="$inputs.image",
            model_id="some/1",
            confidence=0.3,
        ),
    ]

    # when
    with pytest.raises(DuplicatedNameError):
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
    with pytest.raises(DuplicatedNameError):
        validate_outputs_names_are_unique(outputs=outputs)
