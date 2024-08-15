import pytest

from inference.core.workflows.core_steps.models.roboflow.object_detection import (
    v1 as object_detection_version_1,
)
from inference.core.workflows.core_steps.transformations.dynamic_crop import (
    v1 as dynamic_crop_version_1,
)
from inference.core.workflows.errors import DuplicatedNameError
from inference.core.workflows.execution_engine.entities.base import (
    JsonField,
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.compiler.validator import (
    validate_inputs_names_are_unique,
    validate_outputs_names_are_unique,
    validate_steps_names_are_unique,
)


def test_validate_inputs_names_are_unique_when_input_is_valid() -> None:
    # given
    inputs = [
        WorkflowImage(type="WorkflowImage", name="image"),
        WorkflowParameter(type="WorkflowParameter", name="x"),
        WorkflowParameter(type="WorkflowParameter", name="y"),
    ]

    # when
    validate_inputs_names_are_unique(inputs=inputs)

    # then - no error


def test_validate_inputs_names_are_unique_when_input_is_invalid() -> None:
    # given
    inputs = [
        WorkflowImage(type="WorkflowImage", name="image"),
        WorkflowParameter(type="WorkflowParameter", name="x"),
        WorkflowParameter(type="WorkflowParameter", name="x"),
    ]

    # when
    with pytest.raises(DuplicatedNameError):
        validate_inputs_names_are_unique(inputs=inputs)


def test_validate_steps_names_are_unique_when_input_is_valid() -> None:
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
    validate_steps_names_are_unique(steps=steps)

    # then - no error


def test_validate_steps_names_are_unique_when_input_is_invalid() -> None:
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
