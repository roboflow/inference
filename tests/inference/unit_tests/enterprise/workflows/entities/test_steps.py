from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.entities.inputs import (
    InferenceImage,
    InferenceParameter,
)
from inference.enterprise.workflows.entities.steps import (
    LMM,
    ActiveLearningBatchingStrategy,
    ActiveLearningDataCollector,
    AggregationMode,
    ClassesBasedSampling,
    ClassificationModel,
    CloseToThresholdSampling,
    Condition,
    Crop,
    DetectionFilter,
    DetectionFilterDefinition,
    DetectionOffset,
    DetectionsBasedSampling,
    DetectionsConsensus,
    DisabledActiveLearningConfiguration,
    EnabledActiveLearningConfiguration,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    LimitDefinition,
    LMMConfig,
    LMMForClassification,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    Operator,
    RandomSamplingConfig,
    YoloWorld,
)
from inference.enterprise.workflows.errors import (
    ExecutionGraphError,
    InvalidStepInputDetected,
    VariableTypeError,
)


def test_classification_model_validation_when_minimalistic_config_is_provided() -> None:
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = ClassificationModel.parse_obj(data)

    # then
    assert result == ClassificationModel(
        type="ClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "image", "model_id"])
def test_classification_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = ClassificationModel.parse_obj(data)


def test_classification_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ClassificationModel.parse_obj(data)


def test_classification_model_validation_when_model_id_has_invalid_type() -> None:
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ClassificationModel.parse_obj(data)


def test_classification_model_validation_when_active_learning_flag_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ClassificationModel.parse_obj(data)


def test_classification_model_image_selector_when_selector_is_valid() -> None:
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = ClassificationModel.parse_obj(data)
    result.validate_field_selector(
        field_name="image",
        input_step=InferenceImage(type="InferenceImage", name="image"),
    )

    # then - no error is raised


def test_classification_model_image_selector_when_selector_is_invalid() -> None:
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = ClassificationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="some"),
        )


def test_classification_model_selector_when_model_id_is_invalid() -> None:
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = ClassificationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="model_id",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_classification_model_selector_when_disable_active_learning_is_invalid() -> (
    None
):
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        "disable_active_learning": "$inputs.disable_active_learning",
    }

    # when
    result = ClassificationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_classification_model_selector_when_referring_to_field_that_does_not_hold_selector() -> (
    None
):
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = ClassificationModel.parse_obj(data)
    with pytest.raises(ExecutionGraphError):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceParameter(type="InferenceParameter", name="image"),
        )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("confidence", 1.1),
        ("image", "some"),
        ("model_id", 38),
        ("disable_active_learning", "some"),
    ],
)
def test_classification_model_binding_when_parameter_is_invalid(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "ClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = ClassificationModel.parse_obj(data)
    with pytest.raises(VariableTypeError):
        result.validate_field_binding(
            field_name=field_name,
            value=value,
        )


def test_multi_label_classification_model_validation_when_minimalistic_config_is_provided() -> (
    None
):
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = MultiLabelClassificationModel.parse_obj(data)

    # then
    assert result == MultiLabelClassificationModel(
        type="MultiLabelClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "image", "model_id"])
def test_multi_label_classification_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = MultiLabelClassificationModel.parse_obj(data)


def test_multi_label_classification_model_validation_when_invalid_type_provided() -> (
    None
):
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = MultiLabelClassificationModel.parse_obj(data)


def test_multi_label_classification_model_validation_when_model_id_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = MultiLabelClassificationModel.parse_obj(data)


def test_multi_label_classification_model_validation_when_active_learning_flag_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = MultiLabelClassificationModel.parse_obj(data)


def test_multi_label_classification_model_image_selector_when_selector_is_valid() -> (
    None
):
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = MultiLabelClassificationModel.parse_obj(data)
    result.validate_field_selector(
        field_name="image",
        input_step=InferenceImage(type="InferenceImage", name="image"),
    )

    # then - no error is raised


def test_multi_label_classification_model_image_selector_when_selector_is_invalid() -> (
    None
):
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = MultiLabelClassificationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="some"),
        )


def test_multi_label_classification_model_selector_when_model_id_is_invalid() -> None:
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = MultiLabelClassificationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="model_id",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_multi_label_classification_model_selector_when_disable_active_learning_is_invalid() -> (
    None
):
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        "disable_active_learning": "$inputs.disable_active_learning",
    }

    # when
    result = MultiLabelClassificationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_multi_label_classification_model_selector_when_referring_to_field_that_does_not_hold_selector() -> (
    None
):
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = MultiLabelClassificationModel.parse_obj(data)
    with pytest.raises(ExecutionGraphError):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceParameter(type="InferenceParameter", name="image"),
        )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("confidence", 1.1),
        ("image", "some"),
        ("model_id", 38),
        ("disable_active_learning", "some"),
    ],
)
def test_multi_label_classification_model_binding_when_parameter_is_invalid(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "MultiLabelClassificationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = MultiLabelClassificationModel.parse_obj(data)
    with pytest.raises(VariableTypeError):
        result.validate_field_binding(
            field_name=field_name,
            value=value,
        )


def test_object_detection_model_validation_when_minimalistic_config_is_provided() -> (
    None
):
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)

    # then
    assert result == ObjectDetectionModel(
        type="ObjectDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "image", "model_id"])
def test_object_detection_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = ObjectDetectionModel.parse_obj(data)


def test_object_detection_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ObjectDetectionModel.parse_obj(data)


def test_object_detection_model_validation_when_model_id_has_invalid_type() -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ObjectDetectionModel.parse_obj(data)


def test_object_detection_model_validation_when_active_learning_flag_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ObjectDetectionModel.parse_obj(data)


def test_object_detection_model_image_selector_when_selector_is_valid() -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)
    result.validate_field_selector(
        field_name="image",
        input_step=InferenceImage(type="InferenceImage", name="image"),
    )

    # then - no error is raised


def test_object_detection_model_image_selector_when_selector_is_invalid() -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="some"),
        )


def test_object_detection_model_selector_when_model_id_is_invalid() -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="model_id",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_object_detection_model_selector_when_disable_active_learning_is_invalid() -> (
    None
):
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        "disable_active_learning": "$inputs.disable_active_learning",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_object_detection_model_selector_when_referring_to_field_that_does_not_hold_selector() -> (
    None
):
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)
    with pytest.raises(ExecutionGraphError):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceParameter(type="InferenceParameter", name="image"),
        )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("confidence", 1.1),
        ("image", "some"),
        ("disable_active_learning", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence", "some"),
        ("confidence", 1.1),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("max_detections", 0),
        ("max_candidates", 0),
    ],
)
def test_object_detection_model_when_parameters_have_invalid_type(
    parameter: str, value: Any
) -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        parameter: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ObjectDetectionModel.parse_obj(data)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("confidence", 1.1),
        ("image", "some"),
        ("model_id", 38),
        ("disable_active_learning", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence", "some"),
        ("confidence", 1.1),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("max_detections", 0),
        ("max_candidates", 0),
    ],
)
def test_object_detection_model_binding_when_parameter_is_invalid(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)
    with pytest.raises(VariableTypeError):
        result.validate_field_binding(
            field_name=field_name,
            value=value,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "class_agnostic_nms",
        "class_filter",
        "confidence",
        "iou_threshold",
        "max_detections",
        "max_candidates",
    ],
)
def test_object_detection_model_parameters_selector_validation_when_input_is_not_inference_parameter(
    field_name: str,
) -> None:
    # given
    data = {
        "type": "ObjectDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        field_name: "$inputs.some",
    }

    # when
    result = ObjectDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name=field_name,
            input_step=InferenceImage(type="InferenceImage", name="some"),
        )


def test_keypoints_detection_model_validation_when_minimalistic_config_is_provided() -> (
    None
):
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)

    # then
    assert result == KeypointsDetectionModel(
        type="KeypointsDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "image", "model_id"])
def test_keypoints_detection_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = KeypointsDetectionModel.parse_obj(data)


def test_keypoints_object_detection_model_validation_when_invalid_type_provided() -> (
    None
):
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = KeypointsDetectionModel.parse_obj(data)


def test_keypoints_detection_model_validation_when_model_id_has_invalid_type() -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = KeypointsDetectionModel.parse_obj(data)


def test_keypoints_detection_model_validation_when_active_learning_flag_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = KeypointsDetectionModel.parse_obj(data)


def test_keypoints_detection_model_image_selector_when_selector_is_valid() -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)
    result.validate_field_selector(
        field_name="image",
        input_step=InferenceImage(type="InferenceImage", name="image"),
    )

    # then - no error is raised


def test_keypoints_detection_model_image_selector_when_selector_is_invalid() -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="some"),
        )


def test_keypoints_detection_model_selector_when_model_id_is_invalid() -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="model_id",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_keypoints_detection_model_selector_when_disable_active_learning_is_invalid() -> (
    None
):
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        "disable_active_learning": "$inputs.disable_active_learning",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_keypoints_detection_model_selector_when_referring_to_field_that_does_not_hold_selector() -> (
    None
):
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)
    with pytest.raises(ExecutionGraphError):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceParameter(type="InferenceParameter", name="image"),
        )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("image", "some"),
        ("disable_active_learning", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence", "some"),
        ("confidence", 1.1),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("max_detections", 0),
        ("max_candidates", 0),
        ("keypoint_confidence", "some"),
        ("keypoint_confidence", 1.1),
    ],
)
def test_keypoints_detection_model_when_parameters_have_invalid_type(
    parameter: str, value: Any
) -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        parameter: value,
    }

    # when
    with pytest.raises(ValidationError):
        r = KeypointsDetectionModel.parse_obj(data)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("confidence", 1.1),
        ("image", "some"),
        ("model_id", 38),
        ("disable_active_learning", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence", "some"),
        ("confidence", 1.1),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("max_detections", 0),
        ("max_candidates", 0),
        ("keypoint_confidence", "some"),
        ("keypoint_confidence", 1.1),
    ],
)
def test_keypoints_detection_model_binding_when_parameter_is_invalid(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)
    with pytest.raises(VariableTypeError):
        result.validate_field_binding(
            field_name=field_name,
            value=value,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "class_agnostic_nms",
        "class_filter",
        "confidence",
        "iou_threshold",
        "max_detections",
        "max_candidates",
        "keypoint_confidence",
    ],
)
def test_keypoints_detection_model_parameters_selector_validation_when_input_is_not_inference_parameter(
    field_name: str,
) -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        field_name: "$inputs.some",
    }

    # when
    result = KeypointsDetectionModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name=field_name,
            input_step=InferenceImage(type="InferenceImage", name="some"),
        )


def test_instance_segmentation_model_validation_when_minimalistic_config_is_provided() -> (
    None
):
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)

    # then
    assert result == InstanceSegmentationModel(
        type="InstanceSegmentationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "image", "model_id"])
def test_instance_segmentation_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = InstanceSegmentationModel.parse_obj(data)


def test_instance_segmentation_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = InstanceSegmentationModel.parse_obj(data)


def test_instance_segmentation_model_validation_when_model_id_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = InstanceSegmentationModel.parse_obj(data)


def test_instance_segmentation_model_validation_when_active_learning_flag_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = InstanceSegmentationModel.parse_obj(data)


def test_instance_segmentation_model_image_selector_when_selector_is_valid() -> None:
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)
    result.validate_field_selector(
        field_name="image",
        input_step=InferenceImage(type="InferenceImage", name="image"),
    )

    # then - no error is raised


def test_instance_segmentation_model_image_selector_when_selector_is_invalid() -> None:
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="some"),
        )


def test_instance_segmentation_model_selector_when_model_id_is_invalid() -> None:
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="model_id",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_instance_segmentation_model_selector_when_disable_active_learning_is_invalid() -> (
    None
):
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        "disable_active_learning": "$inputs.disable_active_learning",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceImage(type="InferenceImage", name="image"),
        )


def test_instance_segmentation_model_selector_when_referring_to_field_that_does_not_hold_selector() -> (
    None
):
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)
    with pytest.raises(ExecutionGraphError):
        result.validate_field_selector(
            field_name="disable_active_learning",
            input_step=InferenceParameter(type="InferenceParameter", name="image"),
        )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("confidence", 1.1),
        ("image", "some"),
        ("disable_active_learning", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence", "some"),
        ("confidence", 1.1),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("max_detections", 0),
        ("max_candidates", 0),
        ("mask_decode_mode", "some"),
        ("tradeoff_factor", 1.1),
    ],
)
def test_instance_segmentation_model_when_parameters_have_invalid_type(
    parameter: str, value: Any
) -> None:
    # given
    data = {
        "type": "KeypointsDetectionModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        parameter: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = InstanceSegmentationModel.parse_obj(data)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("confidence", 1.1),
        ("image", "some"),
        ("model_id", 38),
        ("disable_active_learning", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence", "some"),
        ("confidence", 1.1),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("max_detections", 0),
        ("max_candidates", 0),
        ("mask_decode_mode", "some"),
        ("tradeoff_factor", 1.1),
    ],
)
def test_instance_segmentation_model_binding_when_parameter_is_invalid(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)
    with pytest.raises(VariableTypeError):
        result.validate_field_binding(
            field_name=field_name,
            value=value,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "class_agnostic_nms",
        "class_filter",
        "confidence",
        "iou_threshold",
        "max_detections",
        "max_candidates",
        "mask_decode_mode",
        "tradeoff_factor",
    ],
)
def test_instance_segmentation_model_parameters_selector_validation_when_input_is_not_inference_parameter(
    field_name: str,
) -> None:
    # given
    data = {
        "type": "InstanceSegmentationModel",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "$inputs.model",
        field_name: "$inputs.some",
    }

    # when
    result = InstanceSegmentationModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        result.validate_field_selector(
            field_name=field_name,
            input_step=InferenceImage(type="InferenceImage", name="some"),
        )


def test_ocr_model_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "OCRModel",
        "name": "some",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = OCRModel.parse_obj(data)


def test_ocr_model_selector_validation_when_invalid_image_input_selector_is_given() -> (
    None
):
    # given
    data = {
        "type": "OCRModel",
        "name": "some",
        "image": "$inputs.image",
    }

    # when
    ocr_model = OCRModel.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        ocr_model.validate_field_selector(
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="some"),
        )


def test_ocr_model_biding_validation_when_invalid_image_input_is_given() -> None:
    # given
    data = {
        "type": "OCRModel",
        "name": "some",
        "image": "$inputs.image",
    }

    # when
    ocr_model = OCRModel.parse_obj(data)
    with pytest.raises(VariableTypeError):
        ocr_model.validate_field_binding(
            field_name="image",
            value="invalid",
        )


def test_crop_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "image": "invalid",
        "detections": "$steps.detection.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = Crop.parse_obj(data)


def test_crop_selector_validation_when_invalid_image_input_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "image": "$inputs.image",
        "detections": "$steps.detection.predictions",
    }

    # when
    crop = Crop.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        crop.validate_field_selector(
            field_name="image",
            input_step=InferenceParameter(type="InferenceParameter", name="some"),
        )


def test_crop_selector_validation_when_invalid_detections_input_selector_is_given() -> (
    None
):
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "image": "$inputs.image",
        "detections": "$steps.detection.predictions",
    }

    # when
    crop = Crop.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        crop.validate_field_selector(
            field_name="detections",
            input_step=InferenceParameter(type="InferenceParameter", name="detections"),
        )


def test_crop_biding_validation_when_invalid_image_input_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "image": "$inputs.image",
        "detections": "$steps.detection.predictions",
    }

    # when
    crop = Crop.parse_obj(data)
    with pytest.raises(VariableTypeError):
        crop.validate_field_binding(
            field_name="image",
            value="invalid",
        )


def test_condition_selector_validation_when_invalid_value_is_provided() -> None:
    # given
    data = {
        "type": "Condition",
        "name": "some",
        "left": "$inputs.left",
        "right": "$inputs.right",
        "operator": "equal",
        "step_if_true": "$steps.a",
        "step_if_false": "$steps.b",
    }

    # when
    condition = Condition.parse_obj(data)
    with pytest.raises(InvalidStepInputDetected):
        condition.validate_field_selector(
            field_name="left",
            input_step=InferenceImage(type="InferenceImage", name="detections"),
        )


def test_detection_filter_selector_validation_when_invalid_predictions_are_given() -> (
    None
):
    # given
    detections_filter = DetectionFilter(
        type="DetectionFilter",
        name="some",
        predictions="$steps.step_a.predictions",
        filter_definition=DetectionFilterDefinition(
            type="DetectionFilterDefinition",
            field_name="confidence",
            operator=Operator.GREATER_THAN,
            reference_value=0.3,
        ),
    )

    # when
    with pytest.raises(InvalidStepInputDetected):
        detections_filter.validate_field_selector(
            field_name="predictions",
            input_step=InferenceImage(type="InferenceImage", name="detections"),
        )


def test_detections_offset_selector_validation_when_invalid_predictions_are_given() -> (
    None
):
    # given
    detections_offset = DetectionOffset(
        type="DetectionOffset",
        name="some",
        predictions="$steps.step_a.predictions",
        offset_x=30,
        offset_y=40,
    )

    # when
    with pytest.raises(InvalidStepInputDetected):
        detections_offset.validate_field_selector(
            field_name="predictions",
            input_step=InferenceImage(type="InferenceImage", name="detections"),
        )


def test_detections_offset_selector_validation_when_invalid_offset_is_given() -> None:
    # given
    detections_offset = DetectionOffset(
        type="DetectionOffset",
        name="some",
        predictions="$steps.step_a.predictions",
        offset_x="$inputs.offset_x",
        offset_y=40,
    )

    # when
    with pytest.raises(InvalidStepInputDetected):
        detections_offset.validate_field_selector(
            field_name="offset_x",
            input_step=InferenceImage(type="InferenceImage", name="detections"),
        )


def test_detections_offset_binding_validation_when_invalid_offset_is_given() -> None:
    # given
    detections_offset = DetectionOffset(
        type="DetectionOffset",
        name="some",
        predictions="$steps.step_a.predictions",
        offset_x="$inputs.offset_x",
        offset_y=40,
    )

    # when
    with pytest.raises(VariableTypeError):
        detections_offset.validate_field_binding(
            field_name="offset_x",
            value="invalid",
        )


def test_detections_consensus_validation_when_valid_specification_given() -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result == DetectionsConsensus(
        type="DetectionsConsensus",
        name="some",
        predictions=["$steps.detection.predictions", "$steps.detection_2.predictions"],
        required_votes=3,
        class_aware=True,
        iou_threshold=0.3,
        confidence=0.0,
        classes_to_consider=None,
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )


@pytest.mark.parametrize("value", [3, "3", True, 3.0, [], set(), {}, None])
def test_detections_consensus_validation_when_predictions_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": value,
        "required_votes": 3,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize("value", [None, 0, -1, "some", []])
def test_detections_consensus_validation_when_required_votes_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize("value", [3, "$inputs.some"])
def test_detections_consensus_validation_when_required_votes_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result.required_votes == value


@pytest.mark.parametrize("value", [None, "some"])
def test_detections_consensus_validation_when_class_aware_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        "class_aware": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize("value", [True, False])
def test_detections_consensus_validation_when_class_aware_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        "class_aware": value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result.class_aware == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("iou_threshold", None),
        ("iou_threshold", -1),
        ("iou_threshold", 2.0),
        ("iou_threshold", "some"),
        ("confidence", None),
        ("confidence", -1),
        ("confidence", 2.0),
        ("confidence", "some"),
    ],
)
def test_detections_consensus_validation_when_range_field_of_invalid_type_given(
    field: str,
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        field: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize(
    "field,value",
    [
        ("iou_threshold", 0.0),
        ("iou_threshold", 1.0),
        ("iou_threshold", 0.5),
        ("iou_threshold", "$inputs.some"),
        ("confidence", 0.0),
        ("confidence", 1.0),
        ("confidence", 0.5),
        ("confidence", "$inputs.some"),
    ],
)
def test_detections_consensus_validation_when_range_field_of_valid_type_given(
    field: str,
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        field: value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert getattr(result, field) == value


@pytest.mark.parametrize("value", ["some", 1, 2.0, True, {}])
def test_detections_consensus_validation_when_classes_to_consider_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        "classes_to_consider": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize("value", ["$inputs.some", [], ["1", "2", "3"]])
def test_detections_consensus_validation_when_classes_to_consider_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        "classes_to_consider": value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result.classes_to_consider == value


@pytest.mark.parametrize(
    "value", ["some", -1, 0, {"some": None}, {"some": 1, "other": -1}]
)
def test_detections_consensus_validation_when_required_objects_of_invalid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        "required_objects": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize(
    "value", [None, "$inputs.some", 1, 10, {"some": 1, "other": 10}]
)
def test_detections_consensus_validation_when_required_objects_of_valid_type_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
        "required_objects": value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result.required_objects == value


def test_detections_consensus_validation_field_predictions_field_selector_when_index_is_not_given() -> (
    None
):
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions", input_step=MagicMock(), index=None
        )


def test_detections_consensus_validation_field_predictions_field_selector_when_index_is_out_of_range() -> (
    None
):
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=MagicMock(),
            index=3,
        )


def test_detections_consensus_validation_field_predictions_field_selector_when_selector_does_not_hold_detections() -> (
    None
):
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.some"],
        "required_votes": 3,
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=ObjectDetectionModel(
                type="ObjectDetectionModel",
                name="detection_2",
                image="$inputs.image",
                model_id="some/1",
            ),
            index=1,
        )


def test_detections_consensus_validation_field_predictions_field_selector_when_selector_point_to_invalid_step() -> (
    None
):
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=Crop(
                type="Crop",
                name="detection_2",
                image="$inputs.image",
                detections="$steps.step.predictions",
            ),
            index=1,
        )


def test_detections_consensus_validation_field_predictions_field_selector_when_selector_point_to_valid_step() -> (
    None
):
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    step.validate_field_selector(
        field_name="predictions",
        input_step=ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="detection_2",
            image="$inputs.image",
            model_id="some/1",
        ),
        index=1,
    )

    # then - no error


@pytest.mark.parametrize(
    "field",
    [
        "required_votes",
        "class_aware",
        "iou_threshold",
        "confidence",
        "classes_to_consider",
        "required_objects",
    ],
)
def test_detections_consensus_validation_field_that_is_supposed_to_be_selector_but_is_not(
    field: str,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 3,
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name=field,
            input_step=MagicMock(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "required_votes",
        "class_aware",
        "iou_threshold",
        "confidence",
        "classes_to_consider",
        "required_objects",
    ],
)
def test_detections_consensus_validation_field_that_is_supposed_to_be_parameter_selector_but_is_not(
    field: str,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": "$inputs.some",
        "class_aware": "$inputs.some",
        "iou_threshold": "$inputs.some",
        "confidence": "$inputs.some",
        "classes_to_consider": "$inputs.some",
        "required_objects": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name=field,
            input_step=Crop(
                type="Crop",
                name="detection_2",
                image="$inputs.image",
                detections="$steps.step.predictions",
            ),
        )


@pytest.mark.parametrize("value", [None, -1, "some", [], 0])
def test_detections_consensus_validate_field_binding_for_required_votes_when_value_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="required_votes",
            value=value,
        )


@pytest.mark.parametrize("value", [1, 10])
def test_detections_consensus_validate_field_binding_for_required_votes_when_value_is_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="required_votes",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize("value", [None, "some", []])
def test_detections_consensus_validate_field_binding_for_class_aware_when_value_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        "class_aware": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="class_aware",
            value=value,
        )


@pytest.mark.parametrize("value", [True, False])
def test_detections_consensus_validate_field_binding_for_class_aware_when_value_is_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        "class_aware": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="class_aware",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize(
    "field, value",
    [
        ("iou_threshold", None),
        ("iou_threshold", -1),
        ("iou_threshold", "some"),
        ("confidence", None),
        ("confidence", -1),
        ("confidence", "some"),
    ],
)
def test_detections_consensus_validate_field_binding_for_zero_one_range_field_when_value_is_invalid(
    field: str, value: Any
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        field: "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name=field,
            value=value,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("iou_threshold", 0.0),
        ("iou_threshold", 0.5),
        ("iou_threshold", 1.0),
        ("confidence", 0.0),
        ("confidence", 0.5),
        ("confidence", 1.0),
    ],
)
def test_detections_consensus_validate_field_binding_for_zero_one_range_field_when_value_is_valid(
    field: str, value: Any
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        field: "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name=field,
        value=value,
    )

    # then - no error


@pytest.mark.parametrize("value", ["some", 1, 2.0, True, {}, ["some", 1]])
def test_detections_consensus_validate_field_binding_for_classes_to_consider_when_value_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        "classes_to_consider": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="classes_to_consider",
            value=value,
        )


@pytest.mark.parametrize("value", [None, ["A", "B"]])
def test_detections_consensus_validate_field_binding_for_classes_to_consider_when_value_is_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        "classes_to_consider": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="classes_to_consider",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize(
    "value", ["some", -1, 0, ["some"], {"some": None}, {"some": 1, "other": -1}]
)
def test_detections_consensus_validate_field_binding_for_required_objects_when_value_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        "required_objects": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="required_objects",
            value=value,
        )


@pytest.mark.parametrize("value", [None, 1, 3, {"some": 1, "other": 2}])
def test_detections_consensus_validate_field_binding_for_required_objects_when_value_is_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "required_votes": 1,
        "required_objects": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="required_objects",
        value=value,
    )

    # then - no error


def test_validate_al_data_collector_when_valid_input_given() -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }

    # when
    result = ActiveLearningDataCollector.parse_obj(specification)

    # then
    assert result == ActiveLearningDataCollector(
        type="ActiveLearningDataCollector",
        name="some",
        image="$inputs.image",
        predictions="$steps.detection.predictions",
        target_dataset="some",
        target_dataset_api_key=None,
        disable_active_learning=False,
        active_learning_configuration=None,
    )


def test_validate_al_data_collector_when_valid_input_with_disabled_al_config_given() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
        "active_learning_configuration": {"enabled": False},
    }

    # when
    result = ActiveLearningDataCollector.parse_obj(specification)

    # then
    assert result == ActiveLearningDataCollector(
        type="ActiveLearningDataCollector",
        name="some",
        image="$inputs.image",
        predictions="$steps.detection.predictions",
        target_dataset="some",
        target_dataset_api_key=None,
        disable_active_learning=False,
        active_learning_configuration=DisabledActiveLearningConfiguration(
            enabled=False
        ),
    )


def test_validate_al_data_collector_when_valid_input_with_enabled_al_config_given() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
        "active_learning_configuration": {
            "enabled": True,
            "persist_predictions": True,
            "sampling_strategies": [
                {
                    "type": "random",
                    "name": "a",
                    "traffic_percentage": 0.6,
                    "limits": [{"type": "daily", "value": 100}],
                },
                {
                    "type": "close_to_threshold",
                    "name": "b",
                    "probability": 0.7,
                    "threshold": 0.5,
                    "epsilon": 0.25,
                    "tags": ["some"],
                    "limits": [{"type": "daily", "value": 200}],
                },
                {
                    "type": "classes_based",
                    "name": "c",
                    "probability": 0.8,
                    "selected_class_names": ["a", "b", "c"],
                    "limits": [{"type": "daily", "value": 300}],
                },
                {
                    "type": "detections_number_based",
                    "name": "d",
                    "probability": 0.9,
                    "more_than": 3,
                    "less_than": 5,
                    "limits": [{"type": "daily", "value": 400}],
                },
            ],
            "batching_strategy": {
                "batches_name_prefix": "my_batches",
                "recreation_interval": "monthly",
            },
        },
    }

    # when
    result = ActiveLearningDataCollector.parse_obj(specification)

    # then
    assert result == ActiveLearningDataCollector(
        type="ActiveLearningDataCollector",
        name="some",
        image="$inputs.image",
        predictions="$steps.detection.predictions",
        target_dataset="some",
        target_dataset_api_key=None,
        disable_active_learning=False,
        active_learning_configuration=EnabledActiveLearningConfiguration(
            enabled=True,
            persist_predictions=True,
            sampling_strategies=[
                RandomSamplingConfig(
                    type="random",
                    name="a",
                    traffic_percentage=0.6,
                    tags=[],
                    limits=[LimitDefinition(type="daily", value=100)],
                ),
                CloseToThresholdSampling(
                    type="close_to_threshold",
                    name="b",
                    probability=0.7,
                    threshold=0.5,
                    epsilon=0.25,
                    max_batch_images=None,
                    only_top_classes=True,
                    minimum_objects_close_to_threshold=1,
                    selected_class_names=None,
                    tags=["some"],
                    limits=[LimitDefinition(type="daily", value=200)],
                ),
                ClassesBasedSampling(
                    type="classes_based",
                    name="c",
                    probability=0.8,
                    selected_class_names=["a", "b", "c"],
                    tags=[],
                    limits=[LimitDefinition(type="daily", value=300)],
                ),
                DetectionsBasedSampling(
                    type="detections_number_based",
                    name="d",
                    probability=0.9,
                    more_than=3,
                    less_than=5,
                    selected_class_names=None,
                    tags=[],
                    limits=[LimitDefinition(type="daily", value=400)],
                ),
            ],
            batching_strategy=ActiveLearningBatchingStrategy(
                batches_name_prefix="my_batches",
                recreation_interval="monthly",
            ),
            tags=[],
            max_image_size=None,
            jpeg_compression_level=95,
        ),
    )


@pytest.mark.parametrize("image_selector", [1, None, "some", 1.3, True])
def test_validate_al_data_collector_image_field_when_field_does_not_hold_selector(
    image_selector: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": image_selector,
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ActiveLearningDataCollector.parse_obj(specification)


@pytest.mark.parametrize("predictions_selector", [1, None, "some", 1.3, True])
def test_validate_al_data_collector_predictions_field_when_field_does_not_hold_selector(
    predictions_selector: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": predictions_selector,
        "target_dataset": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ActiveLearningDataCollector.parse_obj(specification)


@pytest.mark.parametrize("target_dataset", [1, None, 1.3, True])
def test_validate_al_data_collector_target_dataset_field_when_field_contains_invalid_value(
    target_dataset: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": target_dataset,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ActiveLearningDataCollector.parse_obj(specification)


@pytest.mark.parametrize("target_dataset_api_key", [1, 1.3, True])
def test_validate_al_data_collector_target_dataset_api_key_field_when_field_contains_invalid_value(
    target_dataset_api_key: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
        "target_dataset_api_key": target_dataset_api_key,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ActiveLearningDataCollector.parse_obj(specification)


@pytest.mark.parametrize("disable_active_learning", ["some"])
def test_validate_al_data_collector_disable_active_learning_field_when_field_contains_invalid_value(
    disable_active_learning: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
        "disable_active_learning": disable_active_learning,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ActiveLearningDataCollector.parse_obj(specification)


def test_al_data_collector_validate_field_selector_when_field_does_not_hold_selector() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="target_dataset",
            input_step=ObjectDetectionModel(
                type="ObjectDetectionModel",
                name="some",
                image="$inputs.image",
                model_id="some/1",
            ),
        )


def test_al_data_collector_validate_field_selector_when_prediction_field_refers_to_invalid_step() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=Crop(
                type="Crop",
                name="some",
                image="$inputs.image",
                detections="$steps.detection.predictions",
            ),
        )


def test_al_data_collector_validate_field_selector_when_prediction_field_refers_to_invalid_output_of_detection_step() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.image",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=ObjectDetectionModel(
                type="ObjectDetectionModel",
                name="some",
                image="$inputs.image",
                model_id="some/1",
            ),
        )


def test_al_data_collector_validate_field_selector_when_prediction_field_refers_to_invalid_output_of_classification_step() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=ClassificationModel(
                type="ClassificationModel",
                name="some",
                image="$inputs.image",
                model_id="some/1",
            ),
        )


def test_al_data_collector_validate_field_selector_when_prediction_field_refers_to_valid_output_of_classification_step() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.top",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_selector(
        field_name="predictions",
        input_step=ClassificationModel(
            type="ClassificationModel",
            name="some",
            image="$inputs.image",
            model_id="some/1",
        ),
    )

    # then - NO ERROR


def test_al_data_collector_validate_field_selector_when_prediction_field_refers_to_step_bounded_in_different_image() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=ObjectDetectionModel(
                type="ObjectDetectionModel",
                name="some",
                image="$inputs.image2",
                model_id="some/1",
            ),
        )


def test_al_data_collector_validate_field_selector_when_prediction_field_refers_to_step_which_cannot_be_verified_against_image_ref_correctness() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_selector(
        field_name="predictions",
        input_step=DetectionFilter(
            type="DetectionFilter",
            name="detection",
            predictions="$steps.det.predictions",
            filter_definition=DetectionFilterDefinition(
                type="DetectionFilterDefinition",
                field_name="confidence",
                operator="greater_than",
                reference_value=0.3,
            ),
        ),
    )

    # then - NO ERROR


def test_al_data_collector_validate_field_selector_when_image_field_does_not_refer_image() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="image",
            input_step=ObjectDetectionModel(
                type="ObjectDetectionModel",
                name="some",
                image="$inputs.image2",
                model_id="some/1",
            ),
        )


def test_al_data_collector_validate_field_selector_when_image_field_refers_image() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_selector(
        field_name="image",
        input_step=InferenceImage(
            type="InferenceImage",
            name="some",
        ),
    )

    # then - NO ERROR


@pytest.mark.parametrize(
    "field_name",
    ["target_dataset", "target_dataset_api_key", "disable_active_learning"],
)
def test_al_data_collector_validate_fields_that_can_only_accept_inference_parameter_when_invalid_input_is_provided(
    field_name: str,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "some",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name=field_name,
            input_step=InferenceImage(
                type="InferenceImage",
                name="some",
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    ["target_dataset", "target_dataset_api_key", "disable_active_learning"],
)
def test_al_data_collector_validate_fields_that_can_only_accept_inference_parameter_when_valid_input_is_provided(
    field_name: str,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_selector(
        field_name=field_name,
        input_step=InferenceParameter(
            type="InferenceParameter",
            name="some",
        ),
    )

    # then - NO ERROR


def test_al_data_collector_validate_image_binding_when_provided_value_is_valid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="image", value={"type": "url", "value": "https://some.com/image.jpg"}
    )

    # then - NO ERROR


def test_al_data_collector_validate_image_binding_when_provided_value_is_invalid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="image", value="invalid")


def test_al_data_collector_validate_disable_al_flag_binding_when_provided_value_is_valid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="disable_active_learning",
        value=True,
    )

    # then - NO ERROR


def test_al_data_collector_validate_disable_al_flag_binding_when_provided_value_is_invalid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="disable_active_learning",
            value="some",
        )


def test_al_data_collector_validate_target_dataset_binding_when_provided_value_is_valid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="target_dataset",
        value="some",
    )

    # then - NO ERROR


def test_al_data_collector_validate_target_dataset_binding_when_provided_value_is_invalid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="target_dataset",
            value=None,
        )


def test_al_data_collector_validate_target_dataset_api_key_binding_when_provided_value_is_valid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="target_dataset_api_key",
        value="some",
    )

    # then - NO ERROR


def test_al_data_collector_validate_target_dataset_api_key_binding_when_provided_value_is_invalid() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "target_dataset": "$inputs.some",
        "target_dataset_api_key": "$inputs.other",
        "disable_active_learning": "$inputs.value",
    }
    step = ActiveLearningDataCollector.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="target_dataset_api_key",
            value=None,
        )


def test_yolo_world_step_configuration_decoding_when_valid_config_is_given() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": "$inputs.classes",
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    result = YoloWorld.parse_obj(specification)

    # then
    assert result == YoloWorld(
        type="YoloWorld",
        name="step_1",
        image="$inputs.image",
        class_names="$inputs.classes",
        version="s",
        confidence="$inputs.confidence",
    )


@pytest.mark.parametrize("value", ["some", [], np.zeros((192, 168, 3))])
def test_yolo_world_step_image_validation_when_invalid_image_given(value: Any) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": value,
        "class_names": "$inputs.classes",
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    with pytest.raises(ValidationError):
        _ = YoloWorld.parse_obj(specification)


@pytest.mark.parametrize("value", ["some", [1, 2], True, 3])
def test_yolo_world_step_image_validation_when_invalid_class_names_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": value,
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    with pytest.raises(ValidationError):
        _ = YoloWorld.parse_obj(specification)


def test_yolo_world_step_image_validation_when_valid_class_names_given() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    result = YoloWorld.parse_obj(specification)

    # then
    assert result == YoloWorld(
        type="YoloWorld",
        name="step_1",
        image="$inputs.image",
        class_names=["a", "b"],
        version="s",
        confidence="$inputs.confidence",
    )


@pytest.mark.parametrize("value", ["some", [1, 2], True, 3])
def test_yolo_world_step_image_validation_when_invalid_version_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": "$inputs.confidence",
        "version": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = YoloWorld.parse_obj(specification)


@pytest.mark.parametrize("value", ["s", "m", "l", "x", "v2-s", "v2-m", "v2-l", "v2-x"])
def test_yolo_world_step_image_validation_when_valid_version_given(value: Any) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": "$inputs.confidence",
        "version": value,
    }

    # when
    result = YoloWorld.parse_obj(specification)

    # then
    assert result == YoloWorld(
        type="YoloWorld",
        name="step_1",
        image="$inputs.image",
        class_names=["a", "b"],
        version=value,
        confidence="$inputs.confidence",
    )


@pytest.mark.parametrize("value", ["some", [1, 2], 3, 1.1, -0.1])
def test_yolo_world_step_image_validation_when_invalid_confidence_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": value,
        "version": "s",
    }

    # when
    with pytest.raises(ValidationError):
        _ = YoloWorld.parse_obj(specification)


@pytest.mark.parametrize("value", [None, 0.3, 1.0, 0.0])
def test_yolo_world_step_image_validation_when_valid_confidence_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": value,
        "version": "s",
    }

    # when
    result = YoloWorld.parse_obj(specification)

    # then
    assert result == YoloWorld(
        type="YoloWorld",
        name="step_1",
        image="$inputs.image",
        class_names=["a", "b"],
        version="s",
        confidence=value,
    )


def test_yolo_world_step_selector_validation_when_field_is_not_selector() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "s",
    }
    step = YoloWorld.parse_obj(specification)
    inpt_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name="confidence", input_step=inpt_step)


def test_yolo_world_step_image_selector_validation_when_valid_image_given() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "s",
    }
    step = YoloWorld.parse_obj(specification)
    inpt_step = InferenceImage(type="InferenceImage", name="some")

    # when
    step.validate_field_selector(field_name="image", input_step=inpt_step)

    # then - no error expected


def test_yolo_world_step_image_selector_validation_when_invalid_image_given() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "s",
    }
    step = YoloWorld.parse_obj(specification)
    inpt_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name="image", input_step=inpt_step)


def test_yolo_world_step_inference_parameter_selector_validation_when_valid_input_given() -> (
    None
):
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)
    inpt_step = InferenceParameter(type="InferenceParameter", name="version")

    # when
    step.validate_field_selector(field_name="version", input_step=inpt_step)

    # then - no error expected


def test_yolo_world_step_inference_parameter_selector_validation_when_invalid_input_given() -> (
    None
):
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)
    inpt_step = InferenceImage(type="InferenceImage", name="version")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name="version", input_step=inpt_step)


def test_yolo_world_step_image_binding_validation_when_input_is_valid() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="image", value={"type": "url", "value": "https://some/image.jpg"}
    )

    # then - no error


def test_yolo_world_step_image_binding_validation_when_input_is_invalid() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="image", value="invalid")


@pytest.mark.parametrize("value", [None, "s", "l", "m"])
def test_yolo_world_step_version_binding_validation_when_input_is_valid(
    value: Optional[str],
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="version",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize("value", ["some", 1, True])
def test_yolo_world_step_version_binding_validation_when_input_is_invalid(
    value: Optional[str],
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="version",
            value=value,
        )


def test_yolo_world_step_class_names_binding_validation_when_input_is_valid() -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": "$inputs.classes",
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="class_names",
        value=["a", "b"],
    )

    # then - no error


@pytest.mark.parametrize("value", ["some", 1, True, [1, "a"]])
def test_yolo_world_step_class_names_binding_validation_when_input_is_invalid(
    value: Optional[str],
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": "$inputs.classes",
        "confidence": 0.3,
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="class_names", value=value)


@pytest.mark.parametrize("value", [None, 0.0, 0.3, 1.0])
def test_yolo_world_step_confidence_binding_validation_when_input_is_valid(
    value: Optional[float],
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": "$inputs.conf",
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="confidence",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize("value", ["some", -1, -0.1, 1.1])
def test_yolo_world_step_confidence_binding_validation_when_input_is_invalid(
    value: Optional[float],
) -> None:
    # given
    specification = {
        "type": "YoloWorld",
        "name": "step_1",
        "image": "$inputs.some",
        "class_names": ["a", "b"],
        "confidence": "$inputs.conf",
        "version": "$inputs.version",
    }
    step = YoloWorld.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="confidence",
            value=value,
        )


def test_lmm_step_validation_when_input_is_valid() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = LMM.parse_obj(specification)

    # then
    assert result == LMM(
        type="LMM",
        name="step_1",
        image="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="$inputs.lmm_type",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_lmm_step_validation_when_image_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": value,
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMM.parse_obj(specification)


def test_lmm_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "This is my prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = LMM.parse_obj(specification)

    # then
    assert result == LMM(
        type="LMM",
        name="step_1",
        image="$inputs.image",
        prompt="This is my prompt",
        lmm_type="$inputs.lmm_type",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize("value", [None, []])
def test_lmm_step_validation_when_prompt_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": value,
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMM.parse_obj(specification)


@pytest.mark.parametrize("value", ["$inputs.model", "gpt_4v", "cog_vlm"])
def test_lmm_step_validation_when_lmm_type_valid(value: Any) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": value,
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = LMM.parse_obj(specification)

    assert result == LMM(
        type="LMM",
        name="step_1",
        image="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type=value,
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize("value", ["some", None])
def test_lmm_step_validation_when_lmm_type_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": value,
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMM.parse_obj(specification)


@pytest.mark.parametrize("value", ["$inputs.api_key", "my-api-key", None])
def test_lmm_step_validation_when_remote_api_key_valid(value: Any) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": value,
    }

    # when
    result = LMM.parse_obj(specification)

    assert result == LMM(
        type="LMM",
        name="step_1",
        image="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="gpt_4v",
        lmm_config=LMMConfig(),
        remote_api_key=value,
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize(
    "value", [None, "$inputs.some", {"my_field": "my_description"}]
)
def test_lmm_step_validation_when_json_output_valid(value: Any) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": value,
        "remote_api_key": "some",
    }

    # when
    result = LMM.parse_obj(specification)

    assert result == LMM(
        type="LMM",
        name="step_1",
        image="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="gpt_4v",
        lmm_config=LMMConfig(),
        remote_api_key="some",
        json_output=value,
    )


@pytest.mark.parametrize(
    "value",
    [{"my_field": 3}, "some", {"structured_output": "This is registered field"}],
)
def test_lmm_step_validation_when_json_output_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": value,
        "remote_api_key": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMM.parse_obj(specification)


def test_lmm_step_get_output_names_when_structured_output_is_defined() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": {"some": "My field"},
        "remote_api_key": "xxx",
    }
    step = LMM.parse_obj(specification)

    # when
    result = step.get_output_names()

    # then
    assert result == {
        "raw_output",
        "structured_output",
        "image",
        "parent_id",
        "some",
    }, "`some` field must be present"


def test_lmm_step_get_output_names_when_structured_output_is_not_defined() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "xxx",
    }
    step = LMM.parse_obj(specification)

    # when
    result = step.get_output_names()

    # then
    assert result == {
        "raw_output",
        "structured_output",
        "image",
        "parent_id",
    }, "Only base fields must not be present"


def test_lmm_step_validation_of_field_selector_when_attempted_to_validate_non_selector() -> (
    None
):
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "xxx",
    }
    step = LMM.parse_obj(specification)
    input_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name="lmm_type", input_step=input_step)


def test_lmm_step_validation_of_image_field_selector_when_valid_input_given() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "xxx",
    }
    step = LMM.parse_obj(specification)
    input_step = InferenceImage(type="InferenceImage", name="some")

    # when
    step.validate_field_selector(field_name="image", input_step=input_step)

    # then - no error


def test_lmm_step_validation_of_image_field_selector_when_invalid_input_given() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "xxx",
    }
    step = LMM.parse_obj(specification)
    input_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name="image", input_step=input_step)


@pytest.mark.parametrize(
    "field", ["prompt", "lmm_type", "remote_api_key", "json_output"]
)
def test_lmm_step_validation_of_field_that_should_hold_inference_parameter_when_valid_value_provided(
    field: str,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)
    input_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    step.validate_field_selector(field_name=field, input_step=input_step)

    # then - no error


@pytest.mark.parametrize(
    "field", ["prompt", "lmm_type", "remote_api_key", "json_output"]
)
def test_lmm_step_validation_of_field_that_should_hold_inference_parameter_when_invalid_value_provided(
    field: str,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)
    input_step = InferenceImage(type="InferenceImage", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name=field, input_step=input_step)


def test_lmm_step_validation_of_image_field_binding_when_valid_input_provided() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="image", value={"type": "url", "value": "https://some/image.jpg"}
    )

    # then - no error


@pytest.mark.parametrize("value", ["some", 1, True, []])
def test_lmm_step_validation_of_image_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="image", value=value)


def test_lmm_step_validation_of_prompt_field_binding_when_valid_input_provided() -> (
    None
):
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    step.validate_field_binding(field_name="prompt", value="My prompt")

    # then - no error


@pytest.mark.parametrize("value", [[], 1, True, None])
def test_lmm_step_validation_of_prompt_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="prompt", value=value)


@pytest.mark.parametrize("value", ["gpt_4v", "cog_vlm"])
def test_lmm_step_validation_of_lmm_type_field_binding_when_valid_input_provided(
    value: str,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    step.validate_field_binding(field_name="lmm_type", value=value)

    # then - no error


@pytest.mark.parametrize("value", [1, True, None])
def test_lmm_step_validation_of_lmm_type_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="lmm_type", value=value)


@pytest.mark.parametrize("value", [None, "xxx"])
def test_lmm_step_validation_of_remote_api_key_field_binding_when_valid_input_provided(
    value: Optional[str],
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    step.validate_field_binding(field_name="remote_api_key", value=value)

    # then - no error


@pytest.mark.parametrize("value", [1, True])
def test_lmm_step_validation_of_remote_api_key_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="remote_api_key", value=value)


@pytest.mark.parametrize("value", [None, {"a": "b"}])
def test_lmm_step_validation_of_json_output_field_binding_when_valid_input_provided(
    value: Optional[str],
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    step.validate_field_binding(field_name="json_output", value=value)

    # then - no error


@pytest.mark.parametrize(
    "value", [1, True, "some", {"raw_output": "this field name is forbidden"}]
)
def test_lmm_step_validation_of_json_output_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "image": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.json_output",
        "remote_api_key": "$inputs.api_key",
    }
    step = LMM.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="json_output", value=value)


def test_llm_for_classification_step_validation_when_valid_input_given() -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = LMMForClassification.parse_obj(specification)

    # then
    assert result == LMMForClassification(
        type="LMMForClassification",
        name="step_3",
        image="$steps.step_2.crops",
        lmm_type="$inputs.lmm_type",
        classes="$inputs.classification_classes",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", [1, "some", [], True])
def test_llm_for_classification_step_validation_when_invalid_image_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": value,
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMMForClassification.parse_obj(specification)


@pytest.mark.parametrize("value", [1, "some", [], True])
def test_llm_for_classification_step_validation_when_invalid_image_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": value,
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMMForClassification.parse_obj(specification)


@pytest.mark.parametrize("value", ["$inputs.model", "gpt_4v", "cog_vlm"])
def test_llm_for_classification_step_validation_when_lmm_type_valid(value: Any) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": value,
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = LMMForClassification.parse_obj(specification)

    assert result == LMMForClassification(
        type="LMMForClassification",
        name="step_3",
        image="$steps.step_2.crops",
        lmm_type=value,
        classes="$inputs.classification_classes",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", ["some", None])
def test_llm_for_classification_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": value,
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMMForClassification.parse_obj(specification)


@pytest.mark.parametrize("value", ["$inputs.classes", ["a"], ["a", "b"]])
def test_llm_for_classification_step_validation_when_classes_field_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": value,
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = LMMForClassification.parse_obj(specification)

    assert result == LMMForClassification(
        type="LMMForClassification",
        name="step_3",
        image="$steps.step_2.crops",
        lmm_type="gpt_4v",
        classes=value,
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", ["some", None, [], [1, 2]])
def test_llm_for_classification_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": value,
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMMForClassification.parse_obj(specification)


@pytest.mark.parametrize("value", ["$inputs.api_key", "some", None])
def test_llm_for_classification_step_validation_when_remote_api_key_field_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
        "remote_api_key": value,
    }

    # when
    result = LMMForClassification.parse_obj(specification)

    assert result == LMMForClassification(
        type="LMMForClassification",
        name="step_3",
        image="$steps.step_2.crops",
        lmm_type="gpt_4v",
        classes=["a", "b"],
        lmm_config=LMMConfig(),
        remote_api_key=value,
    )


@pytest.mark.parametrize("value", [[], 1])
def test_llm_for_classification_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
        "remote_api_key": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = LMMForClassification.parse_obj(specification)


def test_llm_for_classification_step_validation_of_field_selector_when_field_is_not_selector() -> (
    None
):
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
        "remote_api_key": "xx",
    }
    step = LMMForClassification.parse_obj(specification)
    input_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name="lmm_type", input_step=input_step)


def test_llm_for_classification_step_validation_of_field_selector_when_valid_image_given() -> (
    None
):
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
        "remote_api_key": "xx",
    }
    step = LMMForClassification.parse_obj(specification)
    input_step = InferenceImage(type="InferenceImage", name="some")

    # when
    step.validate_field_selector(field_name="image", input_step=input_step)

    # then - no error


def test_llm_for_classification_step_validation_of_field_selector_when_invalid_image_given() -> (
    None
):
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
        "remote_api_key": "xx",
    }
    step = LMMForClassification.parse_obj(specification)
    input_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name="image", input_step=input_step)


@pytest.mark.parametrize(
    "field", ["lmm_type", "classes", "remote_api_key", "remote_api_key"]
)
def test_lmm_for_classification_step_validation_of_field_that_should_hold_inference_parameter_when_valid_value_provided(
    field: str,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)
    input_step = InferenceParameter(type="InferenceParameter", name="some")

    # when
    step.validate_field_selector(field_name=field, input_step=input_step)

    # then - no error


@pytest.mark.parametrize(
    "field", ["lmm_type", "classes", "remote_api_key", "remote_api_key"]
)
def test_lmm_for_classification_step_validation_of_field_that_should_hold_inference_parameter_when_invalid_value_provided(
    field: str,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)
    input_step = InferenceImage(type="InferenceImage", name="some")

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(field_name=field, input_step=input_step)


def test_lmm_for_classification_step_validation_of_image_field_binding_when_valid_input_provided() -> (
    None
):
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="image", value={"type": "url", "value": "https://some/image.jpg"}
    )

    # then - no error


@pytest.mark.parametrize("value", ["some", 1, True, []])
def test_lmm_for_classification_step_validation_of_image_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="image", value=value)


@pytest.mark.parametrize("value", ["cog_vlm", "gpt_4v"])
def test_lmm_for_classification_step_validation_of_lmm_type_field_binding_when_valid_input_provided(
    value: str,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="lmm_type",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize("value", [None, "some", [], 1, {}])
def test_lmm_for_classification_step_validation_of_lmm_type_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="lmm_type", value=value)


@pytest.mark.parametrize("value", ["cog_vlm", None])
def test_lmm_for_classification_step_validation_of_remote_api_key_field_binding_when_valid_input_provided(
    value: str,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="remote_api_key",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize("value", [[], 1, {}])
def test_lmm_for_classification_step_validation_of_remote_api_key_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="remote_api_key", value=value)


@pytest.mark.parametrize("value", [["a", "b"]])
def test_lmm_for_classification_step_validation_of_classes_field_binding_when_valid_input_provided(
    value: str,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="classes",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize("value", [[], ["a", 1], 1, {}, "some", None])
def test_lmm_for_classification_step_validation_of_classes_field_binding_when_invalid_input_provided(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "image": "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classes",
        "remote_api_key": "$inputs.remote_api_key",
    }
    step = LMMForClassification.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(field_name="classes", value=value)
