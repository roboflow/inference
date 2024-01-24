from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference.enterprise.deployments.entities.inputs import (
    InferenceImage,
    InferenceParameter,
)
from inference.enterprise.deployments.entities.steps import (
    ClassificationModel,
    Condition,
    Crop,
    DetectionFilter,
    DetectionFilterDefinition,
    DetectionOffset,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    Operator, DetectionsConsensus, AggregationMode,
)
from inference.enterprise.deployments.errors import (
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
        _ = KeypointsDetectionModel.parse_obj(data)


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
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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


@pytest.mark.parametrize(
    "value",
    [3, "3", True, 3.0, [], set(), {}, None]
)
def test_detections_consensus_validation_when_predictions_of_invalid_type_given(value: Any) -> None:
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


@pytest.mark.parametrize(
    "value",
    [None, 0, -1, "some", []]
)
def test_detections_consensus_validation_when_required_votes_of_invalid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize(
    "value",
    [3, "$inputs.some"]
)
def test_detections_consensus_validation_when_required_votes_of_valid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result.required_votes == value


@pytest.mark.parametrize(
    "value",
    [None, "some"]
)
def test_detections_consensus_validation_when_class_aware_of_invalid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": 3,
        "class_aware": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize(
    "value",
    [True, False]
)
def test_detections_consensus_validation_when_class_aware_of_valid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
        ("confidence", "some")
    ]
)
def test_detections_consensus_validation_when_range_field_of_invalid_type_given(
    field: str,
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
    ]
)
def test_detections_consensus_validation_when_range_field_of_valid_type_given(
    field: str,
    value: Any,
) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": 3,
        field: value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert getattr(result, field) == value


@pytest.mark.parametrize(
    "value",
    ["some", 1, 2.0, True, {}]
)
def test_detections_consensus_validation_when_classes_to_consider_of_invalid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": 3,
        "classes_to_consider": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize(
    "value",
    ["$inputs.some", [], ["1", "2", "3"]]
)
def test_detections_consensus_validation_when_classes_to_consider_of_valid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": 3,
        "classes_to_consider": value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result.classes_to_consider == value


@pytest.mark.parametrize(
    "value",
    ["some", -1, 0, {"some": None}, {"some": 1, "other": -1}]
)
def test_detections_consensus_validation_when_required_objects_of_invalid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": 3,
        "required_objects": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = DetectionsConsensus.parse_obj(specification)


@pytest.mark.parametrize(
    "value",
    [None, "$inputs.some", 1, 10, {"some": 1, "other": 10}]
)
def test_detections_consensus_validation_when_required_objects_of_valid_type_given(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": 3,
        "required_objects": value,
    }

    # when
    result = DetectionsConsensus.parse_obj(specification)

    # then
    assert result.required_objects == value


def test_detections_consensus_validation_field_predictions_field_selector_when_index_is_not_given() -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": 3,
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(ExecutionGraphError):
        step.validate_field_selector(
            field_name="predictions",
            input_step=MagicMock(),
            index=None
        )


def test_detections_consensus_validation_field_predictions_field_selector_when_index_is_out_of_range() -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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


def test_detections_consensus_validation_field_predictions_field_selector_when_selector_does_not_hold_detections() -> None:
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
                model_id="some/1"
            ),
            index=1,
        )


def test_detections_consensus_validation_field_predictions_field_selector_when_selector_point_to_invalid_step() -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
                detections="$steps.step.predictions"
            ),
            index=1,
        )


def test_detections_consensus_validation_field_predictions_field_selector_when_selector_point_to_valid_step() -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
            model_id="some/1"
        ),
        index=1,
    )

    # then - no error


@pytest.mark.parametrize(
    "field",
    ["required_votes", "class_aware", "iou_threshold", "confidence", "classes_to_consider", "required_objects"]
)
def test_detections_consensus_validation_field_that_is_supposed_to_be_selector_but_is_not(field: str) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
    ["required_votes", "class_aware", "iou_threshold", "confidence", "classes_to_consider", "required_objects"]
)
def test_detections_consensus_validation_field_that_is_supposed_to_be_parameter_selector_but_is_not(field: str) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
                detections="$steps.step.predictions"
            ),
        )


@pytest.mark.parametrize(
    "value",
    [None, -1, "some", [], 0]
)
def test_detections_consensus_validate_field_binding_for_required_votes_when_value_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    with pytest.raises(VariableTypeError):
        step.validate_field_binding(
            field_name="required_votes",
            value=value,
        )


@pytest.mark.parametrize(
    "value",
    [1, 10]
)
def test_detections_consensus_validate_field_binding_for_required_votes_when_value_is_valid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
        "required_votes": "$inputs.some",
    }
    step = DetectionsConsensus.parse_obj(specification)

    # when
    step.validate_field_binding(
        field_name="required_votes",
        value=value,
    )

    # then - no error


@pytest.mark.parametrize(
    "value",
    [None, "some", []]
)
def test_detections_consensus_validate_field_binding_for_class_aware_when_value_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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


@pytest.mark.parametrize(
    "value",
    [True, False]
)
def test_detections_consensus_validate_field_binding_for_class_aware_when_value_is_valid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
    ]
)
def test_detections_consensus_validate_field_binding_for_zero_one_range_field_when_value_is_invalid(field: str, value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
    ]
)
def test_detections_consensus_validate_field_binding_for_zero_one_range_field_when_value_is_valid(field: str, value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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


@pytest.mark.parametrize(
    "value",
    ["some", 1, 2.0, True, {}, ["some", 1]]
)
def test_detections_consensus_validate_field_binding_for_classes_to_consider_when_value_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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


@pytest.mark.parametrize(
    "value",
    [None, ["A", "B"]]
)
def test_detections_consensus_validate_field_binding_for_classes_to_consider_when_value_is_valid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
    "value",
    ["some", -1, 0, ["some"], {"some": None}, {"some": 1, "other": -1}]
)
def test_detections_consensus_validate_field_binding_for_required_objects_when_value_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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


@pytest.mark.parametrize(
    "value",
    [None, 1, 3, {"some": 1, "other": 2}]
)
def test_detections_consensus_validate_field_binding_for_required_objects_when_value_is_valid(value: Any) -> None:
    # given
    specification = {
        "type": "DetectionsConsensus",
        "name": "some",
        "predictions": ["$steps.detection.predictions", "$steps.detection_2.predictions"],
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
