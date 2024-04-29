from typing import Any

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.roboflow.instance_segmentation import (
    BlockManifest,
)


@pytest.mark.parametrize(
    "type_alias", ["RoboflowInstanceSegmentationModel", "InstanceSegmentationModel"]
)
def test_instance_segmentation_model_validation_when_minimalistic_config_is_provided(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "image", "model_id"])
@pytest.mark.parametrize(
    "type_alias", ["RoboflowInstanceSegmentationModel", "InstanceSegmentationModel"]
)
def test_instance_segmentation_model_validation_when_required_field_is_not_given(
    field: str,
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


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
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "type_alias", ["RoboflowInstanceSegmentationModel", "InstanceSegmentationModel"]
)
def test_instance_segmentation_model_validation_when_model_id_has_invalid_type(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "type_alias", ["RoboflowInstanceSegmentationModel", "InstanceSegmentationModel"]
)
def test_instance_segmentation_model_validation_when_active_learning_flag_has_invalid_type(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


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
@pytest.mark.parametrize(
    "type_alias", ["RoboflowInstanceSegmentationModel", "InstanceSegmentationModel"]
)
def test_instance_segmentation_model_when_parameters_have_invalid_type(
    parameter: str,
    value: Any,
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        parameter: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)
