from typing import Any

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.supervision_tools.detections_slicer import (
    BlockManifest,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize(
    "type_alias", ["RoboflowDetectionsInferenceSlicer", "DetectionsInferenceSlicer"]
)
def test_detections_slicer_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        images="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_detections_slicer_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "RoboflowDetectionsInferenceSlicer",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_detections_slicer_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_detections_slicer_validation_when_model_id_has_invalid_type() -> None:
    # given
    data = {
        "type": "RoboflowDetectionsInferenceSlicer",
        "name": "some",
        "images": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("images", "some"),
        ("class_agnostic_nms", "some"),
        ("class_filter", "some"),
        ("confidence", 1.1),
        ("confidence", "some"),
        ("iou_threshold", "some"),
        ("iou_threshold", 1.1),
        ("slice_width", "some"),
        ("slice_width", 0.5),
        ("slice_height", "some"),
        ("slice_height", 0.5),
        ("overlap_ratio_width", "some"),
        ("overlap_ratio_width", 1.1),
        ("overlap_ratio_height", "some"),
        ("overlap_ratio_height", 1.1),

    ],
)
def test_detections_slicer_when_parameters_have_invalid_type(
    parameter: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "RoboflowDetectionsInferenceSlicer",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        parameter: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)
