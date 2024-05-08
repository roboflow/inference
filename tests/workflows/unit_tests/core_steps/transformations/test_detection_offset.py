from typing import Any

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.detection_offset import (
    BlockManifest,
    offset_detection,
)


@pytest.mark.parametrize("offset_width_alias", ["offset_width", "offset_x"])
@pytest.mark.parametrize("offset_height_alias", ["offset_height", "offset_y"])
def test_manifest_parsing_when_valid_data_provided(
    offset_width_alias: str,
    offset_height_alias: str,
) -> None:
    # given
    data = {
        "type": "DetectionOffset",
        "name": "some",
        "predictions": "$steps.some.predictions",
        offset_width_alias: "$inputs.offset_x",
        offset_height_alias: 40,
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="DetectionOffset",
        name="some",
        predictions="$steps.some.predictions",
        offset_width="$inputs.offset_x",
        offset_height=40,
        image_metadata="$steps.some.image",
        prediction_type="$steps.some.prediction_type",
    )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("predictions", "invalid"),
        ("offset_width", -1),
        ("offset_height", -1),
        ("image_metadata", "invalid"),
        ("prediction_type", "invalid"),
    ],
)
def test_manifest_parsing_when_invalid_data_provided(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "DetectionOffset",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "offset_width": "$inputs.offset_x",
        "offset_height": 40,
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }
    data[field_name] = value

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_offset_detection() -> None:
    # given
    detection = {
        "x": 100,
        "y": 200,
        "width": 20,
        "height": 20,
        "parent_id": "p2",
        "detection_id": "two",
        "class_name": "car",
        "confidence": 0.5,
    }

    # when
    result = offset_detection(
        detection=detection,
        offset_width=50,
        offset_height=100,
    )

    # then
    assert result["x"] == 100, "OX center should not be changed"
    assert result["y"] == 200, "OY center should not be changed"
    assert result["width"] == 70, "Width should be offset by 50px"
    assert result["height"] == 120, "Height should be offset by 100px"
    assert (
        result["parent_id"] == "two"
    ), "Parent id should be set to origin detection id"
    assert (
        result["detection_id"] != detection["detection_id"]
    ), "New detection id (random) must be assigned"
