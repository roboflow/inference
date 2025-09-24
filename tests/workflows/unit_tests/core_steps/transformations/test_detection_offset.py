from typing import Any

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.detection_offset.v1 import (
    BlockManifest,
    offset_detections,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/detection_offset@v1", "DetectionOffset"]
)
@pytest.mark.parametrize("offset_width_alias", ["offset_width", "offset_x"])
@pytest.mark.parametrize("offset_height_alias", ["offset_height", "offset_y"])
def test_manifest_parsing_when_valid_data_provided(
    type_alias: str,
    offset_width_alias: str,
    offset_height_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
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
        type=type_alias,
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
    }
    data[field_name] = value

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_offset_detection() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[90, 190, 110, 210]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["two"]),
            "class_name": np.array(["car"]),
            "parent_id": np.array(["p2"]),
            "image_dimensions": np.array([[640, 640]]),
        },
    )

    # when
    result = offset_detections(
        detections=detections,
        offset_width=50,
        offset_height=100,
    )

    # then
    x1, y1, x2, y2 = result.xyxy[0]
    assert x1 == 65, "Left corner should be moved by 25px to the left"
    assert y1 == 140, "Top corner should be moved by 50px to the top"
    assert x2 == 135, "Right corner should be moved by 25px to the right"
    assert y2 == 260, "Right corner should be moved by 50px to the bottom"
    assert result["parent_id"] == str(
        detections["detection_id"][0]
    ), "Parent id should be set to origin detection id"
    assert result["detection_id"] != str(
        detections["parent_id"][0]
    ), "New detection id (random) must be assigned"


def test_offset_detection_when_larger_than_image() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[90, 190, 110, 210]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["two"]),
            "class_name": np.array(["car"]),
            "parent_id": np.array(["p2"]),
            "image_dimensions": np.array([[640, 640]]),
        },
    )

    # when
    result = offset_detections(
        detections=detections,
        offset_width=2000,
        offset_height=2000,
    )

    # then
    x1, y1, x2, y2 = result.xyxy[0]
    assert x1 == 0, "Left corner should be at the image border"
    assert y1 == 0, "Top corner should be at the image border"
    assert x2 == 640, "Right corner should be at the image border"
    assert y2 == 640, "Bottom corner should be at the image border"
    assert result["parent_id"] == str(
        detections["detection_id"][0]
    ), "Parent id should be set to origin detection id"
    assert result["detection_id"] != str(
        detections["parent_id"][0]
    ), "New detection id (random) must be assigned"


def test_offset_detection_when_nothing_predicted() -> None:
    # given
    detections = sv.Detections.empty()

    # when
    result = offset_detections(
        detections=detections,
        offset_width=50,
        offset_height=100,
    )

    # then
    assert len(detections) == 0, "Expected empty detections in output"


def test_offset_detection_with_percentage() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[100, 200, 400, 600]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["three"]),
            "class_name": np.array(["truck"]),
            "parent_id": np.array(["p3"]),
            "image_dimensions": np.array([[640, 640]]),
        },
    )

    # when
    result = offset_detections(
        detections=detections, offset_width=10, offset_height=10, use_percentage=True
    )

    # then
    x1, y1, x2, y2 = result.xyxy[0]
    assert x1 == 85, "Left corner should be moved by 5% of detection width to the left"
    assert y1 == 180, "Top corner should be moved by 5% of detection height to the top"
    assert (
        x2 == 415
    ), "Right corner should be moved by 5% of detection width to the right"
    assert (
        y2 == 620
    ), "Bottom corner should be moved by 5% of detection height to the bottom"
