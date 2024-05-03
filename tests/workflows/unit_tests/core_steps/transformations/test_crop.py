import numpy as np
import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.transformations.crop import (
    BlockManifest,
    crop_image,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_crop_validation_when_valid_manifest_is_given(images_field_alias: str) -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        images_field_alias: "$inputs.image",
        "predictions": "$steps.detection.predictions",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="Crop",
        name="some",
        images="$inputs.image",
        predictions="$steps.detection.predictions",
    )


def test_crop_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "images": "invalid",
        "predictions": "$steps.detection.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_crop_image() -> None:
    # given
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    origin_size = {"height": 1000, "width": 1000}
    predictions = [
        {"x": 10, "y": 10, "width": 20, "height": 20, "detection_id": "one"},
        {"x": 100, "y": 100, "width": 40, "height": 40, "detection_id": "two"},
        {"x": 500, "y": 500, "width": 100, "height": 100, "detection_id": "three"},
    ]
    image[0:20, 0:20] = 39
    image[80:120, 80:120] = 49
    image[450:550, 450:550] = 59

    # when
    result = crop_image(image=image, detections=predictions, origin_size=origin_size)

    # then
    assert len(result) == 3, "Expected 3 crops to be created"
    assert (
        result[0]["crops"]["type"] == "numpy_object"
    ), "Type of image is expected to be numpy_object"
    assert (
        result[0]["crops"]["value"] == (np.ones((20, 20, 3), dtype=np.uint8) * 39)
    ).all(), "Image must have expected size and color"
    assert (
        result[0]["crops"]["parent_id"] == "one"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[0]["crops"]["origin_coordinates"] == {
        "height": 20,
        "width": 20,
        "left_top_x": 0,
        "left_top_y": 0,
        "origin_image_size": {"height": 1000, "width": 1000},
    }, "Appropriate origin coordinates must be attached"
    assert (
        result[1]["crops"]["type"] == "numpy_object"
    ), "Type of image is expected to be numpy_object"
    assert (
        result[1]["crops"]["value"] == (np.ones((40, 40, 3), dtype=np.uint8) * 49)
    ).all(), "Image must have expected size and color"
    assert (
        result[1]["crops"]["parent_id"] == "two"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[1]["crops"]["origin_coordinates"] == {
        "left_top_x": 80,
        "left_top_y": 80,
        "height": 40,
        "width": 40,
        "origin_image_size": {"height": 1000, "width": 1000},
    }, "Appropriate origin coordinates must be attached"
    assert (
        result[2]["crops"]["type"] == "numpy_object"
    ), "Type of image is expected to be numpy_object"
    assert (
        result[2]["crops"]["value"] == (np.ones((100, 100, 3), dtype=np.uint8) * 59)
    ).all(), "Image must have expected size and color"
    assert (
        result[2]["crops"]["parent_id"] == "three"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[2]["crops"]["origin_coordinates"] == {
        "left_top_x": 450,
        "left_top_y": 450,
        "height": 100,
        "width": 100,
        "origin_image_size": {"height": 1000, "width": 1000},
    }, "Appropriate origin coordinates must be attached"
