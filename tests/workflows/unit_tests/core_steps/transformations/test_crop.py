import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.transformations.crop import BlockManifest


def test_crop_validation_when_valid_manifest_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
    }

    # when
    result = BlockManifest.parse_obj(data)

    # then
    assert result == BlockManifest(
        type="Crop",
        name="some",
        image="$inputs.image",
        predictions="$steps.detection.predictions",
    )


def test_crop_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "image": "invalid",
        "predictions": "$steps.detection.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.parse_obj(data)
