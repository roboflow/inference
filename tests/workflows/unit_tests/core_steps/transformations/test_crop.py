import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.transformations.crop import BlockManifest


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
