import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.easy_ocr.v1 import (
    BlockManifest,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/easy_ocr@v1", "EasyOCR"])
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_ocr_model_validation_when_valid_manifest_is_given(
    type_alias: str,
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        images_field_alias: "$steps.crop.crops",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        images="$steps.crop.crops",
    )


def test_ocr_model_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "OCRModel",
        "name": "some",
        "images": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)
