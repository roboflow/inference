import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.foundation.ocr import (
    BlockManifest,
)


def test_ocr_model_validation_when_valid_manifest_is_given() -> None:
    # given
    data = {
        "type": "OCRModel",
        "name": "some",
        "image": "$steps.crop.crops",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="OCRModel",
        name="some",
        image="$steps.crop.crops",
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
        _ = BlockManifest.model_validate(data)
