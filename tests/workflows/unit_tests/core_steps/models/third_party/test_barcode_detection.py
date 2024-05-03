import numpy as np
import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.third_party.barcode_detection import (
    BarcodeDetectorBlock,
    BlockManifest,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize("type_alias", ["BarcodeDetector", "BarcodeDetection"])
def test_manifest_parsing_when_data_is_valid(
    images_field_alias: str, type_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        images_field_alias: "$inputs.image",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        images="$inputs.image",
    )


def test_manifest_parsing_when_image_is_invalid_valid() -> None:
    # given
    data = {
        "type": "BarcodeDetector",
        "name": "some",
        "images": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.asyncio
async def test_barcode_detection(barcode_image: np.ndarray) -> None:
    # given
    step = BarcodeDetectorBlock()

    # when
    result = await step.run_locally(
        [{"type": "numpy_object", "value": barcode_image, "parent_id": "$inputs.image"}]
    )

    # then
    actual_parent_id = result[0]["parent_id"]
    assert actual_parent_id == "$inputs.image"

    values = ["47205255193", "37637448832", "21974251554", "81685630817"]
    actual_predictions = result[0]["predictions"]
    assert len(actual_predictions) == 4
    for prediction in actual_predictions:
        assert prediction["class"] == "barcode"
        assert prediction["class_id"] == 0
        assert prediction["confidence"] == 1.0
        assert prediction["x"] > 0
        assert prediction["y"] > 0
        assert prediction["width"] > 0
        assert prediction["height"] > 0
        assert prediction["detection_id"] is not None
        assert prediction["data"] in values
        assert prediction["parent_id"] == "$inputs.image"

    actual_image = result[0]["image"]
    assert actual_image["height"] == 480
    assert actual_image["width"] == 800

    actual_prediction_type = result[0]["prediction_type"]
    assert actual_prediction_type == "barcode-detection"
