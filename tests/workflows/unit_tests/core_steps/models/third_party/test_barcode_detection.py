import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.third_party.barcode_detection import (
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
    preds = result[0]["predictions"]
    assert len(preds) == 4
    for class_id, (x1, y1, x2, y2), class_name, detection_id, parent_id, confidence, data in \
            zip(preds.class_id, preds.xyxy, preds["class_name"], preds["detection_id"], preds["parent_id"], preds.confidence, preds["data"]):
        assert class_name == "barcode"
        assert class_id == 0
        assert confidence == 1.0
        assert x1 > 0
        assert y1 > 0
        assert x2-x1 > 0
        assert y2-y1 > 0
        assert detection_id is not None
        assert data in values
        assert parent_id == "$inputs.image"

    actual_image = result[0]["image"]
    assert actual_image["height"] == 480
    assert actual_image["width"] == 800

    actual_prediction_type = result[0]["prediction_type"]
    assert actual_prediction_type == "barcode-detection"
