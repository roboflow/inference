import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.third_party.qr_code_detection import (
    BlockManifest,
    QRCodeDetectorBlock,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize("type_alias", ["QRCodeDetector", "QRCodeDetection"])
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
        "type": "QRCodeDetector",
        "name": "some",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.asyncio
async def test_qr_code_detection(qr_codes_image: np.ndarray) -> None:
    # given
    step = QRCodeDetectorBlock()

    # when
    result = await step.run_locally(
        [
            {
                "type": "numpy_object",
                "value": qr_codes_image,
                "parent_id": "$inputs.image",
            }
        ]
    )

    # then
    actual_parent_id = result[0]["parent_id"]
    assert actual_parent_id == "$inputs.image"

    actual_predictions = result[0]["predictions"]
    assert len(actual_predictions) == 3
    for prediction in actual_predictions:
        assert prediction["class"] == "qr_code"
        assert prediction["class_id"] == 0
        assert prediction["confidence"] == 1.0
        assert prediction["x"] > 0
        assert prediction["y"] > 0
        assert prediction["width"] > 0
        assert prediction["height"] > 0
        assert prediction["detection_id"] is not None
        assert prediction["data"] == "https://www.qrfy.com/LEwG_Gj"
        assert prediction["parent_id"] == "$inputs.image"

    actual_image = result[0]["image"]
    assert actual_image["height"] == 1018
    assert actual_image["width"] == 2470

    actual_prediction_type = result[0]["prediction_type"]
    assert actual_prediction_type == "qrcode-detection"
