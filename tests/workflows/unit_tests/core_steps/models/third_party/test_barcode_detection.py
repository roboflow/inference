import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.third_party.barcode_detection.v1 import (
    BarcodeDetectorBlockV1,
    BlockManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize(
    "type_alias",
    ["roboflow_core/barcode_detector@v1", "BarcodeDetector", "BarcodeDetection"],
)
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


def test_barcode_detection(barcode_image: np.ndarray) -> None:
    # given
    step = BarcodeDetectorBlockV1()
    images = Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="$inputs.image"),
                numpy_image=barcode_image,
            )
        ],
        indices=[(0,)],
    )

    # when
    result = step.run(images=images)

    # then
    actual_parent_id = result[0]["predictions"]["parent_id"]
    assert (actual_parent_id == "$inputs.image").all()

    values = ["47205255193", "37637448832", "21974251554", "81685630817"]
    preds = result[0]["predictions"]
    assert len(preds) == 4
    for (
        class_id,
        (x1, y1, x2, y2),
        class_name,
        detection_id,
        parent_id,
        confidence,
        data,
        prediction_type,
        image_dimensions,
        root_parent_id,
        root_parent_coordinates,
        root_parent_dimensions,
        parent_coordinates,
        parent_dimensions,
    ) in zip(
        preds.class_id,
        preds.xyxy,
        preds["class_name"],
        preds["detection_id"],
        preds["parent_id"],
        preds.confidence,
        preds.data["data"],
        preds.data["prediction_type"],
        preds.data["image_dimensions"],
        preds.data["root_parent_id"],
        preds.data["root_parent_coordinates"],
        preds.data["root_parent_dimensions"],
        preds.data["parent_coordinates"],
        preds.data["parent_dimensions"],
    ):
        assert class_name == "barcode"
        assert class_id == 0
        assert confidence == 1.0
        assert x1 > 0
        assert y1 > 0
        assert x2 - x1 > 0
        assert y2 - y1 > 0
        assert detection_id is not None
        assert data in values
        assert parent_id == "$inputs.image"
        assert prediction_type == "barcode-detection"
        assert np.allclose(image_dimensions, np.array(barcode_image.shape[:2]))
        assert root_parent_id == "$inputs.image"
        assert np.allclose(root_parent_coordinates, np.array([0, 0]))
        assert np.allclose(root_parent_dimensions, np.array(barcode_image.shape[:2]))
        assert np.allclose(parent_coordinates, np.array([0, 0]))
        assert np.allclose(parent_dimensions, np.array(barcode_image.shape[:2]))
