from typing import Tuple

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.serializers import serialise_sv_detections
from roboflow_workflows.core_steps.models.third_party.qr_code_detection.v1 import (
    BlockManifest,
    QRCodeDetectorBlockV1,
)
from roboflow_workflows.execution_engine.constants import IMAGE_DIMENSIONS_KEY
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize(
    "type_alias",
    ["roboflow_core/qr_code_detector@v1", "QRCodeDetector", "QRCodeDetection"],
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
        "type": "QRCodeDetector",
        "name": "some",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_qr_code_detection(qr_codes_image: np.ndarray) -> None:
    # given
    step = QRCodeDetectorBlockV1()
    images = Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="$inputs.image"),
                numpy_image=qr_codes_image,
            )
        ],
        indices=[(0,)],
    )

    # when
    result = step.run(images=images)

    # then
    actual_parent_id = result[0]["predictions"]["parent_id"]
    assert (actual_parent_id == "$inputs.image").all()
    preds = result[0]["predictions"]
    assert len(preds) == 3
    for (
        class_id,
        (x1, y1, x2, y2),
        class_name,
        detection_id,
        parent_id,
        confidence,
        url,
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
        preds["data"],
        preds.data["prediction_type"],
        preds.data["image_dimensions"],
        preds.data["root_parent_id"],
        preds.data["root_parent_coordinates"],
        preds.data["root_parent_dimensions"],
        preds.data["parent_coordinates"],
        preds.data["parent_dimensions"],
    ):
        assert class_name == "qr_code"
        assert class_id == 0
        assert confidence == 1.0
        assert x1 > 0
        assert y1 > 0
        assert x2 - x1 > 0
        assert y2 - y1 > 0
        assert detection_id is not None
        assert url == "https://www.qrfy.com/LEwG_Gj"
        assert parent_id == "$inputs.image"
        assert prediction_type == "qrcode-detection"
        assert np.allclose(image_dimensions, np.array(qr_codes_image.shape[:2]))
        assert root_parent_id == "$inputs.image"
        assert np.allclose(root_parent_coordinates, np.array([0, 0]))
        assert np.allclose(root_parent_dimensions, np.array(qr_codes_image.shape[:2]))
        assert np.allclose(parent_coordinates, np.array([0, 0]))
        assert np.allclose(parent_dimensions, np.array(qr_codes_image.shape[:2]))


def test_qr_code_detection_when_image_has_no_qr_code() -> None:
    # given
    step = QRCodeDetectorBlockV1()
    blank_image = np.full((480, 640, 3), 255, dtype=np.uint8)

    # when
    result = step.run(images=_single_image_batch(numpy_image=blank_image))

    # then
    predictions = result[0]["predictions"]
    assert len(predictions) == 0
    assert predictions.metadata[IMAGE_DIMENSIONS_KEY] == [480, 640]
    serialized = serialise_sv_detections(detections=predictions)
    assert serialized["image"] == {"width": 640, "height": 480}
    assert serialized["predictions"] == []


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_qr_code_detection_when_codes_are_rotated(
    qr_codes_image: np.ndarray, quarter_turns: int
) -> None:
    # given
    step = QRCodeDetectorBlockV1()
    upright_result = step.run(images=_single_image_batch(numpy_image=qr_codes_image))
    expected_boxes = _rotate_boxes(
        upright_result[0]["predictions"].xyxy,
        image_shape=qr_codes_image.shape,
        quarter_turns=quarter_turns,
    )
    rotated_image = np.ascontiguousarray(np.rot90(qr_codes_image, k=quarter_turns))

    # when
    result = step.run(images=_single_image_batch(numpy_image=rotated_image))

    # then
    preds = result[0]["predictions"]
    assert len(preds) == 3
    assert (preds.xyxy[:, 2] > preds.xyxy[:, 0]).all()
    assert (preds.xyxy[:, 3] > preds.xyxy[:, 1]).all()
    assert np.allclose(
        _sorted_boxes(preds.xyxy), _sorted_boxes(expected_boxes), atol=2.0
    )
    assert list(preds["data"]) == ["https://www.qrfy.com/LEwG_Gj"] * 3


def test_qr_code_detection_when_code_is_tilted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    # A code turned 45 degrees clockwise. OpenCV lists corners starting from the
    # code's own top-left, which now sits at the top of the diamond.
    corners = np.array([[[50, 0], [100, 50], [50, 100], [0, 50]]], dtype=np.float32)
    monkeypatch.setattr(
        cv2, "QRCodeDetector", lambda: _FakeQRCodeDetector(corners=corners)
    )
    step = QRCodeDetectorBlockV1()
    image = np.full((120, 120, 3), 255, dtype=np.uint8)

    # when
    result = step.run(images=_single_image_batch(numpy_image=image))

    # then
    assert np.allclose(result[0]["predictions"].xyxy, [[0, 0, 100, 100]])


class _FakeQRCodeDetector:
    """Stands in for cv2.QRCodeDetector and reports fixed corners."""

    def __init__(self, corners: np.ndarray) -> None:
        self._corners = corners

    def detectAndDecodeMulti(
        self, image: np.ndarray
    ) -> Tuple[bool, Tuple[str, ...], np.ndarray, None]:
        decoded = ("tilted",) * len(self._corners)

        return True, decoded, self._corners, None


def _single_image_batch(numpy_image: np.ndarray) -> Batch[WorkflowImageData]:
    batch = Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="$inputs.image"),
                numpy_image=numpy_image,
            )
        ],
        indices=[(0,)],
    )

    return batch


def _rotate_boxes(
    boxes: np.ndarray, *, image_shape: Tuple[int, ...], quarter_turns: int
) -> np.ndarray:
    """Move xyxy boxes into the same image turned by np.rot90 (counterclockwise)."""
    height, width = image_shape[:2]
    rotated = np.asarray(boxes, dtype=float)
    for _ in range(quarter_turns):
        x1, y1, x2, y2 = rotated.T
        rotated = np.stack([y1, (width - 1) - x2, y2, (width - 1) - x1], axis=1)
        height, width = width, height

    return rotated


def _sorted_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    order = np.lexsort((boxes[:, 1], boxes[:, 0]))

    return boxes[order]
