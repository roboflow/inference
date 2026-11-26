from typing import Tuple

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("inference_models")

from roboflow_workflows.core_steps.models.third_party.qr_code_detection.v1 import (
    QRCodeDetectorBlockV1 as NumpyQRCodeDetectorBlockV1,
)
from roboflow_workflows.core_steps.models.third_party.qr_code_detection.v1_tensor import (
    QRCodeDetectorBlockV1,
)
from roboflow_workflows.execution_engine.constants import DETECTED_CODE_KEY
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)

from inference_models.models.base.object_detection import Detections


def test_qr_code_detection_when_image_has_no_qr_code() -> None:
    # given
    step = QRCodeDetectorBlockV1()
    blank_image = np.full((480, 640, 3), 255, dtype=np.uint8)

    # when
    result = step.run(images=_single_image_batch(numpy_image=blank_image))

    # then
    predictions = result[0]["predictions"]
    assert isinstance(predictions, Detections)
    assert tuple(predictions.xyxy.shape) == (0, 4)
    assert predictions.bboxes_metadata == []


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_qr_code_detection_when_codes_are_rotated(
    qr_codes_image: np.ndarray, quarter_turns: int
) -> None:
    # given
    step = QRCodeDetectorBlockV1()
    rotated_image = np.ascontiguousarray(np.rot90(qr_codes_image, k=quarter_turns))
    numpy_result = NumpyQRCodeDetectorBlockV1().run(
        images=_single_image_batch(numpy_image=rotated_image)
    )

    # when
    result = step.run(images=_single_image_batch(numpy_image=rotated_image))

    # then
    predictions = result[0]["predictions"]
    boxes = predictions.xyxy.cpu().numpy()
    assert boxes.shape == (3, 4)
    assert (boxes[:, 2] > boxes[:, 0]).all()
    assert (boxes[:, 3] > boxes[:, 1]).all()
    assert np.allclose(
        _sorted_boxes(boxes),
        _sorted_boxes(numpy_result[0]["predictions"].xyxy),
        atol=0.01,
    )
    assert [entry[DETECTED_CODE_KEY] for entry in predictions.bboxes_metadata] == [
        "https://www.qrfy.com/LEwG_Gj"
    ] * 3


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
    boxes = result[0]["predictions"].xyxy.cpu().numpy()
    assert np.allclose(boxes, [[0, 0, 100, 100]])


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


def _sorted_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    order = np.lexsort((boxes[:, 1], boxes[:, 0]))

    return boxes[order]
