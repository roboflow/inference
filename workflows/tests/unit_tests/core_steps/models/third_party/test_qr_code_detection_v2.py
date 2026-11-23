from typing import Tuple, Union

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.models.third_party.qr_code_detection.v1 import (
    QRCodeDetectorBlockV1,
)
from roboflow_workflows.core_steps.models.third_party.qr_code_detection.v2 import (
    BlockManifest,
    QRCodeDetectorBlockV2,
)
from roboflow_workflows.execution_engine.constants import (
    DETECTED_CODE_KEY,
    IMAGE_DIMENSIONS_KEY,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)

SMALL_CODE_TEXT = "https://inference.roboflow.com"


def test_manifest_parsing_when_scale_factor_is_not_given() -> None:
    # given
    data = {
        "type": "roboflow_core/qr_code_detector@v2",
        "name": "qr_code_detector",
        "images": "$inputs.image",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.scale_factor == 1.0


@pytest.mark.parametrize("scale_factor", [1.0, 2.5, 4.0, "$inputs.qr_scale_factor"])
def test_manifest_parsing_when_scale_factor_is_valid(
    scale_factor: Union[float, str],
) -> None:
    # given
    data = {
        "type": "roboflow_core/qr_code_detector@v2",
        "name": "qr_code_detector",
        "images": "$inputs.image",
        "scale_factor": scale_factor,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.scale_factor == scale_factor


@pytest.mark.parametrize("scale_factor", [0.5, 4.5])
def test_manifest_parsing_when_scale_factor_is_out_of_range(
    scale_factor: float,
) -> None:
    # given
    data = {
        "type": "roboflow_core/qr_code_detector@v2",
        "name": "qr_code_detector",
        "images": "$inputs.image",
        "scale_factor": scale_factor,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_qr_code_detection_matches_v1_when_scale_factor_is_one(
    qr_codes_image: np.ndarray,
) -> None:
    # given
    images = _single_image_batch(numpy_image=qr_codes_image)

    # when
    v1_result = QRCodeDetectorBlockV1().run(images=images)
    v2_result = QRCodeDetectorBlockV2().run(images=images, scale_factor=1.0)

    # then
    v1_predictions = v1_result[0]["predictions"]
    v2_predictions = v2_result[0]["predictions"]
    assert len(v2_predictions) == 3
    # OpenCV can list the same codes in a different order from one call to the next.
    assert np.array_equal(
        _sorted_boxes(v2_predictions.xyxy), _sorted_boxes(v1_predictions.xyxy)
    )
    assert sorted(v2_predictions[DETECTED_CODE_KEY]) == sorted(
        v1_predictions[DETECTED_CODE_KEY]
    )


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_qr_code_detection_when_code_is_too_small_to_read_without_zoom(
    quarter_turns: int,
) -> None:
    # given
    image, expected_box = _small_code_image(quarter_turns=quarter_turns)
    step = QRCodeDetectorBlockV2()

    # when
    unzoomed = step.run(images=_single_image_batch(numpy_image=image), scale_factor=1.0)
    zoomed = step.run(images=_single_image_batch(numpy_image=image), scale_factor=2.0)

    # then
    assert len(unzoomed[0]["predictions"]) == 0
    predictions = zoomed[0]["predictions"]
    assert list(predictions[DETECTED_CODE_KEY]) == [SMALL_CODE_TEXT]
    assert np.allclose(predictions.xyxy, [expected_box], atol=1.5)
    assert np.array_equal(predictions[IMAGE_DIMENSIONS_KEY], [image.shape[:2]])


def test_qr_code_detection_when_image_has_no_qr_code_and_zoom_is_on() -> None:
    # given
    step = QRCodeDetectorBlockV2()
    blank_image = np.full((480, 640, 3), 255, dtype=np.uint8)

    # when
    result = step.run(
        images=_single_image_batch(numpy_image=blank_image), scale_factor=3.0
    )

    # then
    assert len(result[0]["predictions"]) == 0


@pytest.mark.parametrize("scale_factor", [0.5, 5.0])
def test_qr_code_detection_when_scale_factor_is_out_of_range_at_runtime(
    scale_factor: float,
) -> None:
    # given
    # A selector skips manifest validation, so the block checks the value itself.
    step = QRCodeDetectorBlockV2()
    image, _ = _small_code_image(quarter_turns=0)

    # when
    with pytest.raises(ValueError):
        _ = step.run(
            images=_single_image_batch(numpy_image=image), scale_factor=scale_factor
        )


def _small_code_image(quarter_turns: int) -> Tuple[np.ndarray, np.ndarray]:
    """A QR code drawn at 2 pixels per module, which OpenCV can't read as it is.

    Returns the image and the box around the code's dark modules.
    """
    code = cv2.QRCodeEncoder.create().encode(SMALL_CODE_TEXT)
    code = cv2.resize(code, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    canvas[150 : 150 + code.shape[0], 260 : 260 + code.shape[1]] = code
    canvas = np.ascontiguousarray(np.rot90(canvas, k=quarter_turns))
    ys, xs = np.nonzero(canvas < 128)
    expected_box = np.array(
        [xs.min() - 0.5, ys.min() - 0.5, xs.max() + 0.5, ys.max() + 0.5]
    )
    image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    return image, expected_box


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
