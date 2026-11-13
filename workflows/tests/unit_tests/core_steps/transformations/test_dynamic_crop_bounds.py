from copy import deepcopy
from typing import Optional, Tuple

import cv2
import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import crop_image
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from supervision.config import ORIENTED_BOX_COORDINATES


@pytest.fixture
def image_pixels() -> np.ndarray:
    return np.arange(20 * 30 * 3, dtype=np.uint16).astype(np.uint8).reshape(20, 30, 3)


@pytest.mark.parametrize(
    "box, clipped",
    [
        pytest.param((3, 5, 14, 17), (3, 5, 14, 17), id="inside"),
        pytest.param((-6, 4, 12, 16), (0, 4, 12, 16), id="left-overlap"),
        pytest.param((4, -6, 16, 12), (4, 0, 16, 12), id="top-overlap"),
        pytest.param((-40, -30, 50, 40), (0, 0, 30, 20), id="oversized"),
        pytest.param((23, 14, 40, 30), (23, 14, 30, 20), id="bottom-right"),
        pytest.param((-20, 4, -5, 16), None, id="fully-left"),
        pytest.param((4, -15, 16, -3), None, id="fully-above"),
        pytest.param((35, 4, 40, 16), None, id="fully-right"),
        pytest.param((4, 25, 16, 30), None, id="fully-below"),
        pytest.param((4, 5, 4, 16), None, id="zero-width"),
        pytest.param((4, 5, 16, 5), None, id="zero-height"),
        pytest.param((16, 5, 4, 17), None, id="inverted"),
        pytest.param((-0.5, 2.5, 10.5, 12.5), (0, 2, 10, 12), id="round-before-clip"),
    ],
)
def test_dynamic_crop_intersects_the_image_canvas(
    image_pixels: np.ndarray,
    box: Tuple[float, float, float, float],
    clipped: Optional[Tuple[int, int, int, int]],
) -> None:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=image_pixels,
    )
    detections = sv.Detections(
        xyxy=np.array([box], dtype=np.float64),
        class_id=np.array([2]),
        confidence=np.array([0.8]),
        data={"detection_id": np.array(["object"])},
    )

    result = crop_image(image, detections, mask_opacity=0.0, background_color=(0, 0, 0))

    assert len(result) == 1
    if clipped is None:
        assert result == [{"crops": None, "predictions": None}]
        return

    x1, y1, x2, y2 = clipped
    np.testing.assert_array_equal(
        result[0]["crops"].numpy_image, image_pixels[y1:y2, x1:x2]
    )
    origin = result[0]["crops"].parent_metadata.origin_coordinates
    assert (origin.left_top_x, origin.left_top_y) == (x1, y1)
    # Keep the original prediction's geometry; only its coordinate frame moves.
    np.testing.assert_array_equal(
        result[0]["predictions"].xyxy,
        np.array([box]) - np.array([x1, y1, x1, y1]),
    )
    np.testing.assert_array_equal(detections.xyxy, np.array([box]))


@pytest.mark.parametrize("mask_opacity", [0.0, 0.5, 1.0])
def test_dynamic_crop_uses_the_clipped_region_for_masks_and_geometry(
    image_pixels: np.ndarray, mask_opacity: float
) -> None:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=image_pixels,
    )
    mask = np.zeros((1, 20, 30), dtype=bool)
    mask[:, 7:11, 4:9] = True
    points = np.array([[[2, 7], [8, 10]]], dtype=np.float32)
    corners = np.array([[[0, 5], [10, 5], [10, 15], [0, 15]]], dtype=np.float32)
    detections = sv.Detections(
        xyxy=np.array([[-10, 5, 12, 40]], dtype=np.float64),
        mask=mask,
        class_id=np.array([2]),
        confidence=np.array([0.8]),
        tracker_id=np.array([17]),
        data={
            "detection_id": np.array(["object"]),
            "class_name": np.array(["thing"]),
            "keypoints_xy": points.copy(),
            "polygon": points.copy(),
            ORIENTED_BOX_COORDINATES: corners.copy(),
        },
        metadata={"retained": "source"},
    )
    original = deepcopy(detections)
    original_pixels = image_pixels.copy()

    result = crop_image(
        image, detections, mask_opacity=mask_opacity, background_color=(14, 26, 38)
    )[0]

    expected_mask = mask[:, 5:20, 0:12]
    expected_pixels = image_pixels[5:20, 0:12].copy()
    if mask_opacity > 0:
        background = np.array([38, 26, 14], dtype=np.uint8)
        overlaid = np.where(expected_mask[0, :, :, None], expected_pixels, background)
        # The established overlay blends masked and original pixels with uint8 rounding.
        expected_pixels = cv2.addWeighted(
            overlaid, mask_opacity, expected_pixels, 1.0 - mask_opacity, 0
        )

    np.testing.assert_array_equal(result["crops"].numpy_image, expected_pixels)
    prediction = result["predictions"]
    np.testing.assert_array_equal(prediction.mask, expected_mask)
    np.testing.assert_array_equal(prediction.xyxy, [[-10, 0, 12, 35]])
    np.testing.assert_array_equal(prediction["keypoints_xy"], points - [0, 5])
    np.testing.assert_array_equal(prediction["polygon"], points - [0, 5])
    np.testing.assert_array_equal(
        prediction[ORIENTED_BOX_COORDINATES], corners - [0, 5]
    )
    np.testing.assert_array_equal(prediction.class_id, original.class_id)
    np.testing.assert_array_equal(prediction.confidence, original.confidence)
    np.testing.assert_array_equal(prediction.tracker_id, original.tracker_id)
    assert prediction.metadata == original.metadata
    assert result["crops"].parent_metadata.parent_id == "object"
    origin = result["crops"].parent_metadata.origin_coordinates
    assert (origin.left_top_x, origin.left_top_y) == (0, 5)
    np.testing.assert_array_equal(image.numpy_image, original_pixels)
    np.testing.assert_array_equal(detections.xyxy, original.xyxy)
    np.testing.assert_array_equal(detections.mask, original.mask)
    for key in original.data:
        np.testing.assert_array_equal(detections[key], original[key])


def test_dynamic_crop_preserves_mixed_detection_order(image_pixels: np.ndarray) -> None:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"), numpy_image=image_pixels
    )
    detections = sv.Detections(
        xyxy=np.array([[-6, 4, 12, 16], [-20, 4, -5, 16], [3, 5, 14, 17]], dtype=float),
        data={"detection_id": np.array(["partial", "outside", "inside"])},
    )

    result = crop_image(image, detections, mask_opacity=0.0, background_color=(0, 0, 0))

    assert len(result) == 3
    assert result[0]["crops"].parent_metadata.parent_id == "partial"
    assert result[1] == {"crops": None, "predictions": None}
    assert result[2]["crops"].parent_metadata.parent_id == "inside"
    np.testing.assert_array_equal(
        result[0]["crops"].numpy_image, image_pixels[4:16, :12]
    )
    np.testing.assert_array_equal(
        result[2]["crops"].numpy_image, image_pixels[5:17, 3:14]
    )
