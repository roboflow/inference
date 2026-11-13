import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.analytics.overlap.v1 import (
    OUTPUT_KEY,
    OverlapBlockV1,
)


def test_coords_overlap():
    assert not OverlapBlockV1.coords_overlap(
        [0, 0, 20, 20], [15, 15, 35, 35], "Center Overlap"
    )
    assert not OverlapBlockV1.coords_overlap(
        [10, 10, 20, 20], [30, 30, 40, 40], "Any Overlap"
    )
    assert OverlapBlockV1.coords_overlap(
        [20, 20, 30, 30], [15, 15, 35, 35], "Center Overlap"
    )
    assert OverlapBlockV1.coords_overlap(
        [0, 0, 20, 20], [15, 15, 35, 35], "Any Overlap"
    )


def _make_segmentation_detections(
    xyxy: np.ndarray,
    masks: np.ndarray,
    class_names: list,
) -> sv.Detections:
    n = len(xyxy)
    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=np.ones(n) * 0.9,
        class_id=np.arange(n),
        data={"class_name": np.array(class_names)},
    )


def test_overlap_run_masks_no_false_positive():
    """Boxes overlap but masks don't, so nothing should be returned (#1987)."""
    # given
    image_h, image_w = 100, 100

    xyxy = np.array(
        [
            [0, 0, 100, 100],  # container, box covers the whole image
            [60, 10, 90, 40],  # item, top right
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, image_h, image_w), dtype=bool)
    masks[0, 50:100, 0:50] = True  # container mask only in the bottom left
    masks[1, 10:40, 60:90] = True  # item mask in the top right

    detections = _make_segmentation_detections(
        xyxy=xyxy, masks=masks, class_names=["container", "item"]
    )

    assert not np.logical_and(
        masks[0], masks[1]
    ).any(), "test setup: masks should not overlap"

    block = OverlapBlockV1()

    # when
    result_any = block.run(
        predictions=detections,
        overlap_type="Any Overlap",
        overlap_class_name="container",
    )
    result_center = block.run(
        predictions=detections,
        overlap_type="Center Overlap",
        overlap_class_name="container",
    )

    # then
    assert (
        len(result_any[OUTPUT_KEY]) == 0
    ), "masks do not intersect, expected no overlap"
    assert (
        len(result_center[OUTPUT_KEY]) == 0
    ), "masks do not intersect, expected no overlap"


def test_masks_overlap():
    image_h, image_w = 100, 100

    # boxes are float32 like real predictions, so the slicing has to
    # handle float coordinates
    overlap_mask = np.zeros((image_h, image_w), dtype=bool)
    overlap_mask[50:100, 0:50] = True
    overlap_bbox = np.array([0, 50, 50, 100], dtype=np.float32)

    # far away from the overlap mask
    other_mask_far = np.zeros((image_h, image_w), dtype=bool)
    other_mask_far[10:40, 60:90] = True
    other_bbox_far = np.array([60, 10, 90, 40], dtype=np.float32)

    # inside the overlap mask
    other_mask_inside = np.zeros((image_h, image_w), dtype=bool)
    other_mask_inside[60:80, 20:40] = True
    other_bbox_inside = np.array([20, 60, 40, 80], dtype=np.float32)

    # no overlap
    assert not OverlapBlockV1.masks_overlap(
        overlap_mask=overlap_mask,
        overlap_bbox=overlap_bbox,
        other_mask=other_mask_far,
        other_bbox=other_bbox_far,
        overlap_type="Center Overlap",
    )
    assert not OverlapBlockV1.masks_overlap(
        overlap_mask=overlap_mask,
        overlap_bbox=overlap_bbox,
        other_mask=other_mask_far,
        other_bbox=other_bbox_far,
        overlap_type="Any Overlap",
    )

    # overlap
    assert OverlapBlockV1.masks_overlap(
        overlap_mask=overlap_mask,
        overlap_bbox=overlap_bbox,
        other_mask=other_mask_inside,
        other_bbox=other_bbox_inside,
        overlap_type="Center Overlap",
    )
    assert OverlapBlockV1.masks_overlap(
        overlap_mask=overlap_mask,
        overlap_bbox=overlap_bbox,
        other_mask=other_mask_inside,
        other_bbox=other_bbox_inside,
        overlap_type="Any Overlap",
    )


def test_overlap_run_bbox_fallback_without_masks():
    """Without masks the block should still use boxes, as before."""
    # given
    xyxy = np.array(
        [
            [0, 0, 100, 100],  # container
            [10, 10, 50, 50],  # item, inside the container box
        ],
        dtype=np.float32,
    )

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([0, 1]),
        data={"class_name": np.array(["container", "item"])},
    )

    block = OverlapBlockV1()

    # when
    result = block.run(
        predictions=detections,
        overlap_type="Any Overlap",
        overlap_class_name="container",
    )

    # then
    assert len(result[OUTPUT_KEY]) == 1
    assert result[OUTPUT_KEY].data["class_name"][0] == "item"


def test_overlap_run_with_masks_true_positive():
    """Masks overlap, so the item should be returned."""
    # given
    image_h, image_w = 100, 100

    xyxy = np.array(
        [
            [0, 0, 100, 100],  # container
            [20, 60, 40, 80],  # item, inside the container mask
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, image_h, image_w), dtype=bool)
    masks[0, 50:100, 0:50] = True  # container mask in the bottom left
    masks[1, 60:80, 20:40] = True  # item mask inside it

    detections = _make_segmentation_detections(
        xyxy=xyxy, masks=masks, class_names=["container", "item"]
    )

    assert np.logical_and(masks[0], masks[1]).any(), "test setup: masks should overlap"

    block = OverlapBlockV1()

    # when
    result = block.run(
        predictions=detections,
        overlap_type="Any Overlap",
        overlap_class_name="container",
    )

    # then
    assert len(result[OUTPUT_KEY]) == 1
    assert result[OUTPUT_KEY].data["class_name"][0] == "item"


def test_overlap_run_no_overlap_at_all():
    # given
    image_h, image_w = 100, 100

    xyxy = np.array(
        [
            [0, 0, 30, 30],  # container, top left
            [70, 70, 90, 90],  # item, bottom right
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, image_h, image_w), dtype=bool)
    masks[0, 0:30, 0:30] = True
    masks[1, 70:90, 70:90] = True

    detections = _make_segmentation_detections(
        xyxy=xyxy, masks=masks, class_names=["container", "item"]
    )

    block = OverlapBlockV1()

    # when
    result = block.run(
        predictions=detections,
        overlap_type="Any Overlap",
        overlap_class_name="container",
    )

    # then
    assert len(result[OUTPUT_KEY]) == 0
