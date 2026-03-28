import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.overlap.v1 import (
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
    """Helper to build sv.Detections with segmentation masks."""
    n = len(xyxy)
    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=np.ones(n) * 0.9,
        class_id=np.arange(n),
        data={"class_name": np.array(class_names)},
    )


def test_overlap_run_masks_no_false_positive():
    """
    Regression test for GitHub Issue #1987:
    https://github.com/roboflow/inference/issues/1987

    When instance segmentation masks are present, the overlap block must
    use pixel-level mask overlap — not just bounding-box overlap.

    Scenario (100x100 image):
      "container" — bbox [0,0,100,100], mask covers ONLY bottom-left
                    quadrant (rows 50-99, cols 0-49)
      "item"      — bbox [60,10,90,40], in the TOP-RIGHT area

    Bounding boxes overlap, but masks do NOT.
    The block should report NO overlap.
    """
    image_h, image_w = 100, 100

    xyxy = np.array(
        [
            [0, 0, 100, 100],  # container — large bbox
            [60, 10, 90, 40],  # item — top-right bbox
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, image_h, image_w), dtype=bool)
    masks[0, 50:100, 0:50] = True  # container mask: bottom-left only
    masks[1, 10:40, 60:90] = True  # item mask: top-right

    detections = _make_segmentation_detections(
        xyxy=xyxy, masks=masks, class_names=["container", "item"]
    )

    # Sanity check: masks do NOT overlap at the pixel level
    assert not np.logical_and(masks[0], masks[1]).any(), (
        "Sanity check failed: masks should not overlap"
    )

    block = OverlapBlockV1()

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

    # Masks don't overlap, so no detections should be returned
    assert len(result_any[OUTPUT_KEY]) == 0, (
        "Expected no overlap — masks do not intersect"
    )
    assert len(result_center[OUTPUT_KEY]) == 0, (
        "Expected no overlap — masks do not intersect"
    )


def test_masks_overlap():
    """Unit tests for the masks_overlap classmethod, mirroring test_coords_overlap."""
    image_h, image_w = 100, 100

    # overlap mask: bottom-left quadrant
    overlap_mask = np.zeros((image_h, image_w), dtype=bool)
    overlap_mask[50:100, 0:50] = True

    # other mask in top-right — no pixel overlap
    other_mask_far = np.zeros((image_h, image_w), dtype=bool)
    other_mask_far[10:40, 60:90] = True
    other_bbox_far = [60, 10, 90, 40]

    # other mask inside overlap — pixel overlap exists
    other_mask_inside = np.zeros((image_h, image_w), dtype=bool)
    other_mask_inside[60:80, 20:40] = True
    other_bbox_inside = [20, 60, 40, 80]

    # No overlap cases
    assert not OverlapBlockV1.masks_overlap(
        overlap_mask, other_mask_far, other_bbox_far, "Center Overlap"
    )
    assert not OverlapBlockV1.masks_overlap(
        overlap_mask, other_mask_far, other_bbox_far, "Any Overlap"
    )

    # Overlap cases
    assert OverlapBlockV1.masks_overlap(
        overlap_mask, other_mask_inside, other_bbox_inside, "Center Overlap"
    )
    assert OverlapBlockV1.masks_overlap(
        overlap_mask, other_mask_inside, other_bbox_inside, "Any Overlap"
    )


def test_overlap_run_bbox_fallback_without_masks():
    """
    When predictions have no masks (plain object detection), the block
    falls back to bounding-box overlap — same behavior as before the fix.
    """
    xyxy = np.array(
        [
            [0, 0, 100, 100],  # container
            [10, 10, 50, 50],  # item — bbox inside container
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
    result = block.run(
        predictions=detections,
        overlap_type="Any Overlap",
        overlap_class_name="container",
    )

    # No masks → bbox overlap is used; bboxes overlap → item is found
    assert len(result[OUTPUT_KEY]) == 1
    assert result[OUTPUT_KEY].data["class_name"][0] == "item"


def test_overlap_run_with_masks_true_positive():
    """
    Control test: when masks DO overlap, the block should (and does)
    report them as overlapping. This verifies the block works correctly
    in the non-buggy case (where bbox overlap matches mask overlap).

    Scenario (100x100 image):
      "container" — bbox [0,0,100,100], mask covers bottom-left quadrant
      "item"      — bbox [20,60,40,80], placed INSIDE the container mask
    """
    image_h, image_w = 100, 100

    xyxy = np.array(
        [
            [0, 0, 100, 100],  # container
            [20, 60, 40, 80],  # item — inside container's mask
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, image_h, image_w), dtype=bool)
    masks[0, 50:100, 0:50] = True  # container mask: bottom-left
    masks[1, 60:80, 20:40] = True  # item mask: inside container mask

    detections = _make_segmentation_detections(
        xyxy=xyxy, masks=masks, class_names=["container", "item"]
    )

    # Sanity check: masks DO overlap
    assert np.logical_and(masks[0], masks[1]).any(), (
        "Sanity check failed: masks should overlap"
    )

    block = OverlapBlockV1()
    result = block.run(
        predictions=detections,
        overlap_type="Any Overlap",
        overlap_class_name="container",
    )

    # This correctly reports overlap (both bbox and mask agree)
    assert len(result[OUTPUT_KEY]) == 1
    assert result[OUTPUT_KEY].data["class_name"][0] == "item"


def test_overlap_run_no_overlap_at_all():
    """
    When neither bboxes nor masks overlap, the block correctly
    reports no overlap. Baseline sanity test.
    """
    image_h, image_w = 100, 100

    xyxy = np.array(
        [
            [0, 0, 30, 30],   # container — top-left
            [70, 70, 90, 90],  # item — bottom-right, far away
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
    result = block.run(
        predictions=detections,
        overlap_type="Any Overlap",
        overlap_class_name="container",
    )

    assert len(result[OUTPUT_KEY]) == 0
