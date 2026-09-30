"""Pixel parity with supervision, and no full-frame mask materialisation."""

import numpy as np
import pytest
import supervision as sv
from supervision.detection.compact_mask import CompactMask

from inference.core.workflows.core_steps.visualizations.common.annotators.compact_polygon import (
    CompactPolygonAnnotator,
)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("thickness", [1, 2, 5])
def test_compact_contours_match_full_frame(seed, thickness, monkeypatch):
    rng = np.random.default_rng(seed)
    h, w = 60, 80
    masks = rng.random((5, h, w)) > 0.8
    masks[0, 5:40, 8:50] = True
    masks[0, 10:30, 12:35] = False  # a hole
    masks[1] = False
    masks[1, :20, :25] = True  # touches image edges
    boxes = np.array(
        [
            [0, 0, w - 1, h - 1],
            [0, 0, 24, 19],
            [15, 12, 60, 42],
            [-4, -5, 20, 30],
            [20, 20, 10, 10],
        ],
        dtype=np.float32,
    )
    compact = CompactMask.from_dense(masks, boxes, (h, w))
    detections = sv.Detections(
        xyxy=boxes, mask=compact, class_id=np.arange(5), confidence=np.ones(5)
    )
    scene = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    lookup = np.array([4, 3, 2, 1, 0])
    expected = sv.PolygonAnnotator(thickness=thickness).annotate(
        scene.copy(), detections, lookup
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Polygon rendering must use crops, not full-frame masks")

    monkeypatch.setattr(CompactMask, "__getitem__", forbidden)
    actual = CompactPolygonAnnotator(thickness=thickness).annotate(
        scene.copy(), detections, lookup
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("with_masks", [True, False])
def test_dense_and_no_mask_inputs_keep_supervision_behavior(with_masks):
    masks = np.ones((1, 20, 30), dtype=bool) if with_masks else None
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 29, 19]]), class_id=np.array([0]), mask=masks
    )
    scene = np.zeros((20, 30, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        CompactPolygonAnnotator().annotate(scene.copy(), detections),
        sv.PolygonAnnotator().annotate(scene.copy(), detections),
    )


def test_pil_image_input_keeps_image_type_and_pixels():
    from PIL import Image

    dense = np.zeros((1, 20, 30), dtype=bool)
    dense[0, 3:15, 4:25] = True
    boxes = np.array([[4, 3, 24, 14]])
    detections = sv.Detections(
        xyxy=boxes,
        class_id=np.array([0]),
        mask=CompactMask.from_dense(dense, boxes, (20, 30)),
    )
    scene = Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8))
    expected = sv.PolygonAnnotator().annotate(scene.copy(), detections)
    actual = CompactPolygonAnnotator().annotate(scene.copy(), detections)
    assert isinstance(actual, Image.Image)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
