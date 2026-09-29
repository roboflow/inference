from typing import Union

import numpy as np
import pytest
import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES

from inference.core.workflows.core_steps.fusion.detections_stitch.v1 import (
    BlockManifest,
    DetectionsStitchBlockV1,
)
from inference.core.workflows.execution_engine.constants import (
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "overlap_filtering_strategy",
    ["none", "nms", "nmm", "$inputs.some"],
)
@pytest.mark.parametrize(
    "iou_threshold",
    [0.5, "$inputs.some"],
)
def test_detections_stitch_v1_manifest_parsing_when_input_valid(
    overlap_filtering_strategy: str,
    iou_threshold: Union[float, str],
) -> None:
    raw_manifest = {
        "type": "roboflow_core/detections_stitch@v1",
        "name": "stitch",
        "reference_image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "overlap_filtering_strategy": overlap_filtering_strategy,
        "iou_threshold": iou_threshold,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/detections_stitch@v1",
        name="stitch",
        reference_image="$inputs.image",
        predictions="$steps.model.predictions",
        overlap_filtering_strategy=overlap_filtering_strategy,
        iou_threshold=iou_threshold,
    )


def test_detections_stitch_v1_manifest_parsing_when_overlap_mode_invalid() -> None:
    raw_manifest = {
        "type": "roboflow_core/detections_stitch@v1",
        "name": "stitch",
        "reference_image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "overlap_filtering_strategy": "invalid",
        "iou_threshold": 0.5,
    }

    # when
    with pytest.raises(ValueError):
        _ = BlockManifest.model_validate(raw_manifest)


def make_test_image(
    width: int,
    height: int,
    left_top_x: int = 0,
    left_top_y: int = 0,
    parent_id: str = "reference",
) -> WorkflowImageData:
    """Create a test WorkflowImageData with specified dimensions."""
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id=parent_id,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=left_top_x,
                left_top_y=left_top_y,
                origin_width=width,
                origin_height=height,
            ),
        ),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def make_test_detections(
    boxes: np.ndarray,
    parent_offset: tuple,
    parent_dims: tuple,
    with_mask: bool = False,
    mask_shape: tuple = None,
    confidence: np.ndarray = None,
    class_id: np.ndarray = None,
) -> sv.Detections:
    """Create test sv.Detections with parent metadata."""
    n_detections = len(boxes)

    mask = None
    if with_mask and mask_shape is not None:
        mask = np.zeros((n_detections, mask_shape[0], mask_shape[1]), dtype=np.bool_)
        for i in range(n_detections):
            x1, y1, x2, y2 = boxes[i].astype(int)
            y1_m = max(0, min(y1, mask_shape[0] - 1))
            y2_m = max(0, min(y2, mask_shape[0]))
            x1_m = max(0, min(x1, mask_shape[1] - 1))
            x2_m = max(0, min(x2, mask_shape[1]))
            if y2_m > y1_m and x2_m > x1_m:
                mask[i, y1_m:y2_m, x1_m:x2_m] = True

    if confidence is None:
        confidence = np.ones(n_detections) * 0.9
    if class_id is None:
        class_id = np.zeros(n_detections, dtype=int)

    return sv.Detections(
        xyxy=boxes,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        data={
            "class_name": np.array([f"class_{i}" for i in class_id]),
            PARENT_COORDINATES_KEY: np.array([parent_offset] * n_detections),
            PARENT_DIMENSIONS_KEY: np.array([parent_dims] * n_detections),
        },
    )


def test_detections_stitch_basic_stitching_without_masks() -> None:
    """Test basic stitching of detections from two crops without masks."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=1000, height=1000)
    predictions = [
        make_test_detections(
            boxes=np.array([[10, 10, 50, 50]]),
            parent_offset=(100, 100),
            parent_dims=(1000, 1000),
        ),
        make_test_detections(
            boxes=np.array([[20, 20, 60, 60]]),
            parent_offset=(500, 500),
            parent_dims=(1000, 1000),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    assert "predictions" in result
    merged = result["predictions"]
    assert len(merged) == 2
    assert np.allclose(merged.xyxy[0], [110, 110, 150, 150])  # +100, +100
    assert np.allclose(merged.xyxy[1], [520, 520, 560, 560])  # +500, +500


def test_detections_stitch_with_masks() -> None:
    """Test stitching of detections with masks."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=500, height=400)
    predictions = [
        make_test_detections(
            boxes=np.array([[10, 10, 50, 50]]),
            parent_offset=(50, 50),
            parent_dims=(400, 500),
            with_mask=True,
            mask_shape=(100, 100),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    assert merged.mask is not None
    assert len(merged) == 1
    assert merged.mask.shape[1:] == (400, 500)


def test_detections_stitch_verify_mask_dimensions_match_reference() -> None:
    """Test that all masks are resized to match reference image dimensions (the key fix)."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=800, height=600)
    predictions = [
        make_test_detections(
            boxes=np.array([[5, 5, 25, 25]]),
            parent_offset=(0, 0),
            parent_dims=(600, 800),
            with_mask=True,
            mask_shape=(100, 100),
        ),
        make_test_detections(
            boxes=np.array([[10, 10, 40, 40]]),
            parent_offset=(200, 200),
            parent_dims=(600, 800),
            with_mask=True,
            mask_shape=(200, 150),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    assert merged.mask is not None
    assert len(merged) == 2
    assert merged.mask.shape == (2, 600, 800)


def test_detections_stitch_multiple_crops_different_dimensions() -> None:
    """Test stitching detections from multiple crops with varying dimensions."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=1000, height=1000)
    predictions = [
        make_test_detections(
            boxes=np.array([[10, 10, 30, 30]]),
            parent_offset=(0, 0),
            parent_dims=(1000, 1000),
            with_mask=True,
            mask_shape=(100, 100),
        ),
        make_test_detections(
            boxes=np.array([[15, 15, 45, 45]]),
            parent_offset=(300, 300),
            parent_dims=(1000, 1000),
            with_mask=True,
            mask_shape=(150, 200),
        ),
        make_test_detections(
            boxes=np.array([[20, 20, 50, 50]]),
            parent_offset=(600, 600),
            parent_dims=(1000, 1000),
            with_mask=True,
            mask_shape=(200, 100),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    assert len(merged) == 3
    assert merged.mask is not None
    assert merged.mask.shape == (3, 1000, 1000)


def test_detections_stitch_empty_predictions() -> None:
    """Test handling of empty predictions list."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=500, height=500)
    predictions = []

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    assert "predictions" in result
    merged = result["predictions"]
    assert len(merged) == 0


def test_detections_stitch_single_prediction() -> None:
    """Test stitching with a single prediction."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=500, height=500)
    predictions = [
        make_test_detections(
            boxes=np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
            parent_offset=(100, 100),
            parent_dims=(500, 500),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    assert len(merged) == 2
    assert np.allclose(merged.xyxy[0], [110, 110, 150, 150])
    assert np.allclose(merged.xyxy[1], [160, 160, 200, 200])


def test_detections_stitch_mix_empty_and_non_empty() -> None:
    """Test stitching with a mix of empty and non-empty detections."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=500, height=500)
    predictions = [
        make_test_detections(
            boxes=np.array([[10, 10, 50, 50]]),
            parent_offset=(100, 100),
            parent_dims=(500, 500),
        ),
        sv.Detections.empty(),
        make_test_detections(
            boxes=np.array([[20, 20, 60, 60]]),
            parent_offset=(200, 200),
            parent_dims=(500, 500),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    assert len(merged) == 2
    assert np.allclose(merged.xyxy[0], [110, 110, 150, 150])
    assert np.allclose(merged.xyxy[1], [220, 220, 260, 260])


def test_detections_stitch_parent_coordinates_attached() -> None:
    """Test that parent coordinates are correctly attached to stitched detections."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(
        width=500,
        height=1000,
        left_top_x=50,
        left_top_y=100,
        parent_id="reference_img",
    )
    predictions = [
        make_test_detections(
            boxes=np.array([[10, 10, 50, 50]]),
            parent_offset=(50, 100),
            parent_dims=(1000, 500),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    assert merged.data["parent_id"][0] == "reference_img"
    assert np.allclose(merged.data["parent_coordinates"][0], [50, 100])
    assert np.allclose(merged.data["parent_dimensions"][0], [1000, 500])


def test_detections_stitch_offset_handling() -> None:
    """Test that detections are correctly moved by their offsets."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=1000, height=1000)
    predictions = [
        make_test_detections(
            boxes=np.array([[0, 0, 50, 50]]),
            parent_offset=(250, 300),
            parent_dims=(1000, 1000),
        ),
    ]

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    expected_box = np.array([[250, 300, 300, 350]])
    assert np.allclose(merged.xyxy, expected_box)


def test_detections_stitch_scaling_detection_error() -> None:
    """Test that error is raised when scaling is detected (unsupported)."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=500, height=500)
    detections = make_test_detections(
        boxes=np.array([[10, 10, 50, 50]]),
        parent_offset=(100, 100),
        parent_dims=(500, 500),
    )
    detections.data[SCALING_RELATIVE_TO_PARENT_KEY] = np.array([0.5])
    predictions = [detections]

    # when
    with pytest.raises(ValueError) as exc_info:
        block.run(
            reference_image=reference_image,
            predictions=predictions,
            overlap_filtering_strategy="none",
            iou_threshold=0.3,
        )

    # then
    assert "Scaled bounding boxes" in str(exc_info.value)


def test_detections_stitch_missing_parent_coordinates_error() -> None:
    """Test that error is raised when parent coordinates are missing."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=500, height=500)
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["class_0"]),
            # Missing PARENT_COORDINATES_KEY
        },
    )

    predictions = [detections]

    # when
    with pytest.raises(RuntimeError) as exc_info:
        block.run(
            reference_image=reference_image,
            predictions=predictions,
            overlap_filtering_strategy="none",
            iou_threshold=0.3,
        )

    # then
    assert "parent_coordinates" in str(exc_info.value)


@pytest.mark.parametrize(
    "parent_offset",
    [
        pytest.param((250, 300), id="positive-offset"),
        pytest.param((0, 0), id="zero-offset"),
        pytest.param((100, 0), id="x-only-offset"),
    ],
)
def test_detections_stitch_shifts_oriented_bounding_box_corners(
    parent_offset: tuple,
) -> None:
    """OBB corners stored in `data['xyxyxyxy']` must follow the same offset
    applied to `xyxy`; without that, stitched OBB detections carry an AABB
    in image coords next to corners still in tile-local coords, which breaks
    downstream OBB-aware NMS / NMM."""
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=1000, height=1000)
    tile_corners = np.array(
        [[[25.0, 50.0], [50.0, 25.0], [75.0, 50.0], [50.0, 75.0]]],
        dtype=np.float32,
    )
    detections = make_test_detections(
        boxes=np.array([[25.0, 25.0, 75.0, 75.0]], dtype=np.float32),
        parent_offset=parent_offset,
        parent_dims=(1000, 1000),
    )
    detections.data[ORIENTED_BOX_COORDINATES] = tile_corners

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=[detections],
        overlap_filtering_strategy="none",
        iou_threshold=0.3,
    )

    # then
    merged = result["predictions"]
    dx, dy = parent_offset
    expected_xyxy = np.array([[25 + dx, 25 + dy, 75 + dx, 75 + dy]], dtype=np.float32)
    expected_corners = tile_corners + np.array([dx, dy], dtype=np.float32)
    assert np.allclose(merged.xyxy, expected_xyxy)
    assert np.allclose(merged.data[ORIENTED_BOX_COORDINATES], expected_corners)


def _dense_reference_stitch(
    predictions: list,
    reference_wh: tuple,
    strategy: str,
    iou_threshold: float,
) -> sv.Detections:
    """The pre-lazy algorithm, kept as the semantic oracle: move every mask to
    a full-size dense array, merge, then filter."""
    from copy import deepcopy

    moved = []
    for detections in predictions:
        detections = deepcopy(detections)
        offset = detections.data[PARENT_COORDINATES_KEY][0][:2].copy()
        detections.xyxy = sv.move_boxes(xyxy=detections.xyxy, offset=offset)
        detections.mask = sv.move_masks(
            masks=detections.mask, offset=offset, resolution_wh=reference_wh
        )
        moved.append(detections)
    merged = sv.Detections.merge(moved)
    if strategy == "none":
        return merged
    if strategy == "nms":
        return merged.with_nms(threshold=iou_threshold)
    return merged.with_nmm(threshold=iou_threshold)


def _random_slice_detections(
    rng: np.random.Generator,
    *,
    count: int,
    slice_shape: tuple,
    parent_offset: tuple,
    parent_dims: tuple,
) -> sv.Detections:
    slice_h, slice_w = slice_shape
    masks = np.zeros((count, slice_h, slice_w), dtype=bool)
    boxes = np.zeros((count, 4), dtype=np.float32)
    for i in range(count):
        w, h = rng.integers(20, slice_w // 3), rng.integers(20, slice_h // 3)
        x1, y1 = rng.integers(0, slice_w - w), rng.integers(0, slice_h - h)
        # non-rectangular, so a box-only shortcut could not pass as mask parity
        masks[i, y1 : y1 + h, x1 : x1 + w] = True
        masks[i, y1 : y1 + h // 2, x1 : x1 + w // 3] = False
        boxes[i] = [x1, y1, x1 + w, y1 + h]
    return sv.Detections(
        xyxy=boxes,
        mask=masks,
        confidence=rng.uniform(0.3, 1.0, size=count).astype(np.float32),
        class_id=rng.integers(0, 2, size=count),
        data={
            "class_name": np.array(["a"] * count),
            PARENT_COORDINATES_KEY: np.array([parent_offset] * count),
            PARENT_DIMENSIONS_KEY: np.array([parent_dims] * count),
        },
    )


@pytest.mark.parametrize("strategy", ["none", "nms", "nmm"])
def test_detections_stitch_mask_stitching_matches_dense_reference(
    strategy: str,
) -> None:
    # given - four overlapping slices, the last one hanging past the reference
    # edge so clipping is exercised too
    rng = np.random.default_rng(7)
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=300, height=200)
    offsets = [(0, 0), (80, 0), (0, 80), (220, 130)]
    predictions = [
        _random_slice_detections(
            rng,
            count=6,
            slice_shape=(100, 100),
            parent_offset=offset,
            parent_dims=(200, 300),
        )
        for offset in offsets
    ]
    expected = _dense_reference_stitch(
        predictions, reference_wh=(300, 200), strategy=strategy, iou_threshold=0.3
    )

    # when
    result = block.run(
        reference_image=reference_image,
        predictions=predictions,
        overlap_filtering_strategy=strategy,
        iou_threshold=0.3,
    )["predictions"]

    # then - identical geometry, order and masks; inputs untouched
    assert len(result) == len(expected)
    assert np.array_equal(result.xyxy, expected.xyxy)
    assert np.array_equal(result.confidence, expected.confidence)
    assert isinstance(result.mask, np.ndarray)
    assert result.mask.dtype == np.bool_
    assert result.mask.shape == (len(expected), 200, 300)
    assert np.array_equal(result.mask, expected.mask)
    assert predictions[0].mask.shape == (6, 100, 100)
    assert predictions[0].xyxy.max() < 100


def test_detections_stitch_rejects_crops_with_and_without_masks() -> None:
    # given
    block = DetectionsStitchBlockV1()
    reference_image = make_test_image(width=300, height=200)
    predictions = [
        make_test_detections(
            boxes=np.array([[10, 10, 50, 50]]),
            parent_offset=(0, 0),
            parent_dims=(200, 300),
            with_mask=True,
            mask_shape=(100, 100),
        ),
        make_test_detections(
            boxes=np.array([[10, 10, 50, 50]]),
            parent_offset=(100, 0),
            parent_dims=(200, 300),
        ),
    ]

    # when / then
    with pytest.raises(ValueError):
        block.run(
            reference_image=reference_image,
            predictions=predictions,
            overlap_filtering_strategy="none",
            iou_threshold=0.3,
        )


def test_detections_stitch_peak_memory_is_bounded_for_sliced_segmentation() -> None:
    """2026-09-28: image_slicer -> sam3 -> detections_stitch on a 1080p shelf
    clip OOM-killed an 8Gi video worker inside this block: 12 slices x ~25
    masks were each re-allocated at full frame size, merged, then mask-NMS'd,
    about 3 GiB per frame. Masks must stay crop-scoped until the survivors of
    overlap filtering are known."""
    import tracemalloc

    rng = np.random.default_rng(3)
    block = DetectionsStitchBlockV1()
    width, height = 1920, 1012
    reference_image = make_test_image(width=width, height=height)
    offsets = [(x, y) for y in (0, 372, 372) for x in (0, 427, 854, 1280)]
    predictions = [
        _random_slice_detections(
            rng,
            count=25,
            slice_shape=(640, 640),
            parent_offset=offset,
            parent_dims=(height, width),
        )
        for offset in offsets
    ]

    tracemalloc.start()
    try:
        result = block.run(
            reference_image=reference_image,
            predictions=predictions,
            overlap_filtering_strategy="nms",
            iou_threshold=0.3,
        )["predictions"]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(result) > 0
    assert result.mask.shape[1:] == (height, width)
    # the dense path measured ~3 GiB here; survivors alone are ~2 MB each
    survivors_bytes = len(result) * height * width
    assert peak < survivors_bytes + 512 * 1024 * 1024, f"peak={peak / 2**30:.2f} GiB"
