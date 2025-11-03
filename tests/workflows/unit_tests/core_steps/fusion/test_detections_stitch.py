from typing import Union

import numpy as np
import pytest
import supervision as sv

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
