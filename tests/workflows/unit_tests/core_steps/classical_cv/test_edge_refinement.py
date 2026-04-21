import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.edge_refinement.v1 import (
    OUTPUT_EDGE_VIS_KEY,
    OUTPUT_PREDICTIONS_KEY,
    EdgeRefinementBlockV1,
    EdgeRefinementManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


# ---------------------------------------------------------------------------
# Tests for EdgeRefinementManifest
# ---------------------------------------------------------------------------


def test_edge_refinement_manifest_valid_creation():
    # when
    manifest = EdgeRefinementManifest(
        type="roboflow_core/edge_refinement@v1",
        name="edge_refinement_1",
        image="$inputs.image",
        segmentation="$steps.segmentation.predictions",
    )

    # then
    assert manifest.type == "roboflow_core/edge_refinement@v1"
    assert manifest.name == "edge_refinement_1"
    assert manifest.image == "$inputs.image"
    assert manifest.segmentation == "$steps.segmentation.predictions"


def test_edge_refinement_manifest_default_parameters():
    # when
    manifest = EdgeRefinementManifest(
        type="roboflow_core/edge_refinement@v1",
        name="edge_refinement_1",
        image="$inputs.image",
        segmentation="$steps.segmentation.predictions",
    )

    # then — check all defaults
    assert manifest.pixel_tolerance == 5
    assert manifest.low_threshold == 50
    assert manifest.high_threshold == 150
    assert manifest.blur_kernel_size == 7
    assert manifest.min_contour_area == 10.0
    assert manifest.dilation_iterations == 0
    assert manifest.band_width == 15
    assert manifest.tangent_window == 5
    assert manifest.clahe_clip_limit == 3.0
    assert manifest.clahe_tile_grid_size == 8
    assert manifest.bilateral_sigma_color == 75.0
    assert manifest.bilateral_sigma_space == 75.0
    assert manifest.dilation_kernel_size == 3


def test_edge_refinement_manifest_custom_parameters():
    # when
    manifest = EdgeRefinementManifest(
        type="roboflow_core/edge_refinement@v1",
        name="edge_refinement_1",
        image="$inputs.image",
        segmentation="$steps.segmentation.predictions",
        pixel_tolerance=10,
        low_threshold=40,
        high_threshold=120,
        blur_kernel_size=5,
        min_contour_area=20.0,
        dilation_iterations=2,
        band_width=20,
        tangent_window=8,
        clahe_clip_limit=2.5,
        clahe_tile_grid_size=16,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=60.0,
        dilation_kernel_size=5,
    )

    # then
    assert manifest.pixel_tolerance == 10
    assert manifest.low_threshold == 40
    assert manifest.high_threshold == 120
    assert manifest.blur_kernel_size == 5
    assert manifest.min_contour_area == 20.0
    assert manifest.dilation_iterations == 2
    assert manifest.band_width == 20
    assert manifest.tangent_window == 8
    assert manifest.clahe_clip_limit == 2.5
    assert manifest.clahe_tile_grid_size == 16
    assert manifest.bilateral_sigma_color == 50.0
    assert manifest.bilateral_sigma_space == 60.0
    assert manifest.dilation_kernel_size == 5


def test_edge_refinement_manifest_validation_when_invalid_image_is_given():
    # given
    data = {
        "type": "roboflow_core/edge_refinement@v1",
        "name": "edge_refinement_1",
        "image": "invalid",
        "segmentation": "$steps.segmentation.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = EdgeRefinementManifest.model_validate(data)


def test_edge_refinement_manifest_describe_outputs():
    # when
    outputs = EdgeRefinementManifest.describe_outputs()

    # then
    assert len(outputs) == 2
    output_names = {output.name for output in outputs}
    assert OUTPUT_PREDICTIONS_KEY in output_names
    assert OUTPUT_EDGE_VIS_KEY in output_names


def test_edge_refinement_manifest_get_execution_engine_compatibility():
    # when
    compat = EdgeRefinementManifest.get_execution_engine_compatibility()

    # then
    assert compat == ">=1.3.0,<2.0.0"


# ---------------------------------------------------------------------------
# Tests for EdgeRefinementBlockV1 with empty/missing masks
# ---------------------------------------------------------------------------


def test_edge_refinement_block_with_no_masks(dogs_image: np.ndarray) -> None:
    # given
    block = EdgeRefinementBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 100, 150]], dtype=np.float32),
    )

    # when
    result = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=dogs_image,
        ),
        segmentation=detections,
        pixel_tolerance=5,
        low_threshold=50,
        high_threshold=150,
        blur_kernel_size=7,
        min_contour_area=10.0,
        dilation_iterations=0,
        band_width=15,
        tangent_window=5,
        clahe_clip_limit=3.0,
        clahe_tile_grid_size=8,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        dilation_kernel_size=3,
    )

    # then
    assert OUTPUT_PREDICTIONS_KEY in result
    assert OUTPUT_EDGE_VIS_KEY in result
    assert isinstance(result[OUTPUT_PREDICTIONS_KEY], sv.Detections)
    assert isinstance(result[OUTPUT_EDGE_VIS_KEY], WorkflowImageData)


def test_edge_refinement_block_with_empty_detections(
    dogs_image: np.ndarray,
) -> None:
    # given
    block = EdgeRefinementBlockV1()
    detections = sv.Detections(
        xyxy=np.array([], dtype=np.float32).reshape(0, 4),
    )

    # when
    result = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=dogs_image,
        ),
        segmentation=detections,
        pixel_tolerance=5,
        low_threshold=50,
        high_threshold=150,
        blur_kernel_size=7,
        min_contour_area=10.0,
        dilation_iterations=0,
        band_width=15,
        tangent_window=5,
        clahe_clip_limit=3.0,
        clahe_tile_grid_size=8,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        dilation_kernel_size=3,
    )

    # then
    assert OUTPUT_PREDICTIONS_KEY in result
    assert OUTPUT_EDGE_VIS_KEY in result
    assert len(result[OUTPUT_PREDICTIONS_KEY]) == 0


# ---------------------------------------------------------------------------
# Tests for EdgeRefinementBlockV1 with masks
# ---------------------------------------------------------------------------


def test_edge_refinement_block_with_simple_mask(dogs_image: np.ndarray) -> None:
    # given
    block = EdgeRefinementBlockV1()

    # create a simple circular mask
    mask = np.zeros(dogs_image.shape[:2], dtype=bool)
    H, W = mask.shape
    center_y, center_x = H // 2, W // 2
    radius = 50
    y, x = np.ogrid[:H, :W]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    mask[mask_circle] = True

    detections = sv.Detections(
        xyxy=np.array(
            [[center_x - radius, center_y - radius, center_x + radius, center_y + radius]],
            dtype=np.float32,
        ),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    # when
    result = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=dogs_image,
        ),
        segmentation=detections,
        pixel_tolerance=5,
        low_threshold=50,
        high_threshold=150,
        blur_kernel_size=7,
        min_contour_area=10.0,
        dilation_iterations=0,
        band_width=15,
        tangent_window=5,
        clahe_clip_limit=3.0,
        clahe_tile_grid_size=8,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        dilation_kernel_size=3,
    )

    # then
    assert OUTPUT_PREDICTIONS_KEY in result
    assert OUTPUT_EDGE_VIS_KEY in result
    refined_detections = result[OUTPUT_PREDICTIONS_KEY]
    edge_vis = result[OUTPUT_EDGE_VIS_KEY]

    assert isinstance(refined_detections, sv.Detections)
    assert len(refined_detections) == 1
    assert refined_detections.mask is not None
    assert refined_detections.mask[0].shape == mask.shape

    assert isinstance(edge_vis, WorkflowImageData)
    assert edge_vis.numpy_image.shape == (dogs_image.shape[0], dogs_image.shape[1], 3)


def test_edge_refinement_block_preserves_detection_metadata(
    dogs_image: np.ndarray,
) -> None:
    # given
    block = EdgeRefinementBlockV1()

    mask = np.zeros(dogs_image.shape[:2], dtype=bool)
    H, W = mask.shape
    mask[50:150, 50:150] = True

    confidence = np.array([0.95])
    class_id = np.array([1])

    detections = sv.Detections(
        xyxy=np.array([[50.0, 50.0, 150.0, 150.0]], dtype=np.float32),
        mask=np.array([mask]),
        confidence=confidence,
        class_id=class_id,
    )

    # when
    result = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=dogs_image,
        ),
        segmentation=detections,
        pixel_tolerance=5,
        low_threshold=50,
        high_threshold=150,
        blur_kernel_size=7,
        min_contour_area=10.0,
        dilation_iterations=0,
        band_width=15,
        tangent_window=5,
        clahe_clip_limit=3.0,
        clahe_tile_grid_size=8,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        dilation_kernel_size=3,
    )

    # then
    refined_detections = result[OUTPUT_PREDICTIONS_KEY]
    assert len(refined_detections) == 1
    assert np.allclose(refined_detections.confidence, confidence)
    assert np.array_equal(refined_detections.class_id, class_id)


def test_edge_refinement_block_with_dilation(dogs_image: np.ndarray) -> None:
    # given
    block = EdgeRefinementBlockV1()

    mask = np.zeros(dogs_image.shape[:2], dtype=bool)
    mask[100:200, 100:200] = True

    detections = sv.Detections(
        xyxy=np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32),
        mask=np.array([mask]),
        confidence=np.array([0.8]),
        class_id=np.array([0]),
    )

    # when — test with dilation enabled
    result = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=dogs_image,
        ),
        segmentation=detections,
        pixel_tolerance=5,
        low_threshold=50,
        high_threshold=150,
        blur_kernel_size=7,
        min_contour_area=10.0,
        dilation_iterations=2,  # Enable dilation
        band_width=15,
        tangent_window=5,
        clahe_clip_limit=3.0,
        clahe_tile_grid_size=8,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        dilation_kernel_size=5,
    )

    # then
    assert OUTPUT_PREDICTIONS_KEY in result
    assert OUTPUT_EDGE_VIS_KEY in result
    assert result[OUTPUT_PREDICTIONS_KEY].mask is not None


# ---------------------------------------------------------------------------
# Tests for EdgeRefinementBlockV1 with multiple masks
# ---------------------------------------------------------------------------


def test_edge_refinement_block_with_multiple_masks(dogs_image: np.ndarray) -> None:
    # given
    block = EdgeRefinementBlockV1()

    # create two rectangular masks
    mask1 = np.zeros(dogs_image.shape[:2], dtype=bool)
    mask1[50:150, 50:150] = True

    mask2 = np.zeros(dogs_image.shape[:2], dtype=bool)
    mask2[200:300, 100:200] = True

    detections = sv.Detections(
        xyxy=np.array(
            [
                [50, 50, 150, 150],
                [100, 200, 200, 300],
            ],
            dtype=np.float32,
        ),
        mask=np.array([mask1, mask2]),
        confidence=np.array([0.9, 0.85]),
        class_id=np.array([0, 1]),
    )

    # when
    result = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=dogs_image,
        ),
        segmentation=detections,
        pixel_tolerance=5,
        low_threshold=50,
        high_threshold=150,
        blur_kernel_size=7,
        min_contour_area=10.0,
        dilation_iterations=0,
        band_width=15,
        tangent_window=5,
        clahe_clip_limit=3.0,
        clahe_tile_grid_size=8,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        dilation_kernel_size=3,
    )

    # then
    refined_detections = result[OUTPUT_PREDICTIONS_KEY]
    assert len(refined_detections) == 2
    assert refined_detections.mask is not None
    assert len(refined_detections.mask) == 2
    assert np.array_equal(refined_detections.confidence, np.array([0.9, 0.85]))
    assert np.array_equal(refined_detections.class_id, np.array([0, 1]))
