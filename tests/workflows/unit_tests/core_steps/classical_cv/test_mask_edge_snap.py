import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.mask_edge_snap.v1 import (
    MaskEdgeSnapBlockV1,
    MaskEdgeSnapManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_mask_edge_snap_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    data = {
        "type": "roboflow_core/mask_edge_snap@v1",
        "name": "mask_edge_snap",
        images_field_alias: "$inputs.image",
        "segmentation": "$steps.segmentation_model.predictions",
    }

    result = MaskEdgeSnapManifest.model_validate(data)

    assert result == MaskEdgeSnapManifest(
        type="roboflow_core/mask_edge_snap@v1",
        name="mask_edge_snap",
        image="$inputs.image",
        segmentation="$steps.segmentation_model.predictions",
    )


def test_mask_edge_snap_validation_when_invalid_image_is_given() -> None:
    data = {
        "type": "roboflow_core/mask_edge_snap@v1",
        "name": "mask_edge_snap",
        "image": "invalid",
        "segmentation": "$steps.segmentation_model.predictions",
    }

    with pytest.raises(ValidationError):
        _ = MaskEdgeSnapManifest.model_validate(data)


def test_mask_edge_snap_validation_when_invalid_segmentation_is_given() -> None:
    data = {
        "type": "roboflow_core/mask_edge_snap@v1",
        "name": "mask_edge_snap",
        "image": "$inputs.image",
        "segmentation": "invalid",
    }

    with pytest.raises(ValidationError):
        _ = MaskEdgeSnapManifest.model_validate(data)


def test_mask_edge_snap_block_with_empty_segmentation(dogs_image: np.ndarray) -> None:
    block = MaskEdgeSnapBlockV1()

    # Empty segmentation - should pass through unchanged
    empty_segmentation = sv.Detections.empty()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        segmentation=empty_segmentation,
        pixel_tolerance=15,
        sigma=1.0,
        min_contour_area=50.0,
        dilation_iterations=2,
        boundary_band_width=15,
        adaptive_window_size=41,
    )

    assert output is not None
    assert "refined_segmentation" in output
    assert "edges" in output
    assert len(output["refined_segmentation"]) == 0


def test_mask_edge_snap_block_with_single_mask(dogs_image: np.ndarray) -> None:
    block = MaskEdgeSnapBlockV1()

    # Create a single mask (e.g., from instance segmentation model)
    h, w = dogs_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        segmentation=segmentation,
        pixel_tolerance=15,
        sigma=1.0,
        min_contour_area=50.0,
        dilation_iterations=2,
        boundary_band_width=15,
        adaptive_window_size=41,
    )

    assert output is not None
    assert "refined_segmentation" in output
    assert "edges" in output
    assert len(output["refined_segmentation"]) == 1
    assert output["refined_segmentation"].mask is not None
    assert output["refined_segmentation"].mask[0].shape == (h, w)


def test_mask_edge_snap_block_with_multiple_masks(dogs_image: np.ndarray) -> None:
    block = MaskEdgeSnapBlockV1()

    # Create multiple masks
    h, w = dogs_image.shape[:2]
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[50:150, 100:200] = True

    mask2 = np.zeros((h, w), dtype=bool)
    mask2[200:300, 300:400] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0], [300.0, 200.0, 400.0, 300.0]]),
        mask=np.array([mask1, mask2]),
        confidence=np.array([0.9, 0.85]),
        class_id=np.array([0, 1]),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        segmentation=segmentation,
        pixel_tolerance=15,
        sigma=1.0,
        min_contour_area=50.0,
        dilation_iterations=2,
        boundary_band_width=15,
        adaptive_window_size=41,
    )

    assert output is not None
    assert "refined_segmentation" in output
    assert "edges" in output
    assert len(output["refined_segmentation"]) == 2
    assert output["refined_segmentation"].mask.shape[0] == 2


def test_mask_edge_snap_block_with_grayscale_image() -> None:
    block = MaskEdgeSnapBlockV1()

    # Create grayscale image
    gray_image = np.random.randint(100, 200, (256, 256), dtype=np.uint8)

    # Create a mask
    h, w = gray_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=gray_image,
        ),
        segmentation=segmentation,
        pixel_tolerance=15,
        sigma=1.0,
        min_contour_area=50.0,
        dilation_iterations=2,
        boundary_band_width=15,
        adaptive_window_size=41,
    )

    assert output is not None
    assert "refined_segmentation" in output
    assert "edges" in output
    assert len(output["refined_segmentation"]) == 1


def test_mask_edge_snap_block_with_different_parameters(
    dogs_image: np.ndarray,
) -> None:
    block = MaskEdgeSnapBlockV1()

    h, w = dogs_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    # Test with permissive parameters (low sigma, small tolerance)
    output1 = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        segmentation=segmentation,
        pixel_tolerance=5,
        sigma=0.3,
        min_contour_area=10.0,
        dilation_iterations=1,
        boundary_band_width=10,
        adaptive_window_size=21,
    )

    # Test with strict parameters (high sigma, large tolerance)
    output2 = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        segmentation=segmentation,
        pixel_tolerance=50,
        sigma=2.0,
        min_contour_area=200.0,
        dilation_iterations=5,
        boundary_band_width=50,
        adaptive_window_size=81,
    )

    # Both should return valid results
    assert output1 is not None
    assert output2 is not None
    assert "refined_segmentation" in output1
    assert "refined_segmentation" in output2
    assert len(output1["refined_segmentation"]) == 1
    assert len(output2["refined_segmentation"]) == 1


def test_mask_edge_snap_block_preserves_detection_metadata(
    dogs_image: np.ndarray,
) -> None:
    block = MaskEdgeSnapBlockV1()

    h, w = dogs_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    xyxy = np.array([[100.0, 50.0, 200.0, 150.0]])
    confidence = np.array([0.95])
    class_id = np.array([5])
    tracker_id = np.array([123])

    segmentation = sv.Detections(
        xyxy=xyxy,
        mask=np.array([mask]),
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        segmentation=segmentation,
        pixel_tolerance=15,
        sigma=1.0,
        min_contour_area=50.0,
        dilation_iterations=2,
        boundary_band_width=15,
        adaptive_window_size=41,
    )

    refined = output["refined_segmentation"]
    assert np.allclose(refined.xyxy, xyxy)
    assert np.allclose(refined.confidence, confidence)
    assert np.array_equal(refined.class_id, class_id)
    assert np.array_equal(refined.tracker_id, tracker_id)


def test_mask_edge_snap_block_with_bgra_image() -> None:
    block = MaskEdgeSnapBlockV1()

    # Create BGRA image
    bgra_image = np.random.randint(100, 200, (256, 256, 4), dtype=np.uint8)

    h, w = bgra_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=bgra_image,
        ),
        segmentation=segmentation,
        pixel_tolerance=15,
        sigma=1.0,
        min_contour_area=50.0,
        dilation_iterations=2,
        boundary_band_width=15,
        adaptive_window_size=41,
    )

    assert output is not None
    assert "refined_segmentation" in output
    assert len(output["refined_segmentation"]) == 1
