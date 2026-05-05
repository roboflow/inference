import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine


def _build_mask_edge_snap_workflow() -> dict:
    """Build a workflow with Mask Edge Snap block for testing."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {"type": "InferenceParameter", "name": "segmentation"},
            {
                "type": "WorkflowParameter",
                "name": "pixel_tolerance",
                "default_value": 15,
            },
            {"type": "WorkflowParameter", "name": "sigma", "default_value": 1.0},
            {
                "type": "WorkflowParameter",
                "name": "min_contour_area",
                "default_value": 50.0,
            },
            {
                "type": "WorkflowParameter",
                "name": "dilation_iterations",
                "default_value": 2,
            },
            {
                "type": "WorkflowParameter",
                "name": "boundary_band_width",
                "default_value": 15,
            },
            {
                "type": "WorkflowParameter",
                "name": "adaptive_window_size",
                "default_value": 41,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/mask_edge_snap@v1",
                "name": "mask_edge_snap",
                "image": "$inputs.image",
                "segmentation": "$inputs.segmentation",
                "pixel_tolerance": "$inputs.pixel_tolerance",
                "sigma": "$inputs.sigma",
                "min_contour_area": "$inputs.min_contour_area",
                "dilation_iterations": "$inputs.dilation_iterations",
                "boundary_band_width": "$inputs.boundary_band_width",
                "adaptive_window_size": "$inputs.adaptive_window_size",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.mask_edge_snap.*",
            }
        ],
    }


@pytest.mark.slow
def test_mask_edge_snap_workflow_with_empty_segmentation(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test Mask Edge Snap workflow with empty segmentation."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_mask_edge_snap_workflow(),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # Create empty segmentation
    empty_segmentation = sv.Detections.empty()

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "segmentation": empty_segmentation,
            "pixel_tolerance": 15,
            "sigma": 1.0,
            "min_contour_area": 50.0,
            "dilation_iterations": 2,
            "boundary_band_width": 15,
            "adaptive_window_size": 41,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    output = result[0]["result"]
    assert "refined_segmentation" in output
    assert "edges" in output
    assert len(output["refined_segmentation"]) == 0


@pytest.mark.slow
def test_mask_edge_snap_workflow_with_single_mask(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test Mask Edge Snap workflow with a single mask."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_mask_edge_snap_workflow(),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # Create a single mask
    h, w = dogs_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "segmentation": segmentation,
            "pixel_tolerance": 15,
            "sigma": 1.0,
            "min_contour_area": 50.0,
            "dilation_iterations": 2,
            "boundary_band_width": 15,
            "adaptive_window_size": 41,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    output = result[0]["result"]
    assert "refined_segmentation" in output
    assert "edges" in output
    refined = output["refined_segmentation"]
    assert len(refined) == 1
    assert refined.mask is not None
    assert refined.mask[0].shape == (h, w)


@pytest.mark.slow
def test_mask_edge_snap_workflow_with_multiple_masks(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test Mask Edge Snap workflow with multiple masks."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_mask_edge_snap_workflow(),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

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

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "segmentation": segmentation,
            "pixel_tolerance": 15,
            "sigma": 1.0,
            "min_contour_area": 50.0,
            "dilation_iterations": 2,
            "boundary_band_width": 15,
            "adaptive_window_size": 41,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    output = result[0]["result"]
    assert "refined_segmentation" in output
    assert "edges" in output
    refined = output["refined_segmentation"]
    assert len(refined) == 2
    assert refined.mask.shape[0] == 2


@pytest.mark.slow
def test_mask_edge_snap_workflow_with_permissive_parameters(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test Mask Edge Snap workflow with permissive parameters."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_mask_edge_snap_workflow(),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # Create a mask
    h, w = dogs_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    # when - using permissive parameters
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "segmentation": segmentation,
            "pixel_tolerance": 5,
            "sigma": 0.3,
            "min_contour_area": 10.0,
            "dilation_iterations": 1,
            "boundary_band_width": 10,
            "adaptive_window_size": 21,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1
    output = result[0]["result"]
    assert "refined_segmentation" in output
    assert "edges" in output


@pytest.mark.slow
def test_mask_edge_snap_workflow_with_strict_parameters(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test Mask Edge Snap workflow with strict parameters."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_mask_edge_snap_workflow(),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # Create a mask
    h, w = dogs_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    # when - using strict parameters
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "segmentation": segmentation,
            "pixel_tolerance": 50,
            "sigma": 2.0,
            "min_contour_area": 200.0,
            "dilation_iterations": 5,
            "boundary_band_width": 50,
            "adaptive_window_size": 81,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1
    output = result[0]["result"]
    assert "refined_segmentation" in output
    assert "edges" in output


@pytest.mark.slow
def test_mask_edge_snap_workflow_with_different_mask_sizes(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test Mask Edge Snap workflow with masks of different sizes."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_mask_edge_snap_workflow(),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # Create masks with different sizes
    h, w = dogs_image.shape[:2]

    # Small mask
    small_mask = np.zeros((h, w), dtype=bool)
    small_mask[100:120, 150:170] = True

    # Medium mask
    medium_mask = np.zeros((h, w), dtype=bool)
    medium_mask[50:200, 100:300] = True

    segmentation = sv.Detections(
        xyxy=np.array([[150.0, 100.0, 170.0, 120.0], [100.0, 50.0, 300.0, 200.0]]),
        mask=np.array([small_mask, medium_mask]),
        confidence=np.array([0.9, 0.85]),
        class_id=np.array([0, 1]),
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "segmentation": segmentation,
            "pixel_tolerance": 15,
            "sigma": 1.0,
            "min_contour_area": 50.0,
            "dilation_iterations": 2,
            "boundary_band_width": 15,
            "adaptive_window_size": 41,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    output = result[0]["result"]
    assert "refined_segmentation" in output
    assert "edges" in output
    refined = output["refined_segmentation"]
    assert len(refined) == 2, "Both masks should be refined"
    assert refined.mask.shape[0] == 2


def _build_mask_edge_snap_with_morphological_preprocessing_workflow() -> dict:
    """Build a workflow with morphological preprocessing before Mask Edge Snap."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {"type": "InferenceParameter", "name": "segmentation"},
        ],
        "steps": [
            {
                "type": "roboflow_core/morphological_transformation@v2",
                "name": "morphological_preprocessing",
                "image": "$inputs.image",
                "operation": "Opening then Closing",
                "kernel_size": 5,
            },
            {
                "type": "roboflow_core/mask_edge_snap@v1",
                "name": "mask_edge_snap",
                "image": "$steps.morphological_preprocessing.image",
                "segmentation": "$inputs.segmentation",
                "pixel_tolerance": 15,
                "sigma": 1.0,
                "min_contour_area": 50.0,
                "dilation_iterations": 2,
                "boundary_band_width": 15,
                "adaptive_window_size": 41,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.mask_edge_snap.*",
            }
        ],
    }


@pytest.mark.slow
def test_mask_edge_snap_workflow_with_morphological_preprocessing(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test Mask Edge Snap workflow with morphological preprocessing for noise reduction.

    This test demonstrates the preprocessing pipeline:
    Morphological Opening+Closing (remove noise) -> Mask Edge Snap (refine edges)
    This is a recommended approach for challenging imagery with debris or hot pixels.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_mask_edge_snap_with_morphological_preprocessing_workflow(),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # Create a mask
    h, w = dogs_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[50:150, 100:200] = True

    segmentation = sv.Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 150.0]]),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )

    # when - run workflow with morphological preprocessing
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "segmentation": segmentation,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    output = result[0]["result"]
    assert "refined_segmentation" in output
    assert "edges" in output
    refined = output["refined_segmentation"]
    assert len(refined) == 1
    assert refined.mask is not None
    assert refined.mask[0].shape == (h, w)

