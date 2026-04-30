import copy

import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_CLASSICAL_CV_PREPROCESSING = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/contrast_enhancement@v1",
            "name": "enhance_contrast",
            "image": "$inputs.image",
            "clip_limit": 0,
            "contrast_multiplier": 1.5,
            "normalize_brightness": True,
        },
        {
            "type": "roboflow_core/morphological_transformation@v2",
            "name": "enhance_structures",
            "image": "$steps.enhance_contrast.image",
            "operation": "Opening then Closing",
            "kernel_size": 5,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "preprocessed_image",
            "selector": "$steps.enhance_structures.image",
        },
        {
            "type": "JsonField",
            "name": "contrast_enhanced",
            "selector": "$steps.enhance_contrast.image",
        },
    ],
}

WORKFLOW_WITH_EDGE_SNAP = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "InferenceParameter", "name": "segmentation"},
    ],
    "steps": [
        {
            "type": "roboflow_core/contrast_enhancement@v1",
            "name": "enhance_contrast",
            "image": "$inputs.image",
            "clip_limit": 0,
            "contrast_multiplier": 1.5,
            "normalize_brightness": True,
        },
        {
            "type": "roboflow_core/morphological_transformation@v2",
            "name": "enhance_structures",
            "image": "$steps.enhance_contrast.image",
            "operation": "Opening then Closing",
            "kernel_size": 5,
        },
        {
            "type": "roboflow_core/mask_edge_snap@v1",
            "name": "refine_edges",
            "image": "$steps.enhance_structures.image",
            "segmentation": "$inputs.segmentation",
            "pixel_tolerance": 15,
            "sigma": 1.0,
            "min_contour_area": 50,
            "dilation_iterations": 2,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "refined_segmentation",
            "selector": "$steps.refine_edges.refined_segmentation",
        },
        {
            "type": "JsonField",
            "name": "preprocessed_image",
            "selector": "$steps.enhance_structures.image",
        },
    ],
}


def test_classical_cv_preprocessing_pipeline_with_color_image(
    model_manager: ModelManager,
) -> None:
    """
    Test the complete preprocessing pipeline with a realistic low-contrast, noisy image.
    Verifies that all blocks work together correctly and produce valid outputs.
    """
    # given - Create a realistic low-contrast, noisy image
    base_image = np.ones((200, 200, 3), dtype=np.uint8) * 100
    # Add some noise
    np.random.seed(42)
    noise = np.random.randint(-20, 20, (200, 200, 3), dtype=np.int16)
    noisy_image = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLASSICAL_CV_PREPROCESSING,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": noisy_image},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    assert "preprocessed_image" in result_dict
    assert "contrast_enhanced" in result_dict

    # Extract numpy arrays from WorkflowImageData objects
    preprocessed = result_dict["preprocessed_image"].numpy_image
    contrast_enhanced = result_dict["contrast_enhanced"].numpy_image

    # Verify output dimensions match input
    assert preprocessed.shape == noisy_image.shape
    assert contrast_enhanced.shape == noisy_image.shape

    # Verify data types
    assert preprocessed.dtype == np.uint8
    assert contrast_enhanced.dtype == np.uint8

    # Verify that preprocessing actually changed the image
    assert not np.array_equal(preprocessed, noisy_image)
    assert not np.array_equal(contrast_enhanced, noisy_image)


def test_classical_cv_preprocessing_pipeline_with_grayscale_image(
    model_manager: ModelManager,
) -> None:
    """
    Test the preprocessing pipeline with a grayscale image.
    Verifies that blocks handle grayscale inputs correctly.
    """
    # given - Create a grayscale low-contrast image
    grayscale_image = np.ones((150, 150), dtype=np.uint8) * 80
    np.random.seed(42)
    noise = np.random.randint(-15, 15, (150, 150), dtype=np.int16)
    noisy_grayscale = np.clip(grayscale_image.astype(np.int16) + noise, 0, 255).astype(
        np.uint8
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLASSICAL_CV_PREPROCESSING,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": noisy_grayscale},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    preprocessed = result_dict["preprocessed_image"].numpy_image

    assert preprocessed.shape[:2] == noisy_grayscale.shape
    assert preprocessed.dtype == np.uint8
    assert not np.array_equal(preprocessed, noisy_grayscale)


def test_classical_cv_preprocessing_pipeline_with_high_contrast_image(
    model_manager: ModelManager,
) -> None:
    """
    Test the preprocessing pipeline with a high-contrast image.
    Verifies that the pipeline doesn't over-process already good images.
    """
    # given - Create a high-contrast image (bimodal histogram)
    high_contrast_image = np.ones((200, 200, 3), dtype=np.uint8) * 50
    high_contrast_image[100:, :] = 200

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLASSICAL_CV_PREPROCESSING,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": high_contrast_image},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    preprocessed = result_dict["preprocessed_image"].numpy_image

    # Output should still match input dimensions
    assert preprocessed.shape == high_contrast_image.shape
    assert preprocessed.dtype == np.uint8


def test_classical_cv_preprocessing_pipeline_with_bgra_image(
    model_manager: ModelManager,
) -> None:
    """
    Test the preprocessing pipeline with a BGRA image.
    Verifies that alpha channels are preserved throughout the pipeline.
    """
    # given - Create a BGRA image with alpha channel
    bgra_image = np.ones((100, 100, 4), dtype=np.uint8) * 100
    bgra_image[:, :, 3] = 255  # Alpha channel at full opacity

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLASSICAL_CV_PREPROCESSING,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": bgra_image},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    preprocessed = result_dict["preprocessed_image"].numpy_image

    # Verify BGRA shape is preserved
    assert preprocessed.shape == bgra_image.shape
    assert preprocessed.dtype == np.uint8


@pytest.mark.parametrize(
    "operation",
    [
        "Opening",
        "Closing",
        "Opening then Closing",
        "Gradient",
        "Top Hat",
        "Black Hat",
    ],
)
def test_classical_cv_preprocessing_pipeline_with_different_operations(
    model_manager: ModelManager, operation: str
) -> None:
    """
    Test the preprocessing pipeline with different morphological operations.
    Verifies that the pipeline is flexible with operation selection.
    """
    # given
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 120

    workflow_def = copy.deepcopy(WORKFLOW_WITH_CLASSICAL_CV_PREPROCESSING)
    # Update the morphological operation
    workflow_def["steps"][1]["operation"] = operation

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_def,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": test_image},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    preprocessed = result_dict["preprocessed_image"].numpy_image
    assert preprocessed.shape == test_image.shape
    assert preprocessed.dtype == np.uint8


def test_classical_cv_preprocessing_pipeline_all_intermediate_outputs(
    model_manager: ModelManager,
) -> None:
    """
    Test that all intermediate outputs are accessible in the result.
    Verifies the pipeline produces outputs at each stage.
    """
    # given
    test_image = np.ones((150, 150, 3), dtype=np.uint8) * 90
    np.random.seed(42)
    noise = np.random.randint(-10, 10, (150, 150, 3), dtype=np.int16)
    noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLASSICAL_CV_PREPROCESSING,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": noisy_image},
    )

    # then - Verify all outputs are present
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    assert "preprocessed_image" in result_dict
    assert "contrast_enhanced" in result_dict

    # Verify each output represents a processing stage
    original = noisy_image
    contrast_enhanced = result_dict["contrast_enhanced"].numpy_image
    preprocessed = result_dict["preprocessed_image"].numpy_image

    # Each stage should produce different results
    assert not np.array_equal(original, contrast_enhanced)
    assert not np.array_equal(contrast_enhanced, preprocessed)

    # But all should have same shape and dtype
    assert contrast_enhanced.shape == original.shape
    assert preprocessed.shape == original.shape


def test_edge_snap_cv_pipeline_with_segmentation(
    model_manager: ModelManager,
) -> None:
    """
    Test the complete edge snap pipeline with preprocessing and mask refinement.
    Verifies that the preprocessing and edge refinement work together correctly.
    """
    # given - Create a test image with a synthetic object
    test_image = np.ones((200, 200, 3), dtype=np.uint8) * 100
    # Add a rectangle-like object
    test_image[50:150, 50:150] = 150
    # Add some noise
    np.random.seed(42)
    noise = np.random.randint(-20, 20, (200, 200, 3), dtype=np.int16)
    noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Create synthetic segmentation with a mask
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 1  # Simple rectangular mask

    detections = sv.Detections(
        xyxy=np.array([[50, 50, 150, 150]]),
        confidence=np.array([0.95]),
        class_id=np.array([0]),
        mask=mask[np.newaxis, :, :],
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_EDGE_SNAP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": noisy_image, "segmentation": detections},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    assert "refined_segmentation" in result_dict
    assert "preprocessed_image" in result_dict

    refined_seg = result_dict["refined_segmentation"]
    preprocessed = result_dict["preprocessed_image"].numpy_image

    # Verify outputs
    assert refined_seg is not None
    assert preprocessed.shape == noisy_image.shape
    assert preprocessed.dtype == np.uint8

    # Verify that refined_segmentation has masks
    assert hasattr(refined_seg, "mask")
    assert refined_seg.mask is not None or len(refined_seg.mask) == 0


def test_edge_snap_cv_pipeline_with_empty_segmentation(
    model_manager: ModelManager,
) -> None:
    """
    Test the edge snap pipeline with empty segmentation.
    Verifies that the pipeline handles empty predictions gracefully.
    """
    # given - Create a simple test image
    test_image = np.ones((150, 150, 3), dtype=np.uint8) * 120

    # Create empty detections
    detections = sv.Detections.empty()

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_EDGE_SNAP,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": test_image, "segmentation": detections},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    result_dict = result[0]
    assert "refined_segmentation" in result_dict
    preprocessed = result_dict["preprocessed_image"].numpy_image

    # Verify preprocessing still works
    assert preprocessed.shape == test_image.shape
    assert preprocessed.dtype == np.uint8
