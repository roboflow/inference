import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_STITCHING_OCR_DETECTIONS_TOLERANCE = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "ocr-oy9a7/1",
        },
        {"type": "WorkflowParameter", "name": "tolerance", "default_value": 10},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "ocr_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/stitch_ocr_detections@v1",
            "name": "detections_stitch",
            "predictions": "$steps.ocr_detection.predictions",
            "stitching_algorithm": "tolerance",
            "reading_direction": "left_to_right",
            "tolerance": "$inputs.tolerance",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "ocr_text",
            "selector": "$steps.detections_stitch.ocr_text",
        },
    ],
}

WORKFLOW_STITCHING_OCR_DETECTIONS_OTSU = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "ocr-oy9a7/1",
        },
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
        {
            "type": "WorkflowParameter",
            "name": "threshold_multiplier",
            "default_value": 1.0,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "ocr_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/stitch_ocr_detections@v1",
            "name": "detections_stitch",
            "predictions": "$steps.ocr_detection.predictions",
            "stitching_algorithm": "otsu",
            "reading_direction": "left_to_right",
            "otsu_threshold_multiplier": "$inputs.threshold_multiplier",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "ocr_text",
            "selector": "$steps.detections_stitch.ocr_text",
        },
    ],
}

WORKFLOW_STITCHING_OCR_DETECTIONS_COLLIMATE = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "ocr-oy9a7/1",
        },
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
        {
            "type": "WorkflowParameter",
            "name": "collimate_tolerance",
            "default_value": 10,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "ocr_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/stitch_ocr_detections@v1",
            "name": "detections_stitch",
            "predictions": "$steps.ocr_detection.predictions",
            "stitching_algorithm": "collimate",
            "reading_direction": "left_to_right",
            "collimate_tolerance": "$inputs.collimate_tolerance",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "ocr_text",
            "selector": "$steps.detections_stitch.ocr_text",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows for OCR",
    use_case_title="Workflow with model detecting individual characters and text stitching (tolerance algorithm)",
    use_case_description="""
This workflow extracts and organizes text from an image using OCR with the tolerance-based stitching algorithm.
It detects individual characters or words and their positions, then groups nearby text into lines based on a
specified pixel `tolerance` for spacing and arranges them in reading order (`left-to-right`).

The tolerance algorithm is best for consistent font sizes and well-aligned horizontal/vertical text.
    """,
    workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS_TOLERANCE,
    workflow_name_in_app="ocr-detections-stitch-v2-tolerance",
)
def test_ocr_stitching_v2_tolerance_algorithm(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS_TOLERANCE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": multi_line_text_image,
            "tolerance": 20,
            "confidence": 0.6,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "ocr_text",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["ocr_text"] == "MAKE\nTHISDAY\nGREAT"


@add_to_workflows_gallery(
    category="Workflows for OCR",
    use_case_title="Workflow with model detecting individual characters and text stitching (Otsu algorithm)",
    use_case_description="""
This workflow extracts and organizes text from an image using OCR with the Otsu thresholding algorithm.
It detects individual characters and uses Otsu's method on normalized gap distances to automatically find
the optimal threshold separating character gaps from word gaps.

The Otsu algorithm is resolution-invariant and works well with variable font sizes and automatic word
boundary detection. It detects bimodal distributions to distinguish single words from multi-word text.
    """,
    workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS_OTSU,
    workflow_name_in_app="ocr-detections-stitch-v2-otsu",
)
def test_ocr_stitching_v2_otsu_algorithm(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS_OTSU,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": multi_line_text_image,
            "confidence": 0.6,
            "threshold_multiplier": 1.0,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "ocr_text",
    }, "Expected all declared outputs to be delivered"
    # Otsu may insert spaces between words if it detects bimodal distribution
    assert isinstance(result[0]["ocr_text"], str)
    assert len(result[0]["ocr_text"]) > 0


@add_to_workflows_gallery(
    category="Workflows for OCR",
    use_case_title="Workflow with model detecting individual characters and text stitching (collimate algorithm)",
    use_case_description="""
This workflow extracts and organizes text from an image using OCR with the collimate algorithm.
It detects individual characters and uses greedy parent-child traversal to follow text flow,
building lines through traversal rather than bucketing.

The collimate algorithm is best for skewed, curved, or non-axis-aligned text where traditional
bucket-based line grouping may fail.
    """,
    workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS_COLLIMATE,
    workflow_name_in_app="ocr-detections-stitch-v2-collimate",
)
def test_ocr_stitching_v2_collimate_algorithm(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_STITCHING_OCR_DETECTIONS_COLLIMATE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": multi_line_text_image,
            "confidence": 0.6,
            "collimate_tolerance": 15,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "ocr_text",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(result[0]["ocr_text"], str)
    assert len(result[0]["ocr_text"]) > 0


@pytest.mark.parametrize(
    "algorithm,workflow_definition",
    [
        ("tolerance", WORKFLOW_STITCHING_OCR_DETECTIONS_TOLERANCE),
        ("otsu", WORKFLOW_STITCHING_OCR_DETECTIONS_OTSU),
        ("collimate", WORKFLOW_STITCHING_OCR_DETECTIONS_COLLIMATE),
    ],
)
def test_ocr_stitching_v2_all_algorithms_produce_output(
    model_manager: ModelManager,
    multi_line_text_image: np.ndarray,
    roboflow_api_key: str,
    algorithm: str,
    workflow_definition: dict,
) -> None:
    """Test that all stitching algorithms produce valid output."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    runtime_params = {
        "image": multi_line_text_image,
        "confidence": 0.6,
    }
    if algorithm == "tolerance":
        runtime_params["tolerance"] = 20
    elif algorithm == "otsu":
        runtime_params["threshold_multiplier"] = 1.0
    elif algorithm == "collimate":
        runtime_params["collimate_tolerance"] = 15

    result = execution_engine.run(runtime_parameters=runtime_params)

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert "ocr_text" in result[0], "Expected ocr_text in output"
    assert isinstance(result[0]["ocr_text"], str), "Expected string output"
    # All algorithms should detect some text
    assert (
        len(result[0]["ocr_text"]) > 0
    ), f"Algorithm {algorithm} produced empty output"
