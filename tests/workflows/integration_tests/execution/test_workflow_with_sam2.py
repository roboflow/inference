import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

SIMPLE_SAM_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "mask_threshold", "default_value": 0.0},
        {"type": "WorkflowParameter", "name": "version", "default_value": "hiera_tiny"},
    ],
    "steps": [
        {
            "type": "roboflow_core/segment_anything@v1",
            "name": "segment_anything",
            "images": "$inputs.image",
            "threshold": "$inputs.mask_threshold",
            "version": "$inputs.version",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.segment_anything.predictions",
        },
    ],
}


def test_sam2_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager, dogs_image: np.ndarray
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SIMPLE_SAM_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "mask_threshold": 0.0,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input images"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["predictions"].mask is not None, "Expected mask to be delivered"
    assert result[0]["predictions"].mask.shape == (1, 427, 640)
    assert (
        result[0]["predictions"].data["prediction_type"][0] == "instance-segmentation"
    )


def test_sam2_workflow_when_minimal_valid_input_provided_but_filtering_discard_mask(
    model_manager: ModelManager, dogs_image: np.ndarray
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SIMPLE_SAM_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "mask_threshold": 0.9,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input images"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["predictions"]) == 0, "Expected no predictions"


GROUNDED_SAM_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "mask_threshold", "default_value": 0.0},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detection",
            "model_id": "yolov8n-640",
            "images": "$inputs.image",
            "class_filter": ["dog"],
        },
        {
            "type": "roboflow_core/segment_anything@v1",
            "name": "segment_anything",
            "images": "$inputs.image",
            "boxes": "$steps.detection.predictions",
            "threshold": "$inputs.mask_threshold",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "sam_predictions",
            "selector": "$steps.segment_anything.predictions",
        },
    ],
}


def test_grounded_sam2_workflow(
    model_manager: ModelManager, dogs_image: np.ndarray
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=GROUNDED_SAM_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "mask_threshold": 0.5,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input images"
    assert set(result[0].keys()) == {
        "sam_predictions",
    }, "Expected all declared outputs to be delivered"
    assert np.allclose(
        result[0]["sam_predictions"].xyxy,
        np.array([[321, 223, 582, 405], [226, 73, 382, 381]]),
        atol=1e-1,
    ), "Expected bboxes to be the same as measured while test creation"
    assert np.allclose(
        result[0]["sam_predictions"].confidence, np.array([0.9602, 0.93673]), atol=1e-4
    ), "Expected confidence to be the same as measured while test creation"
    assert result[0]["sam_predictions"]["class_name"].tolist() == [
        "dog",
        "dog",
    ], "Expected class names to be correct"
