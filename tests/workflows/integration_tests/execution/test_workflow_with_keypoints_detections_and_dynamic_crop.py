import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_keypoint_detection_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-pose-640",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "dynamic_crop",
            "images": "$inputs.image",
            "predictions": "$steps.model.predictions",
        },
        {
            "type": "roboflow_core/keypoint_visualization@v1",
            "name": "model_keypoint_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions",
        },
        {
            "type": "roboflow_core/keypoint_visualization@v1",
            "name": "dynamic_crops_keypoint_visualization",
            "image": "$steps.dynamic_crop.crops",
            "predictions": "$steps.dynamic_crop.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_keypoint_visualization",
            "coordinates_system": "own",
            "selector": "$steps.model_keypoint_visualization.image",
        },
        {
            "type": "JsonField",
            "name": "dynamic_crops_keypoint_visualization",
            "coordinates_system": "own",
            "selector": "$steps.dynamic_crops_keypoint_visualization.image",
        },
        {
            "type": "JsonField",
            "name": "dynamic_crop_predictions",
            "selector": "$steps.dynamic_crop.predictions",
        },
    ],
}


def test_workflow_with_keypoints_and_dynamic_crop(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One set of images provided, so one output expected"
    assert set(result[0].keys()) == {
        "model_keypoint_visualization",
        "dynamic_crops_keypoint_visualization",
        "dynamic_crop_predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["dynamic_crops_keypoint_visualization"]) == 8
    ), "Expected 8 crops"
    assert (
        result[0]["model_keypoint_visualization"].numpy_image.shape == crowd_image.shape
    ), "Expected visualization of model predictions to be of the same shape as original image"
    assert (
        type(result[0]["dynamic_crop_predictions"]) == list
    ), "Expected dynamic crops predictions to be a list"
    assert (
        len(result[0]["dynamic_crop_predictions"]) == 8
    ), "Expected 8-elements list containing sv.Detections for each crop"
