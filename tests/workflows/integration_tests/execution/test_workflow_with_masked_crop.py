import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

MASKED_CROP_LEGACY_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-seg-640",
        },
        {
            "type": "WorkflowParameter",
            "name": "confidence",
            "default_value": 0.4,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_instance_segmentation_model@v1",
            "name": "segmentation",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.segmentation.predictions",
            "mask_opacity": 1.0,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.cropping.crops",
        },
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.segmentation.predictions",
        },
    ],
}


def test_legacy_workflow_with_masked_crop(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MASKED_CROP_LEGACY_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "crops",
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["crops"]) == 2, "Expected 2 crops for two dogs detected"
    crop_image = result[0]["crops"][0].numpy_image
    (x_min, y_min, x_max, y_max) = (
        result[0]["predictions"].xyxy[0].round().astype(dtype=int)
    )
    crop_mask = result[0]["predictions"].mask[0][y_min:y_max, x_min:x_max]
    pixels_outside_mask = np.where(
        np.stack([crop_mask] * 3, axis=-1) == 0,
        crop_image,
        np.zeros_like(crop_image),
    )
    pixels_sum = pixels_outside_mask.sum()
    assert pixels_sum == 0, "Expected everything black outside mask"


MASKED_CROP_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-seg-640",
        },
        {
            "type": "WorkflowParameter",
            "name": "confidence",
            "default_value": 0.4,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
            "name": "segmentation",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.segmentation.predictions",
            "mask_opacity": 1.0,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "crops",
            "selector": "$steps.cropping.crops",
        },
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.segmentation.predictions",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with data transformations",
    use_case_title="Instance Segmentation results with background subtracted",
    use_case_description="""
This example showcases how to extract all instances detected by instance segmentation model
as separate crops without background.
    """,
    workflow_definition=MASKED_CROP_WORKFLOW,
    workflow_name_in_app="segmentation-plus-masked-crop",
)
def test_workflow_with_masked_crop(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MASKED_CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "crops",
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["crops"]) == 2, "Expected 2 crops for two dogs detected"
    crop_image = result[0]["crops"][0].numpy_image
    (x_min, y_min, x_max, y_max) = (
        result[0]["predictions"].xyxy[0].round().astype(dtype=int)
    )
    crop_mask = result[0]["predictions"].mask[0][y_min:y_max, x_min:x_max]
    pixels_outside_mask = np.where(
        np.stack([crop_mask] * 3, axis=-1) == 0,
        crop_image,
        np.zeros_like(crop_image),
    )
    pixels_sum = pixels_outside_mask.sum()
    assert pixels_sum == 0, "Expected everything black outside mask"


def test_workflow_with_masked_crop_when_nothing_gets_predicted(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MASKED_CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": dogs_image, "confidence": 0.99}
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "crops",
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["crops"]) == 0, "Expected 0 crops detected"
