import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.execution_engine.core import ExecutionEngine

DETECTION_PLUS_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "dog-breed-xpaq6/1",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.breds_classification.predictions",
        }
    ],
}


@pytest.mark.asyncio
async def test_detection_plus_classification_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTION_PLUS_CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert set(result.keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert [result["predictions"][0]["top"], result["predictions"][1]["top"]] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"
