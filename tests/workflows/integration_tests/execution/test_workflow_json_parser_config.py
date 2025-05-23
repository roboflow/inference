import numpy as np
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

JSON_PARSER_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {
            "type": "WorkflowParameter",
            "name": "config",
            "default_value": "{\"model_id\": \"yolov8n-640\"}",
        },
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/json_parser@v1",
            "name": "json_parser",
            "raw_json": "$inputs.config",
            "expected_fields": ["model_id"],
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "$steps.json_parser.model_id",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "json_parser",
            "selector": "$steps.json_parser.model_id",
        },
        {
            "type": "JsonField",
            "name": "model_predictions",
            "selector": "$steps.model.predictions",
        },
    ],
}


def test_workflow_with_json_parameter(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=JSON_PARSER_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "config": "{\"model_id\": \"yolov8n-640\"}",
        }
    )

    assert len(result) == 1
    assert set(result[0].keys()) == {"json_parser", "model_predictions"}
    assert result[0]["json_parser"] == "yolov8n-640"
    assert isinstance(result[0]["model_predictions"], sv.Detections)
