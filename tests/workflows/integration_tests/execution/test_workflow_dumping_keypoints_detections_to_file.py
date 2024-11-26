import json
import os
from glob import glob

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_KEYPOINTS_DETECTION_AND_FILE_SINK = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "target_dir",
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_keypoint_detection_model@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-pose-640",
        },
        {
            "type": "roboflow_core/expression@v1",
            "name": "json_formatter",
            "data": {"predictions": "$steps.model.predictions"},
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {
                    "type": "DynamicCaseResult",
                    "parameter_name": "predictions",
                    "operations": [
                        {"type": "DetectionsToDictionary"},
                        {"type": "ConvertDictionaryToJSON"},
                    ],
                },
            },
            "data_operations": {},
        },
        {
            "type": "roboflow_core/local_file_sink@v1",
            "name": "local_file_sink",
            "content": "$steps.json_formatter.output",
            "output_mode": "separate_files",
            "target_directory": "$inputs.target_dir",
            "file_name_prefix": "my_name_123_",
            "file_type": "json",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "error_status",
            "coordinates_system": "own",
            "selector": "$steps.local_file_sink.error_status",
        }
    ],
}


def test_workflow_with_data_aggregation(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    empty_directory: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_KEYPOINTS_DETECTION_AND_FILE_SINK,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, crowd_image],
            "target_dir": empty_directory,
        }
    )

    # then
    assert results[0]["error_status"] is False, "Expected no errors"
    persisted_files = glob(os.path.join(empty_directory, "*.json"))
    assert len(persisted_files) == 2, "Expected 2 dumped predictions"
    decoded_prediction_1 = _load_json_file(path=persisted_files[0])
    decoded_prediction_2 = _load_json_file(path=persisted_files[1])
    assert set(decoded_prediction_1.keys()) == {
        "image",
        "predictions",
    }, "Expected serialised payload to have correct keys"
    assert set(decoded_prediction_2.keys()) == {
        "image",
        "predictions",
    }, "Expected serialised payload to have correct keys"
    assert len(decoded_prediction_1["predictions"]) == 8, "Expected 8 objects detected"
    assert len(decoded_prediction_2["predictions"]) == 8, "Expected 8 objects detected"
    for detection in decoded_prediction_1["predictions"]:
        assert isinstance(detection["keypoints"], list), "Keypoints list is expected"
    for detection in decoded_prediction_2["predictions"]:
        assert isinstance(detection["keypoints"], list), "Keypoints list is expected"


def _load_json_file(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
