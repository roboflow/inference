import numpy as np
import pytest
import supervision as sv

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import (
    Detections as NativeDetections,
)
from tests.workflows.integration_tests.execution.tensor_input_utils import (
    numpy_image_as_tensor,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the object detection model block emits a
# native inference_models.Detections instead of sv.Detections. The sv-typed test below
# is skipped when the flag is on; the `*_tensor_native` parity test (skipped when the
# flag is off) asserts the same workflow result expressed as the native dataclass.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections output; model block emits native Detections under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

JSON_PARSER_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {
            "type": "WorkflowParameter",
            "name": "config",
            "default_value": '{"model_id": "yolov8n-640"}',
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
            "type": "roboflow_core/roboflow_object_detection_model@v3",
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


@_NUMPY_ONLY
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
            "config": '{"model_id": "yolov8n-640"}',
        }
    )

    assert len(result) == 1
    assert set(result[0].keys()) == {"json_parser", "model_predictions"}
    assert result[0]["json_parser"] == "yolov8n-640"
    assert isinstance(result[0]["model_predictions"], sv.Detections)


@_TENSOR_ONLY
def test_workflow_with_json_parameter_tensor_native(
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
            "config": '{"model_id": "yolov8n-640"}',
        }
    )

    assert len(result) == 1
    assert set(result[0].keys()) == {"json_parser", "model_predictions"}
    assert result[0]["json_parser"] == "yolov8n-640"
    # Under ENABLE_TENSOR_DATA_REPRESENTATION the model block emits a native
    # inference_models.Detections (torch-backed) rather than sv.Detections.
    predictions = result[0]["model_predictions"]
    assert isinstance(predictions, NativeDetections)
    # Same semantic result as the numpy path: two dogs detected (COCO class_id 16).
    assert len(predictions.xyxy) == 2
    assert predictions.class_id.tolist() == [16, 16]


@_TENSOR_ONLY
def test_workflow_with_json_parameter_with_tensor_input(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # Same as test_workflow_with_json_parameter_tensor_native, but the image arrives
    # ALREADY materialised as a CHW RGB device tensor (is_tensor_materialised() == True),
    # so the OD model block runs its on-device tensor path. Results must match the
    # numpy-input variant.
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
            "image": numpy_image_as_tensor(dogs_image),
            "config": '{"model_id": "yolov8n-640"}',
        }
    )

    assert len(result) == 1
    assert set(result[0].keys()) == {"json_parser", "model_predictions"}
    assert result[0]["json_parser"] == "yolov8n-640"
    # Under ENABLE_TENSOR_DATA_REPRESENTATION the model block emits a native
    # inference_models.Detections (torch-backed) rather than sv.Detections.
    predictions = result[0]["model_predictions"]
    assert isinstance(predictions, NativeDetections)
    # Same semantic result as the numpy path: two dogs detected (COCO class_id 16).
    assert len(predictions.xyxy) == 2
    assert predictions.class_id.tolist() == [16, 16]
