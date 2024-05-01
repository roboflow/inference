import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.entities.base import StepExecutionMode
from inference.enterprise.workflows.execution_engine.core import ExecutionEngine

MULTI_STAGES_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionFilter",
            "name": "filter",
            "predictions": "$steps.detection.predictions",
            "image_metadata": "$steps.detection.image",
            "prediction_type": "$steps.detection.prediction_type",
            "filter_definition": {
                "type": "DetectionFilterDefinition",
                "field_name": "class",
                "operator": "equal",
                "reference_value": "car",
            },
        },
        {
            "type": "DetectionOffset",
            "name": "offset",
            "predictions": "$steps.filter.predictions",
            "image_metadata": "$steps.filter.image",
            "prediction_type": "$steps.filter.prediction_type",
            "offset_x": 10,
            "offset_y": 10,
        },
        {
            "type": "Crop",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.offset.predictions",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "plates_detection",
            "image": "$steps.crop.crops",
            "model_id": "vehicle-registration-plates-trudk/2",
        },
        {
            "type": "DetectionOffset",
            "name": "plates_offset",
            "predictions": "$steps.plates_detection.predictions",
            "image_metadata": "$steps.plates_detection.image",
            "prediction_type": "$steps.plates_detection.prediction_type",
            "offset_x": 50,
            "offset_y": 50,
        },
        {
            "type": "Crop",
            "name": "plates_crops",
            "image": "$steps.crop.crops",
            "predictions": "$steps.plates_offset.predictions",
        },
        {
            "type": "OCRModel",
            "name": "ocr",
            "image": "$steps.plates_crops.crops",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "plates_ocr", "selector": "$steps.ocr.result"},
    ],
}


@pytest.mark.asyncio
async def test_static_crop_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MULTI_STAGES_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={
            "image": license_plate_image,
        }
    )

    # then
    assert set(result.keys()) == {
        "plates_ocr"
    }, "Expected all declared outputs to be delivered"
    assert len(result["plates_ocr"]) == 3, "Expected 3 predictions with OCRed values"
    assert result["plates_ocr"] == [
        "",
        "23948072",
        "",
    ], "Expected OCR results to be as verified manually while creating the test. Two outputs are empty due to insufficient quality of OCR model"
