import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.entities.base import StepExecutionMode
from inference.enterprise.workflows.execution_engine.core import ExecutionEngine

MULTI_STAGES_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
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
                "operator": "==",
                "reference_value": "car",
            },
        },
        {
            "type": "DetectionOffset",
            "name": "offset",
            "predictions": "$steps.filter.predictions",
            "image_metadata": "$steps.filter.image",
            "prediction_type": "$steps.filter.prediction_type",
            "offset_width": 10,
            "offset_height": 10,
        },
        {
            "type": "Crop",
            "name": "cars_crops",
            "image": "$inputs.image",
            "predictions": "$steps.offset.predictions",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "plates_detection",
            "image": "$steps.cars_crops.crops",
            "model_id": "vehicle-registration-plates-trudk/2",
        },
        {
            "type": "DetectionOffset",
            "name": "plates_offset",
            "predictions": "$steps.plates_detection.predictions",
            "image_metadata": "$steps.plates_detection.image",
            "prediction_type": "$steps.plates_detection.prediction_type",
            "offset_width": 50,
            "offset_height": 50,
        },
        {
            "type": "Crop",
            "name": "plates_crops",
            "image": "$steps.cars_crops.crops",
            "predictions": "$steps.plates_offset.predictions",
        },
        {
            "type": "OCRModel",
            "name": "ocr",
            "image": "$steps.plates_crops.crops",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "cars_crops",
            "selector": "$steps.cars_crops.crops",
        },
        {
            "type": "JsonField",
            "name": "plates_crops",
            "selector": "$steps.plates_crops.crops",
        },
        {"type": "JsonField", "name": "plates_ocr", "selector": "$steps.ocr.result"},
    ],
}


@pytest.mark.asyncio
async def test_detection_plus_ocr_workflow_when_minimal_valid_input_provided(
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
        "plates_ocr",
        "plates_crops",
        "cars_crops",
    }, "Expected all declared outputs to be delivered"
    assert len(result["cars_crops"]) == 3, "Expected 3 cars to be detected"
    assert np.allclose(
        result["cars_crops"][0]["value"],
        license_plate_image[475:666, 109:351, :],
        atol=5,
    ), "Expected car to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result["cars_crops"][1]["value"],
        license_plate_image[380:990, 761:1757, :],
        atol=5,
    ), "Expected car to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result["cars_crops"][2]["value"],
        license_plate_image[489:619, 417:588, :],
        atol=5,
    ), "Expected car to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result["plates_crops"][0]["value"],
        license_plate_image[475 + 94 : 475 + 162, 109 + 58 : 109 + 179, :],
        atol=5,
    ), "Expected license plate to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result["plates_crops"][1]["value"],
        license_plate_image[380 + 373 : 380 + 486, 761 + 593 : 761 + 873, :],
        atol=5,
    ), "Expected license plate to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result["plates_crops"][2]["value"],
        license_plate_image[489 + 56 : 489 + 118, 417 + 49 : 417 + 143, :],
        atol=5,
    ), "Expected license plate to be detected exactly in coordinates matching reference run"
    assert len(result["plates_ocr"]) == 3, "Expected 3 predictions with OCRed values"
    # For some reason at different platform OCR gives different results, despite
    # checking to operate on the same input images as in reference runs
