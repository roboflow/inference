import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

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
            "class_filter": ["car"],
        },
        {
            "type": "DetectionOffset",
            "name": "offset",
            "predictions": "$steps.detection.predictions",
            "image_metadata": "$steps.detection.image",
            "prediction_type": "$steps.detection.prediction_type",
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


def test_detection_plus_ocr_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MULTI_STAGES_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": license_plate_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    print(result[0])
    assert set(result[0].keys()) == {
        "plates_ocr",
        "plates_crops",
        "cars_crops",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["cars_crops"]) == 3, "Expected 3 cars to be detected"
    assert np.allclose(
        result[0]["cars_crops"][0].numpy_image,
        license_plate_image[475:666, 109:351, :],
        atol=5,
    ), "Expected car to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result[0]["cars_crops"][1].numpy_image,
        license_plate_image[380:990, 761:1757, :],
        atol=5,
    ), "Expected car to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result[0]["cars_crops"][2].numpy_image,
        license_plate_image[489:619, 417:588, :],
        atol=5,
    ), "Expected car to be detected exactly in coordinates matching reference run"
    assert (
        len(result[0]["plates_crops"]) == 3
    ), "Expected 3 sets of plates crops, one set for each crop of car, as there were three cars detected originally"
    assert (
        len(result[0]["plates_crops"][0]) == 1
    ), "Single plate detected for first car crop"
    assert (
        len(result[0]["plates_crops"][1]) == 1
    ), "Single plate detected for second car crop"
    assert (
        len(result[0]["plates_crops"][2]) == 1
    ), "Single plate detected for third car crop"
    assert np.allclose(
        result[0]["plates_crops"][0][0].numpy_image,
        license_plate_image[475 + 94 : 475 + 162, 109 + 58 : 109 + 179, :],
        atol=5,
    ), "Expected license plate to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result[0]["plates_crops"][1][0].numpy_image,
        license_plate_image[380 + 373 : 380 + 486, 761 + 593 : 761 + 873, :],
        atol=5,
    ), "Expected license plate to be detected exactly in coordinates matching reference run"
    assert np.allclose(
        result[0]["plates_crops"][2][0].numpy_image,
        license_plate_image[489 + 56 : 489 + 118, 417 + 49 : 417 + 143, :],
        atol=5,
    ), "Expected license plate to be detected exactly in coordinates matching reference run"
    assert len(result[0]["plates_ocr"]) == 3, "Expected 3 predictions with OCRed values"
    # TODO: verify the issue
    # For some reason at different platform OCR gives different results, despite
    # checking to operate on the same input images as in reference runs
