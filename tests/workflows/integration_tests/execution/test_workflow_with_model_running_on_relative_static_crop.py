import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.core import ExecutionEngine

RELATIVE_STATIC_CROP_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.7},
        {"type": "WorkflowParameter", "name": "x_center"},
        {"type": "WorkflowParameter", "name": "y_center"},
        {"type": "WorkflowParameter", "name": "width"},
        {"type": "WorkflowParameter", "name": "height"},
    ],
    "steps": [
        {
            "type": "RelativeStaticCrop",
            "name": "crop",
            "image": "$inputs.image",
            "x_center": "$inputs.x_center",
            "y_center": "$inputs.y_center",
            "width": "$inputs.width",
            "height": "$inputs.height",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$steps.crop.crops",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "crop", "selector": "$steps.crop.crops"},
        {"type": "JsonField", "name": "result", "selector": "$steps.detection.*"},
        {
            "type": "JsonField",
            "name": "result_in_own_coordinates",
            "selector": "$steps.detection.*",
            "coordinates_system": "own",
        },
    ],
}


def test_static_crop_workflow_when_minimal_valid_input_provided(
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
        workflow_definition=RELATIVE_STATIC_CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "x_center": 0.5,
            "y_center": 0.5,
            "height": 0.5,
            "width": 0.5,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    assert set(result[0].keys()) == {
        "result",
        "result_in_own_coordinates",
        "crop",
    }, "Expected to see all defined outputs"
    assert result[0]["crop"].numpy_image.shape == (
        212,
        320,
        3,
    ), "Expected cropped image to be half the size of original image"
    assert np.allclose(
        crowd_image[106:318, 160:480, :], result[0]["crop"].numpy_image, atol=5
    ), "Expected crop to be made in central area of input image, as specified in inputs"
    parent_coordinates_detections: sv.Detections = result[0]["result"]["predictions"]
    assert np.allclose(
        parent_coordinates_detections.xyxy,
        np.array(
            [
                [181, 273, 240, 317],
                [419, 258, 458, 317],
                [160, 268, 184, 317],
                [270, 266, 331, 317],
                [250, 252, 261, 283],
                [390, 267, 415, 318],
            ]
        ),
        atol=1,
    ), "Expected detections in parent coordinates to be as manually validated at test creation"
    own_coordinates_detections: sv.Detections = result[0]["result_in_own_coordinates"][
        "predictions"
    ]
    assert np.allclose(
        own_coordinates_detections.xyxy,
        np.array(
            [
                [21, 167, 80, 211],
                [259, 152, 298, 211],
                [0, 162, 24, 211],
                [110, 160, 171, 211],
                [90, 146, 101, 177],
                [230, 161, 255, 212],
            ]
        ),
        atol=1,
    ), "Expected detections in own coordinates to be as manually validated at test creation"


def test_test_static_crop_workflow_when_crop_coordinate_not_provided(
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
        workflow_definition=RELATIVE_STATIC_CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "model_id": "yolov8n-640",
                "y_center": 0.5,
                "height": 0.5,
                "width": 0.5,
            }
        )


def test_test_static_crop_workflow_when_invalid_crop_coordinates_defined(
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
        workflow_definition=RELATIVE_STATIC_CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "model_id": "yolov8n-640",
                "x_center": 1.5,
                "y_center": 0.5,
                "height": 0.5,
                "width": 0.5,
            }
        )
