import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.entities.base import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError, StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine

OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "RoboflowDetectionsInferenceSlicer",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.detection.*"}
    ],
}

EXPECTED_OBJECT_DETECTION_BBOXES = np.array(
    [
        [112, 270, 145, 320],
        [249, 252, 261, 284],
        [240, 251, 250, 283],
        [158, 274, 180, 318],
        [420, 258, 458, 320],
        [342, 262, 361, 320],
        [358, 260, 374, 292],
        [552, 260, 576, 320],
        [271, 283, 300, 320],
        [270, 267, 320, 383],
        [181, 273, 236, 384],
        [158, 268, 184, 349],
        [138, 295, 156, 315],
        [119, 283, 136, 306],
        [144, 264, 164, 311],
        [388, 267, 415, 342],
        [303, 308, 326, 380],
        [553, 259, 598, 365],
        [523, 258, 557, 360],
        [525, 285, 546, 317]
    ]
)
EXPECTED_OBJECT_DETECTION_CONFIDENCES = np.array(
    [   0.86251378,
        0.74914879,
        0.68643397,
        0.31664759,
        0.84777606,
        0.75367212,
        0.70453328,
        0.50743264,
        0.35423759,
        0.94660336,
        0.87591493,
        0.72702628,
        0.57335436,
        0.48346052,
        0.43254578,
        0.8038348,
        0.59269905,
        0.90392375,
        0.85327947,
        0.39380407
    ]
)


@pytest.mark.asyncio
async def test_object_detection_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={"image": crowd_image, "model_id": "yolov8n-640"}
    )

    # then
    assert set(result.keys()) == {
        "result"
    }, "Only single output key should be extracted"
    assert len(result["result"]) == 1, "Result for single image is expected"
    detections: sv.Detections = result["result"][0]["predictions"]
    assert np.allclose(
        detections.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES,
        atol=1,
    ), "Expected bboxes to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES,
        atol=0.01,
    ), "Expected confidences to match what was validated manually as workflow outcome"


@pytest.mark.asyncio
async def test_object_detection_workflow_when_batch_input_provided(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={
            "image": [crowd_image, crowd_image],
            "model_id": "yolov8n-640",
        }
    )

    # then
    assert set(result.keys()) == {
        "result"
    }, "Only single output key should be extracted"
    assert len(result["result"]) == 2, "Results for botch images are expected"
    detections_1: sv.Detections = result["result"][0]["predictions"]
    detections_2: sv.Detections = result["result"][1]["predictions"]
    assert np.allclose(
        detections_1.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES,
        atol=1,
    ), "Expected bboxes for first image to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections_1.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES,
        atol=0.01,
    ), "Expected confidences for first image to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections_2.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES,
        atol=1,
    ), "Expected bboxes for 2nd image to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections_2.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES,
        atol=0.01,
    ), "Expected confidences for 2nd image to match what was validated manually as workflow outcome"


@pytest.mark.asyncio
async def test_object_detection_workflow_when_confidence_is_restricted_by_input_parameter(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "confidence": 0.8,
        }
    )
    expected_detection_indices = EXPECTED_OBJECT_DETECTION_CONFIDENCES >= 0.8

    # then
    assert set(result.keys()) == {
        "result"
    }, "Only single output key should be extracted"
    assert len(result["result"]) == 1, "Result for single image is expected"
    detections: sv.Detections = result["result"][0]["predictions"]
    print(detections.xyxy)
    assert np.allclose(
        detections.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES[expected_detection_indices],
        atol=1,
    ), "Expected bboxes to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES[expected_detection_indices],
        atol=0.01,
    ), "Expected confidences to match what was validated manually as workflow outcome"


@pytest.mark.asyncio
async def test_object_detection_workflow_when_model_id_not_provided_in_input(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = await execution_engine.run_async(
            runtime_parameters={
                "image": crowd_image,
            }
        )


@pytest.mark.asyncio
async def test_object_detection_workflow_when_image_not_provided_in_input(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = await execution_engine.run_async(
            runtime_parameters={
                "model_id": "yolov8n-640",
            }
        )


@pytest.mark.asyncio
async def test_object_detection_workflow_when_confidence_provided_with_invalid_type(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = await execution_engine.run_async(
            runtime_parameters={
                "image": crowd_image,
                "model_id": "yolov8n-640",
                "confidence": None,
            }
        )


@pytest.mark.asyncio
async def test_object_detection_workflow_when_model_id_cannot_be_resolved_to_valid_model(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    with pytest.raises(StepExecutionError):
        _ = await execution_engine.run_async(
            runtime_parameters={
                "image": crowd_image,
                "model_id": "invalid",
            }
        )
