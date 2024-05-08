import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.entities.base import StepExecutionMode
from inference.enterprise.workflows.errors import RuntimeInputError, StepExecutionError
from inference.enterprise.workflows.execution_engine.core import ExecutionEngine

OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
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
        [180, 273, 244, 383],
        [271, 266, 328, 383],
        [552, 259, 598, 365],
        [113, 269, 145, 347],
        [416, 258, 457, 365],
        [521, 257, 555, 360],
        [387, 264, 414, 342],
        [158, 267, 183, 349],
        [324, 256, 345, 320],
        [341, 261, 362, 338],
        [247, 251, 262, 284],
        [239, 251, 249, 282],
    ]
)
EXPECTED_OBJECT_DETECTION_CONFIDENCES = np.array(
    [
        0.84284,
        0.83957,
        0.81555,
        0.80455,
        0.75804,
        0.75794,
        0.71715,
        0.71408,
        0.71003,
        0.56938,
        0.54092,
        0.43511,
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
    detections = sv.Detections.from_inference(result["result"][0])
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
    detections_1 = sv.Detections.from_inference(result["result"][0])
    detections_2 = sv.Detections.from_inference(result["result"][1])
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

    # then
    assert set(result.keys()) == {
        "result"
    }, "Only single output key should be extracted"
    assert len(result["result"]) == 1, "Result for single image is expected"
    detections = sv.Detections.from_inference(result["result"][0])
    assert np.allclose(
        detections.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES[:4],
        atol=1,
    ), "Expected bboxes to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES[:4],
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
