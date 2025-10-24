import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError, StepExecutionError, ClientCausedStepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

CONSENSUS_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_1",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.3,
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_2",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.83,
        },
        {
            "type": "DetectionsConsensus",
            "name": "consensus",
            "predictions_batches": [
                "$steps.detection_1.predictions",
                "$steps.detection_2.predictions",
            ],
            "required_votes": 2,
            "required_objects": {
                "person": 2,
            },
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.consensus.*"}
    ],
}

EXPECTED_OBJECT_DETECTION_BBOXES = np.array(
    [
        [180, 273, 244, 383],
        [271, 266, 328, 383],
    ]
)
EXPECTED_OBJECT_DETECTION_CONFIDENCES = np.array(
    [
        0.84284,
        0.83957,
    ]
)


@add_to_workflows_gallery(
    category="Workflows with multiple models",
    use_case_title="Workflow presenting models ensemble",
    use_case_description="""
This workflow presents how to combine predictions from multiple models running against the same 
input image with the block called Detections Consensus. 

First, we run two object detections models steps and we combine their predictions. Fusion may be 
performed in different scenarios based on Detections Consensus step configuration:

- you may combine predictions from models detecting different objects and then require only single 
model vote to add predicted bounding box to the output prediction

- you may combine predictions from models detecting the same objects and expect multiple positive 
votes to accept bounding box to the output prediction - this way you may improve the quality of 
predictions
    """,
    workflow_definition=CONSENSUS_WORKFLOW,
    workflow_name_in_app="detections-consensus",
)
def test_consensus_workflow_when_minimal_valid_input_provided(
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
        workflow_definition=CONSENSUS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": crowd_image, "model_id": "yolov8n-640"}
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "result"
    }, "Only single output key should be extracted"
    detections: sv.Detections = result[0]["result"]["predictions"]
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
    assert (
        result[0]["result"]["object_present"] is True
    ), "Detected 2 instances of person in combined prediction, so `object_present` should be marked True"
    assert (
        abs(result[0]["result"]["presence_confidence"]["person"] - 0.84284) < 1e-4
    ), "Expected presence confidence to be max of merged person class confidence"


def test_consensus_workflow_when_batch_input_provided(
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
        workflow_definition=CONSENSUS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, crowd_image],
            "model_id": "yolov8n-640",
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "result"
    }, "Only single output key should be extracted in first output"
    assert set(result[1].keys()) == {
        "result"
    }, "Only single output key should be extracted in second output"
    detections_1: sv.Detections = result[0]["result"]["predictions"]
    detections_2: sv.Detections = result[1]["result"]["predictions"]
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


def test_consensus_workflow_when_confidence_is_restricted_by_input_parameter(
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
        workflow_definition=CONSENSUS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "confidence": 0.8,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    detections: sv.Detections = result[0]["result"]["predictions"]
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


def test_consensus_workflow_when_image_not_provided_in_input(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CONSENSUS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "model_id": "yolov8n-640",
            }
        )


def test_consensus_workflow_when_model_id_cannot_be_resolved_to_valid_model(
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
        workflow_definition=CONSENSUS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(ClientCausedStepExecutionError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "model_id": "invalid",
            }
        )
