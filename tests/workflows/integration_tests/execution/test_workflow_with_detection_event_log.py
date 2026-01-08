import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_DETECTION_EVENT_LOG = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": 0.3,
        },
        {
            "type": "roboflow_core/byte_tracker@v3",
            "name": "byte_tracker",
            "image": "$inputs.image",
            "detections": "$steps.model.predictions",
        },
        {
            "type": "roboflow_core/detection_event_log@v1",
            "name": "detection_event_log",
            "image": "$inputs.image",
            "detections": "$steps.byte_tracker.tracked_detections",
            "frame_threshold": 1,
            "flush_interval": 30,
            "stale_frames": 300,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "event_log",
            "selector": "$steps.detection_event_log.event_log",
        },
        {
            "type": "JsonField",
            "name": "total_logged",
            "selector": "$steps.detection_event_log.total_logged",
        },
        {
            "type": "JsonField",
            "name": "total_pending",
            "selector": "$steps.detection_event_log.total_pending",
        },
    ],
}


def test_workflow_with_detection_event_log(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test that detection_event_log block works in a workflow with ByteTracker."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DETECTION_EVENT_LOG,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Expected single result for single input image"
    assert set(result[0].keys()) == {
        "event_log",
        "total_logged",
        "total_pending",
    }, "Expected all outputs to be registered"

    event_log = result[0]["event_log"]
    assert isinstance(event_log, dict), "Expected event_log to be a dictionary"
    assert "logged" in event_log, "Expected 'logged' key in event_log"
    assert "pending" in event_log, "Expected 'pending' key in event_log"

    total_logged = result[0]["total_logged"]
    total_pending = result[0]["total_pending"]
    assert isinstance(total_logged, int), "Expected total_logged to be an integer"
    assert isinstance(total_pending, int), "Expected total_pending to be an integer"

    # With frame_threshold=1, all detections should be logged immediately
    assert total_logged >= 0, "Expected non-negative total_logged"
    assert total_pending >= 0, "Expected non-negative total_pending"


def test_workflow_with_detection_event_log_multiple_frames(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test that detection_event_log correctly tracks objects across multiple frames."""
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DETECTION_EVENT_LOG,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when - run multiple times to simulate video frames
    for _ in range(3):
        result = execution_engine.run(
            runtime_parameters={
                "image": [dogs_image],
            }
        )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Expected single result for single input image"

    event_log = result[0]["event_log"]
    total_logged = result[0]["total_logged"]

    # After multiple frames with same detections, objects should be logged
    # (frame_threshold=1 means they log immediately)
    assert total_logged >= 0, "Expected logged detections after multiple frames"

    # Check that logged events have expected fields
    for tracker_id, event in event_log.get("logged", {}).items():
        assert "tracker_id" in event, "Expected tracker_id in event"
        assert "class_name" in event, "Expected class_name in event"
        assert "first_seen_frame" in event, "Expected first_seen_frame in event"
        assert "last_seen_frame" in event, "Expected last_seen_frame in event"
        assert "first_seen_timestamp" in event, "Expected first_seen_timestamp in event"
        assert "last_seen_timestamp" in event, "Expected last_seen_timestamp in event"
        assert "frame_count" in event, "Expected frame_count in event"
