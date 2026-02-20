import datetime

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

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

    # Relative times are frame-based: (frame - 1) / fps
    # WorkflowImageData defaults to fps=30 when no video metadata is provided
    # Frame 1: (1-1)/30 = 0.0, Frame 3: (3-1)/30 = 0.0666...
    # Note: With auto-extraction of reference_timestamp from frame_timestamp,
    # absolute timestamps (*_timestamp) are now also present

    assert total_logged == 2
    assert result[0]["total_pending"] == 0
    assert event_log["pending"] == {}

    # Verify each logged event has the expected fields
    for tracker_id in ["1", "2"]:
        event = event_log["logged"][tracker_id]
        assert event["tracker_id"] == int(tracker_id)
        assert event["class_name"] == "dog"
        assert event["first_seen_frame"] == 1
        assert event["first_seen_relative"] == 0.0
        assert event["last_seen_frame"] == 3
        assert event["last_seen_relative"] == 2 / 30
        assert event["frame_count"] == 3
        # Auto-extracted timestamps should also be present
        assert "first_seen_timestamp" in event
        assert "last_seen_timestamp" in event


def test_workflow_with_detection_event_log_auto_extracts_reference_timestamp(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test that detection_event_log auto-extracts reference_timestamp from frame_timestamp."""
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

    # Create WorkflowImageData with frame_timestamp to trigger auto-extraction
    frame_ts = datetime.datetime.fromtimestamp(1726570875.0).astimezone(
        tz=datetime.timezone.utc
    )
    metadata = VideoMetadata(
        video_identifier="test_video",
        frame_number=1,
        fps=30.0,
        frame_timestamp=frame_ts,
        comes_from_video_file=True,
    )
    parent_metadata = ImageParentMetadata(parent_id="test_frame_1")
    image_with_metadata = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=dogs_image,
        video_metadata=metadata,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [image_with_metadata],
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Expected single result for single input image"

    event_log = result[0]["event_log"]
    total_logged = result[0]["total_logged"]

    # With frame_threshold=1, all detections should be logged immediately
    assert total_logged >= 1, "Expected at least one logged detection"

    # Check that absolute timestamps are present (auto-extracted from frame_timestamp)
    logged_events = event_log.get("logged", {})
    if logged_events:
        first_event = next(iter(logged_events.values()))
        # Relative timestamps should always be present
        assert "first_seen_relative" in first_event
        assert "last_seen_relative" in first_event
        # Absolute timestamps should be present due to auto-extraction
        assert (
            "first_seen_timestamp" in first_event
        ), "Expected auto-extracted first_seen_timestamp"
        assert (
            "last_seen_timestamp" in first_event
        ), "Expected auto-extracted last_seen_timestamp"
        # first_seen_relative = 0.0, so first_seen_timestamp should equal reference_timestamp
        # reference_timestamp = frame_ts - relative_time = 1726570875.0 - 0.0 = 1726570875.0
        assert first_event["first_seen_timestamp"] == 1726570875.0
