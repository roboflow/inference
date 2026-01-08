import datetime
import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.detection_event_log.v1 import (
    DetectionEventLogBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def create_workflow_image_data(
    video_id: str = "vid_1",
    frame_number: int = 1,
    fps: float = 30.0,
) -> WorkflowImageData:
    """Helper to create WorkflowImageData with video metadata."""
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=fps,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    parent_metadata = ImageParentMetadata(parent_id=f"img_{frame_number}")
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=image,
        video_metadata=metadata,
    )


def create_detections(
    tracker_ids: list,
    class_names: list = None,
) -> sv.Detections:
    """Helper to create detections with tracker IDs."""
    n = len(tracker_ids)
    if n == 0:
        return sv.Detections.empty()

    xyxy = np.array([[i * 10, i * 10, i * 10 + 50, i * 10 + 50] for i in range(n)])
    detections = sv.Detections(
        xyxy=xyxy,
        tracker_id=np.array(tracker_ids),
    )
    if class_names:
        detections.data["class_name"] = class_names
    return detections


def test_first_detection_creates_pending_event() -> None:
    """Test that first detection creates a pending event."""
    # Given
    block = DetectionEventLogBlockV1()
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    assert result["total_pending"] == 1
    assert result["total_logged"] == 0
    assert "1" in result["event_log"]["pending"]
    event = result["event_log"]["pending"]["1"]
    assert event["tracker_id"] == 1
    assert event["class_name"] == "dog"
    assert event["first_seen_frame"] == 1
    assert event["frame_count"] == 1


def test_event_moves_to_logged_after_threshold() -> None:
    """Test that event moves from pending to logged after frame_threshold."""
    # Given
    block = DetectionEventLogBlockV1()
    frame_threshold = 5

    # When - simulate 5 frames with same detection
    for frame in range(1, 6):
        image_data = create_workflow_image_data(frame_number=frame)
        detections = create_detections([1], ["cat"])
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=frame_threshold,
            flush_interval=30,
            stale_frames=300,
        )

    # Then
    assert result["total_logged"] == 1
    assert result["total_pending"] == 0
    assert "1" in result["event_log"]["logged"]
    event = result["event_log"]["logged"]["1"]
    assert event["frame_count"] == 5
    assert event["first_seen_frame"] == 1
    assert event["last_seen_frame"] == 5


def test_multiple_detections_tracked_separately() -> None:
    """Test that multiple tracker IDs are tracked separately."""
    # Given
    block = DetectionEventLogBlockV1()
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1, 2, 3], ["dog", "cat", "person"])

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    assert result["total_pending"] == 3
    assert result["total_logged"] == 0
    assert "1" in result["event_log"]["pending"]
    assert "2" in result["event_log"]["pending"]
    assert "3" in result["event_log"]["pending"]


def test_frame_count_increments_on_each_detection() -> None:
    """Test that frame_count increments each time a tracker is seen."""
    # Given
    block = DetectionEventLogBlockV1()

    # When - run 3 frames
    for frame in range(1, 4):
        image_data = create_workflow_image_data(frame_number=frame)
        detections = create_detections([1], ["dog"])
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=5,
            flush_interval=30,
            stale_frames=300,
        )

    # Then
    event = result["event_log"]["pending"]["1"]
    assert event["frame_count"] == 3
    assert event["first_seen_frame"] == 1
    assert event["last_seen_frame"] == 3


def test_empty_detections_returns_current_log() -> None:
    """Test that empty detections returns current log without changes."""
    # Given
    block = DetectionEventLogBlockV1()

    # First add a detection
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # When - run with empty detections
    image_data = create_workflow_image_data(frame_number=2)
    empty_detections = sv.Detections.empty()
    result = block.run(
        image=image_data,
        detections=empty_detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then - event should still be there, frame_count unchanged
    assert result["total_pending"] == 1
    event = result["event_log"]["pending"]["1"]
    assert event["frame_count"] == 1


def test_detections_without_tracker_id_returns_current_log() -> None:
    """Test that detections without tracker_id returns current log."""
    # Given
    block = DetectionEventLogBlockV1()
    image_data = create_workflow_image_data(frame_number=1)
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
    )  # No tracker_id

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    assert result["total_pending"] == 0
    assert result["total_logged"] == 0


def test_stale_events_removed_after_flush_interval() -> None:
    """Test that stale events are removed during flush."""
    # Given
    block = DetectionEventLogBlockV1()
    flush_interval = 5
    stale_frames = 3

    # Add detection on frame 1
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=1,  # Log immediately
        flush_interval=flush_interval,
        stale_frames=stale_frames,
    )

    # When - advance to frame 6 (flush_interval reached) without seeing tracker 1
    # and stale_frames (3) exceeded since last seen at frame 1
    for frame in range(2, 7):
        image_data = create_workflow_image_data(frame_number=frame)
        detections = create_detections([2], ["cat"])  # Different tracker
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=1,
            flush_interval=flush_interval,
            stale_frames=stale_frames,
        )

    # Then - tracker 1 should be removed (stale), tracker 2 should be logged
    assert "1" not in result["event_log"]["logged"]
    assert "1" not in result["event_log"]["pending"]
    assert "2" in result["event_log"]["logged"]


def test_separate_video_streams_tracked_independently() -> None:
    """Test that different video IDs maintain separate event logs."""
    # Given
    block = DetectionEventLogBlockV1()

    # When - add detection to video 1
    image_data_1 = create_workflow_image_data(video_id="vid_1", frame_number=1)
    detections_1 = create_detections([1], ["dog"])
    result_1 = block.run(
        image=image_data_1,
        detections=detections_1,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Add detection to video 2
    image_data_2 = create_workflow_image_data(video_id="vid_2", frame_number=1)
    detections_2 = create_detections([1], ["cat"])  # Same tracker_id, different video
    result_2 = block.run(
        image=image_data_2,
        detections=detections_2,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then - both videos should have their own event for tracker 1
    assert result_1["total_pending"] == 1
    assert result_1["event_log"]["pending"]["1"]["class_name"] == "dog"

    assert result_2["total_pending"] == 1
    assert result_2["event_log"]["pending"]["1"]["class_name"] == "cat"


def test_timestamp_calculated_from_fps() -> None:
    """Test that timestamp is calculated as frame_number / fps."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    frame_number = 90
    image_data = create_workflow_image_data(frame_number=frame_number, fps=fps)
    detections = create_detections([1], ["dog"])

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    event = result["event_log"]["pending"]["1"]
    expected_timestamp = frame_number / fps  # 90 / 30 = 3.0
    assert event["first_seen_timestamp"] == pytest.approx(expected_timestamp)


def test_detections_passed_through() -> None:
    """Test that detections are passed through in output."""
    # Given
    block = DetectionEventLogBlockV1()
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1, 2], ["dog", "cat"])

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    assert result["detections"] is detections
    assert len(result["detections"]) == 2


def test_class_name_fallback_to_class_id() -> None:
    """Test that class_id is used when class_name not available."""
    # Given
    block = DetectionEventLogBlockV1()
    image_data = create_workflow_image_data(frame_number=1)
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        tracker_id=np.array([1]),
        class_id=np.array([42]),
    )

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    event = result["event_log"]["pending"]["1"]
    assert event["class_name"] == "class_42"


def test_new_tracker_added_mid_stream() -> None:
    """Test that new trackers can be added after initial detections."""
    # Given
    block = DetectionEventLogBlockV1()

    # Frame 1: tracker 1 only
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # When - Frame 2: tracker 1 and new tracker 2
    image_data = create_workflow_image_data(frame_number=2)
    detections = create_detections([1, 2], ["dog", "cat"])
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    assert result["total_pending"] == 2
    assert result["event_log"]["pending"]["1"]["frame_count"] == 2
    assert result["event_log"]["pending"]["2"]["frame_count"] == 1
    assert result["event_log"]["pending"]["2"]["first_seen_frame"] == 2
