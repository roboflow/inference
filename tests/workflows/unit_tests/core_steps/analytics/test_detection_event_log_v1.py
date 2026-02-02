import datetime

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.analytics.detection_event_log.v1 import (
    MAX_VIDEOS,
    BlockManifest,
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
    comes_from_video_file: bool = True,
    frame_timestamp: datetime.datetime = None,
) -> WorkflowImageData:
    """Helper to create WorkflowImageData with video metadata."""
    if frame_timestamp is None:
        frame_timestamp = datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        )
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=fps,
        frame_timestamp=frame_timestamp,
        comes_from_video_file=comes_from_video_file,
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


def test_relative_time_calculated_from_internal_frame_counter() -> None:
    """Test that relative time is calculated from internal frame counter and fps."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    image_data = create_workflow_image_data(frame_number=90, fps=fps)
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
    # Internal frame counter is 1 for first call, so relative = (1-1)/30 = 0.0
    assert event["first_seen_relative"] == pytest.approx(0.0)
    # first_seen_timestamp IS present due to auto-extraction from frame_timestamp
    # (frame_timestamp is always present in VideoMetadata)
    assert "first_seen_timestamp" in event


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


def test_absolute_timestamps_with_reference() -> None:
    """Test that absolute timestamps are calculated correctly with reference_timestamp."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    reference_timestamp = 1000.0  # Unix timestamp when video started
    image_data = create_workflow_image_data(frame_number=90, fps=fps)
    detections = create_detections([1], ["dog"])

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
        reference_timestamp=reference_timestamp,
    )

    # Then
    event = result["event_log"]["pending"]["1"]
    # Internal frame counter is 1, so first_seen_relative = (1-1)/30 = 0.0
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["last_seen_relative"] == pytest.approx(0.0)
    # first_seen_timestamp = first_seen_relative + reference_timestamp = 0.0 + 1000.0
    assert event["first_seen_timestamp"] == pytest.approx(1000.0)
    assert event["last_seen_timestamp"] == pytest.approx(1000.0)


def test_relative_timestamps_update_over_frames() -> None:
    """Test that last_seen_relative updates as object is tracked across frames."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0

    # When - run 3 frames
    for frame in [30, 60, 90]:  # metadata frame numbers (not used for relative calc)
        image_data = create_workflow_image_data(frame_number=frame, fps=fps)
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
    # Internal frame counter: 1, 2, 3
    # first_seen_relative = (1-1)/30 = 0.0
    # last_seen_relative = (3-1)/30 = 2/30 ≈ 0.0667
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["last_seen_relative"] == pytest.approx(2.0 / 30.0)


def test_relative_timestamps_in_logged_events() -> None:
    """Test that relative timestamps are included in logged events."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    frame_threshold = 3

    # When - run enough frames to log the event
    for frame in [30, 60, 90]:
        image_data = create_workflow_image_data(frame_number=frame, fps=fps)
        detections = create_detections([1], ["dog"])
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=frame_threshold,
            flush_interval=30,
            stale_frames=300,
        )

    # Then
    assert result["total_logged"] == 1
    event = result["event_log"]["logged"]["1"]
    # Internal frame counter: 1, 2, 3
    # first_seen_relative = (1-1)/30 = 0.0
    # last_seen_relative = (3-1)/30 = 2/30 ≈ 0.0667
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["last_seen_relative"] == pytest.approx(2.0 / 30.0)


def test_absolute_timestamp_equals_relative_plus_reference() -> None:
    """Test that absolute timestamp equals relative timestamp plus reference."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    reference_timestamp = 1726570875.0  # Unix timestamp when video started
    image_data = create_workflow_image_data(frame_number=30, fps=fps)
    detections = create_detections([1], ["dog"])

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
        reference_timestamp=reference_timestamp,
    )

    # Then
    event = result["event_log"]["pending"]["1"]
    # first_seen_relative = (1-1)/30 = 0.0 (internal frame 1)
    # first_seen_timestamp = 0.0 + 1726570875.0
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["first_seen_timestamp"] == pytest.approx(reference_timestamp)
    assert event["last_seen_timestamp"] == pytest.approx(
        event["last_seen_relative"] + reference_timestamp
    )


def test_fallback_fps_used_when_metadata_has_no_fps() -> None:
    """Test that fallback_fps is used when video metadata has no FPS."""
    # Given
    block = DetectionEventLogBlockV1()
    # Create image data with fps=0 (no FPS info)
    image_data = create_workflow_image_data(frame_number=1, fps=0)
    detections = create_detections([1], ["dog"])
    fallback_fps = 10.0

    # When - run 3 frames
    for _ in range(3):
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=5,
            flush_interval=30,
            stale_frames=300,
            fallback_fps=fallback_fps,
        )

    # Then
    event = result["event_log"]["pending"]["1"]
    # Internal frame counter: 1, 2, 3
    # first_seen_relative = (1-1)/10 = 0.0
    # last_seen_relative = (3-1)/10 = 0.2
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["last_seen_relative"] == pytest.approx(0.2)


def test_metadata_fps_takes_precedence_over_fallback() -> None:
    """Test that metadata FPS is used when available, not fallback_fps."""
    # Given
    block = DetectionEventLogBlockV1()
    metadata_fps = 30.0
    fallback_fps = 10.0  # Should NOT be used
    image_data = create_workflow_image_data(frame_number=1, fps=metadata_fps)
    detections = create_detections([1], ["dog"])

    # When - run 3 frames
    for _ in range(3):
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=5,
            flush_interval=30,
            stale_frames=300,
            fallback_fps=fallback_fps,
        )

    # Then
    event = result["event_log"]["pending"]["1"]
    # Should use metadata_fps=30, not fallback_fps=10
    # first_seen_relative = (1-1)/30 = 0.0
    # last_seen_relative = (3-1)/30 = 2/30 ≈ 0.0667
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["last_seen_relative"] == pytest.approx(2.0 / 30.0)


def test_fallback_fps_default_is_one() -> None:
    """Test that fallback_fps defaults to 1.0 when not specified."""
    # Given
    block = DetectionEventLogBlockV1()
    # Create image data with fps=0 (no FPS info)
    image_data = create_workflow_image_data(frame_number=1, fps=0)
    detections = create_detections([1], ["dog"])

    # When - run 3 frames without specifying fallback_fps
    for _ in range(3):
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=5,
            flush_interval=30,
            stale_frames=300,
            # fallback_fps not specified, should default to 1.0
        )

    # Then
    event = result["event_log"]["pending"]["1"]
    # With default fallback_fps=1.0:
    # first_seen_relative = (1-1)/1 = 0.0
    # last_seen_relative = (3-1)/1 = 2.0
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["last_seen_relative"] == pytest.approx(2.0)


def test_oldest_video_evicted_when_max_exceeded() -> None:
    """Test that oldest video is evicted when MAX_VIDEOS is exceeded."""
    # Given
    block = DetectionEventLogBlockV1()

    # When - add MAX_VIDEOS + 1 different video streams
    for i in range(MAX_VIDEOS + 1):
        video_id = f"vid_{i}"
        image_data = create_workflow_image_data(video_id=video_id, frame_number=1)
        detections = create_detections([1], ["dog"])
        block.run(
            image=image_data,
            detections=detections,
            frame_threshold=5,
            flush_interval=30,
            stale_frames=300,
        )

    # Then - should only have MAX_VIDEOS entries
    assert len(block._event_logs) == MAX_VIDEOS
    assert len(block._last_access) == MAX_VIDEOS
    assert len(block._frame_count) == MAX_VIDEOS

    # The oldest video (vid_0) should be evicted
    assert "vid_0" not in block._event_logs
    # The most recent video should still be present
    assert f"vid_{MAX_VIDEOS}" in block._event_logs


def test_recently_accessed_video_not_evicted() -> None:
    """Test that recently accessed videos are not evicted."""
    # Given
    block = DetectionEventLogBlockV1()

    # Add MAX_VIDEOS videos
    for i in range(MAX_VIDEOS):
        video_id = f"vid_{i}"
        image_data = create_workflow_image_data(video_id=video_id, frame_number=1)
        detections = create_detections([1], ["dog"])
        block.run(
            image=image_data,
            detections=detections,
            frame_threshold=5,
            flush_interval=30,
            stale_frames=300,
        )

    # Access vid_0 again to make it recent
    image_data = create_workflow_image_data(video_id="vid_0", frame_number=2)
    detections = create_detections([1], ["dog"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # When - add a new video that exceeds the limit
    image_data = create_workflow_image_data(video_id="vid_new", frame_number=1)
    detections = create_detections([1], ["cat"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then - vid_0 should still be present (was recently accessed)
    assert "vid_0" in block._event_logs
    # vid_1 should be evicted (oldest after vid_0 was re-accessed)
    assert "vid_1" not in block._event_logs
    # New video should be present
    assert "vid_new" in block._event_logs


def test_auto_extract_reference_timestamp_from_frame_timestamp() -> None:
    """Test that reference_timestamp is auto-extracted from frame_timestamp when not provided."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    # frame_timestamp is the wall-clock time of this frame
    frame_ts = datetime.datetime.fromtimestamp(1726570875.0).astimezone(
        tz=datetime.timezone.utc
    )
    image_data = create_workflow_image_data(frame_number=1, fps=fps, frame_timestamp=frame_ts)
    detections = create_detections([1], ["dog"])

    # When - run without providing reference_timestamp
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
        reference_timestamp=None,  # Not provided, should auto-extract
    )

    # Then - absolute timestamps should be present (auto-extracted)
    event = result["event_log"]["pending"]["1"]
    assert "first_seen_timestamp" in event
    assert "last_seen_timestamp" in event
    # first_seen_relative = (1-1)/30 = 0.0, so first_seen_timestamp = frame_ts
    assert event["first_seen_timestamp"] == pytest.approx(1726570875.0)


def test_auto_extracted_reference_timestamp_reused_across_frames() -> None:
    """Test that auto-extracted reference_timestamp is stored and reused for subsequent frames."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    # First frame at time 1000.0
    frame_ts_1 = datetime.datetime.fromtimestamp(1000.0).astimezone(
        tz=datetime.timezone.utc
    )
    # Second frame at time 1000.5 (0.5 seconds later, but internal frame 2)
    frame_ts_2 = datetime.datetime.fromtimestamp(1000.5).astimezone(
        tz=datetime.timezone.utc
    )

    # When - run first frame
    image_data_1 = create_workflow_image_data(
        frame_number=1, fps=fps, frame_timestamp=frame_ts_1
    )
    detections = create_detections([1], ["dog"])
    result_1 = block.run(
        image=image_data_1,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Run second frame with different frame_timestamp
    image_data_2 = create_workflow_image_data(
        frame_number=2, fps=fps, frame_timestamp=frame_ts_2
    )
    result_2 = block.run(
        image=image_data_2,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then - reference should be extracted from first frame and reused
    # Frame 1: relative = 0.0, reference = 1000.0 - 0.0 = 1000.0
    # Frame 2: relative = (2-1)/30 = 0.0333..., absolute = 0.0333 + 1000.0 = 1000.0333...
    event = result_2["event_log"]["pending"]["1"]
    assert event["first_seen_timestamp"] == pytest.approx(1000.0)
    assert event["last_seen_timestamp"] == pytest.approx(1000.0 + (1.0 / 30.0))

    # Verify the stored reference timestamp
    assert "vid_1" in block._reference_timestamps
    assert block._reference_timestamps["vid_1"] == pytest.approx(1000.0)


def test_explicit_reference_timestamp_takes_precedence() -> None:
    """Test that explicitly provided reference_timestamp takes precedence over auto-extraction."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    frame_ts = datetime.datetime.fromtimestamp(1726570875.0).astimezone(
        tz=datetime.timezone.utc
    )
    explicit_reference = 2000.0  # Different from frame_ts
    image_data = create_workflow_image_data(frame_number=1, fps=fps, frame_timestamp=frame_ts)
    detections = create_detections([1], ["dog"])

    # When - run with explicit reference_timestamp
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
        reference_timestamp=explicit_reference,
    )

    # Then - should use explicit reference, not auto-extracted
    event = result["event_log"]["pending"]["1"]
    # first_seen_relative = 0.0, so first_seen_timestamp = 0.0 + 2000.0
    assert event["first_seen_timestamp"] == pytest.approx(2000.0)
    # Should NOT store in _reference_timestamps when explicit value provided
    assert "vid_1" not in block._reference_timestamps


def test_reference_timestamp_evicted_with_video() -> None:
    """Test that stored reference_timestamp is evicted when video is evicted."""
    # Given
    block = DetectionEventLogBlockV1()
    frame_ts = datetime.datetime.fromtimestamp(1000.0).astimezone(
        tz=datetime.timezone.utc
    )

    # When - add MAX_VIDEOS + 1 different video streams
    for i in range(MAX_VIDEOS + 1):
        video_id = f"vid_{i}"
        image_data = create_workflow_image_data(
            video_id=video_id, frame_number=1, frame_timestamp=frame_ts
        )
        detections = create_detections([1], ["dog"])
        block.run(
            image=image_data,
            detections=detections,
            frame_threshold=5,
            flush_interval=30,
            stale_frames=300,
        )

    # Then - should only have MAX_VIDEOS entries in _reference_timestamps
    assert len(block._reference_timestamps) == MAX_VIDEOS
    # The oldest video (vid_0) should be evicted
    assert "vid_0" not in block._reference_timestamps
    # The most recent video should still be present
    assert f"vid_{MAX_VIDEOS}" in block._reference_timestamps


def test_absolute_timestamps_auto_extracted_with_frame_timestamp() -> None:
    """Test that absolute timestamps are included when frame_timestamp is available for auto-extraction."""
    # Given
    block = DetectionEventLogBlockV1()
    fps = 30.0
    frame_ts = datetime.datetime.fromtimestamp(1726570875.0).astimezone(
        tz=datetime.timezone.utc
    )
    image_data = create_workflow_image_data(frame_number=1, fps=fps, frame_timestamp=frame_ts)
    detections = create_detections([1], ["dog"])

    # When - run without reference_timestamp (should auto-extract)
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=5,
        flush_interval=30,
        stale_frames=300,
    )

    # Then - both relative and absolute timestamps should be present
    event = result["event_log"]["pending"]["1"]
    assert "first_seen_relative" in event
    assert "last_seen_relative" in event
    assert "first_seen_timestamp" in event
    assert "last_seen_timestamp" in event
    # Verify the values
    assert event["first_seen_relative"] == pytest.approx(0.0)
    assert event["first_seen_timestamp"] == pytest.approx(1726570875.0)


def test_manifest_accepts_step_output_selector_for_reference_timestamp() -> None:
    """Test that BlockManifest accepts step output selectors for reference_timestamp."""
    # Given - a manifest with reference_timestamp from a step output
    manifest_data = {
        "type": "roboflow_core/detection_event_log@v1",
        "name": "detection_event_log",
        "image": "$inputs.image",
        "detections": "$steps.byte_tracker.tracked_detections",
        "reference_timestamp": "$steps.extract_timestamp.reference_timestamp",
    }

    # When - parse the manifest
    manifest = BlockManifest.model_validate(manifest_data)

    # Then - should accept step output selector
    assert manifest.reference_timestamp == "$steps.extract_timestamp.reference_timestamp"


def test_manifest_accepts_workflow_parameter_for_reference_timestamp() -> None:
    """Test that BlockManifest still accepts workflow parameters for reference_timestamp."""
    # Given - a manifest with reference_timestamp from a workflow parameter
    manifest_data = {
        "type": "roboflow_core/detection_event_log@v1",
        "name": "detection_event_log",
        "image": "$inputs.image",
        "detections": "$steps.byte_tracker.tracked_detections",
        "reference_timestamp": "$inputs.video_start_time",
    }

    # When - parse the manifest
    manifest = BlockManifest.model_validate(manifest_data)

    # Then - should accept workflow parameter selector
    assert manifest.reference_timestamp == "$inputs.video_start_time"


def test_manifest_accepts_literal_float_for_reference_timestamp() -> None:
    """Test that BlockManifest still accepts literal float values for reference_timestamp."""
    # Given - a manifest with reference_timestamp as a literal float
    manifest_data = {
        "type": "roboflow_core/detection_event_log@v1",
        "name": "detection_event_log",
        "image": "$inputs.image",
        "detections": "$steps.byte_tracker.tracked_detections",
        "reference_timestamp": 1726570875.0,
    }

    # When - parse the manifest
    manifest = BlockManifest.model_validate(manifest_data)

    # Then - should accept literal float
    assert manifest.reference_timestamp == 1726570875.0


def test_complete_events_empty_when_no_stale_events() -> None:
    """Test that complete_events is empty when no events go stale."""
    # Given
    block = DetectionEventLogBlockV1()
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])

    # When
    result = block.run(
        image=image_data,
        detections=detections,
        frame_threshold=1,
        flush_interval=30,
        stale_frames=300,
    )

    # Then
    assert "complete_events" in result
    assert result["complete_events"] == {}


def test_complete_events_contains_logged_stale_events() -> None:
    """Test that complete_events contains logged events when they go stale."""
    # Given
    block = DetectionEventLogBlockV1()
    flush_interval = 5
    stale_frames = 3
    frame_threshold = 1  # Log immediately

    # Add detection on frame 1 - should be logged immediately
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=frame_threshold,
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
            frame_threshold=frame_threshold,
            flush_interval=flush_interval,
            stale_frames=stale_frames,
        )

    # Then - tracker 1 should be in complete_events
    assert "1" in result["complete_events"]
    complete_event = result["complete_events"]["1"]
    assert complete_event["tracker_id"] == 1
    assert complete_event["class_name"] == "dog"
    assert complete_event["first_seen_frame"] == 1
    assert complete_event["frame_count"] == 1


def test_complete_events_excludes_pending_stale_events() -> None:
    """Test that complete_events does NOT contain pending events that go stale."""
    # Given
    block = DetectionEventLogBlockV1()
    flush_interval = 5
    stale_frames = 3
    frame_threshold = 10  # High threshold - event won't be logged

    # Add detection on frame 1 - should stay pending (frame_count=1 < threshold=10)
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=frame_threshold,
        flush_interval=flush_interval,
        stale_frames=stale_frames,
    )

    # When - advance to frame 6 (flush_interval reached) without seeing tracker 1
    for frame in range(2, 7):
        image_data = create_workflow_image_data(frame_number=frame)
        detections = create_detections([2], ["cat"])  # Different tracker
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=frame_threshold,
            flush_interval=flush_interval,
            stale_frames=stale_frames,
        )

    # Then - tracker 1 should NOT be in complete_events (was pending, not logged)
    assert "1" not in result["complete_events"]
    # Tracker 1 should also be removed from event_log
    assert "1" not in result["event_log"]["logged"]
    assert "1" not in result["event_log"]["pending"]


def test_complete_events_has_correct_format() -> None:
    """Test that complete_events has the correct format with all expected fields."""
    # Given
    block = DetectionEventLogBlockV1()
    flush_interval = 5
    stale_frames = 3
    frame_threshold = 2
    reference_timestamp = 1000.0

    # Add detection for frames 1-2 to reach threshold
    for frame in range(1, 3):
        image_data = create_workflow_image_data(frame_number=frame)
        detections = create_detections([1], ["dog"])
        block.run(
            image=image_data,
            detections=detections,
            frame_threshold=frame_threshold,
            flush_interval=flush_interval,
            stale_frames=stale_frames,
            reference_timestamp=reference_timestamp,
        )

    # When - advance to trigger stale removal
    # Capture each result to find the one with complete_events
    flush_result = None
    for frame in range(3, 8):
        image_data = create_workflow_image_data(frame_number=frame)
        detections = create_detections([2], ["cat"])
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=frame_threshold,
            flush_interval=flush_interval,
            stale_frames=stale_frames,
            reference_timestamp=reference_timestamp,
        )
        # Capture the result when flush happens (complete_events not empty)
        if result["complete_events"]:
            flush_result = result

    # Then - complete_events should have all expected fields
    assert flush_result is not None, "Expected flush to occur with complete_events"
    assert "1" in flush_result["complete_events"]
    event = flush_result["complete_events"]["1"]
    assert "tracker_id" in event
    assert "class_name" in event
    assert "first_seen_frame" in event
    assert "last_seen_frame" in event
    assert "first_seen_relative" in event
    assert "last_seen_relative" in event
    assert "first_seen_timestamp" in event  # Has reference_timestamp
    assert "last_seen_timestamp" in event
    assert "frame_count" in event
    # Verify values
    assert event["tracker_id"] == 1
    assert event["class_name"] == "dog"
    assert event["first_seen_frame"] == 1
    assert event["last_seen_frame"] == 2
    assert event["frame_count"] == 2


def test_complete_events_only_appears_for_one_frame() -> None:
    """Test that complete_events only contains the event for the frame when it goes stale, not subsequent frames."""
    # Given
    block = DetectionEventLogBlockV1()
    flush_interval = 5
    stale_frames = 3
    frame_threshold = 1  # Log immediately

    # Add detection on frame 1 - should be logged immediately
    image_data = create_workflow_image_data(frame_number=1)
    detections = create_detections([1], ["dog"])
    block.run(
        image=image_data,
        detections=detections,
        frame_threshold=frame_threshold,
        flush_interval=flush_interval,
        stale_frames=stale_frames,
    )

    # Advance to frame 6 (flush_interval reached) - tracker 1 should go stale
    complete_events_frames = []
    for frame in range(2, 10):
        image_data = create_workflow_image_data(frame_number=frame)
        detections = create_detections([2], ["cat"])  # Different tracker
        result = block.run(
            image=image_data,
            detections=detections,
            frame_threshold=frame_threshold,
            flush_interval=flush_interval,
            stale_frames=stale_frames,
        )
        if "1" in result["complete_events"]:
            complete_events_frames.append(frame)

    # Then - tracker 1 should only appear in complete_events for exactly one frame
    assert len(complete_events_frames) == 1, (
        f"Expected complete_events to contain tracker 1 for exactly 1 frame, "
        f"but it appeared in frames: {complete_events_frames}"
    )
