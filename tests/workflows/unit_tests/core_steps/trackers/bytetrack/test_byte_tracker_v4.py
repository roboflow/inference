import datetime

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.trackers._utils import InstanceCache
from inference.core.workflows.core_steps.trackers.bytetrack.v4 import ByteTrackerBlockV4
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def _wrap_with_workflow_image(metadata: VideoMetadata) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def test_byte_tracker() -> None:
    # given
    # With minimum_consecutive_frames=1, tracks are spawned on frame 1 with tracker_id=-1
    # and confirmed (IDs assigned) on frame 2 when matched for the first time.
    # So: frame 1 → 0 confirmed, frame 2 → 4 confirmed (new), frame 3 → 3 (already seen).
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [[10, 10, 20, 20], [21, 10, 31, 20], [31, 10, 41, 20], [100, 100, 110, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [[12, 10, 22, 20], [23, 10, 33, 20], [33, 10, 43, 20], [110, 100, 120, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
    )
    frame3_detections = sv.Detections(
        xyxy=np.array([[14, 10, 24, 20], [25, 10, 35, 20], [35, 10, 45, 20]]),
        confidence=np.array([0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    frame3_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=12,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    block = ByteTrackerBlockV4()

    # when
    block.run(
        image=_wrap_with_workflow_image(frame1_metadata),
        detections=frame1_detections,
        minimum_consecutive_frames=1,
    )
    frame2_result = block.run(
        image=_wrap_with_workflow_image(frame2_metadata),
        detections=frame2_detections,
        minimum_consecutive_frames=1,
    )
    frame3_result = block.run(
        image=_wrap_with_workflow_image(frame3_metadata),
        detections=frame3_detections,
        minimum_consecutive_frames=1,
    )

    # then
    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    # The 4th detection in frame 2 ([110,100,120,110]) has 0 IoU with the spawned
    # track from frame 1 ([100,100,110,110]) so it spawns a new unconfirmed track
    # (tracker_id=-1, filtered out). Only 3 tracks are confirmed in frame 2.
    assert len(frame2_ids) == 3, "Expected 3 unique tracking ids in frame 2"
    assert len(frame3_ids) == 3, "Expected 3 unique tracking ids in frame 3"
    assert frame3_ids == frame2_ids, "Expected the same 3 IDs in frame 3 as frame 2"

    assert (
        len(frame2_result["new_instances"]) == 3
    ), "Expected 3 new instances in frame 2"
    assert (
        len(frame2_result["already_seen_instances"]) == 0
    ), "Expected no already-seen instances in frame 2"
    assert (
        len(frame3_result["new_instances"]) == 0
    ), "Expected no new instances in frame 3"
    assert (
        len(frame3_result["already_seen_instances"]) == 3
    ), "Expected all 3 detections to be already-seen in frame 3"


def test_byte_tracker_not_video_file() -> None:
    # given
    frame1_detections = sv.Detections(
        xyxy=np.array(
            [[10, 10, 20, 20], [21, 10, 31, 20], [31, 10, 41, 20], [100, 100, 110, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
    )
    frame2_detections = sv.Detections(
        xyxy=np.array(
            [[12, 10, 22, 20], [23, 10, 33, 20], [33, 10, 43, 20], [110, 100, 120, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
    )
    frame3_detections = sv.Detections(
        xyxy=np.array([[14, 10, 24, 20], [25, 10, 35, 20], [35, 10, 45, 20]]),
        confidence=np.array([0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1]),
    )
    frame1_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=False,
    )
    frame2_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570876).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=False,
    )
    frame3_metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=12,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570877).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=False,
    )
    block = ByteTrackerBlockV4()

    # when
    block.run(
        image=_wrap_with_workflow_image(frame1_metadata),
        detections=frame1_detections,
        minimum_consecutive_frames=1,
    )
    frame2_result = block.run(
        image=_wrap_with_workflow_image(frame2_metadata),
        detections=frame2_detections,
        minimum_consecutive_frames=1,
    )
    frame3_result = block.run(
        image=_wrap_with_workflow_image(frame3_metadata),
        detections=frame3_detections,
        minimum_consecutive_frames=1,
    )

    # then
    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3, "Expected 3 unique tracking ids in frame 2"
    assert len(frame3_ids) == 3, "Expected 3 unique tracking ids in frame 3"
    assert frame3_ids == frame2_ids, "Expected the same 3 IDs in frame 3 as frame 2"


def test_byte_tracker_missing_fps() -> None:
    # given — fps=None should not raise; block logs a warning and uses fps=0
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20]]),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=1,
        fps=None,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    block = ByteTrackerBlockV4()

    # when / then — must not raise
    result = block.run(
        image=_wrap_with_workflow_image(metadata),
        detections=detections,
        minimum_consecutive_frames=1,
    )
    assert "tracked_detections" in result
    assert "new_instances" in result
    assert "already_seen_instances" in result


def test_instance_cache_when_new_tracker_id_is_added() -> None:
    # given
    cache = InstanceCache(size=10)

    # when
    result = cache.record_instance(tracker_id=0)

    # then
    assert result is False, "Expected id not to be seen in cache previously"


def test_instance_cache_when_already_seen_tracker_id_is_added() -> None:
    # given
    cache = InstanceCache(size=10)

    # when
    _ = cache.record_instance(tracker_id=0)
    result = cache.record_instance(tracker_id=0)

    # then
    assert result is True, "Expected id to be seen in cache previously"


def test_instance_cache_flushing_on_overflow() -> None:
    # given
    cache = InstanceCache(size=10)

    # when
    for i in range(11):
        _ = cache.record_instance(tracker_id=i)
    result_for_zero = cache.record_instance(tracker_id=0)
    result_for_two = cache.record_instance(tracker_id=2)

    # then
    assert (
        result_for_zero is False
    ), "Expected id=0 not to be in cache after flush"
    assert (
        result_for_two is True
    ), "Expected id=2 to still be in cache as first non-flushed id"
