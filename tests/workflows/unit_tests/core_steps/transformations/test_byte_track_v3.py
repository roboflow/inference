import datetime

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.transformations.byte_tracker.v3 import (
    ByteTrackerBlockV3,
    InstanceCache,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_byte_tracker() -> None:
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
    byte_tracker_block = ByteTrackerBlockV3()

    # when
    frame1_result = byte_tracker_block.run(
        image=_wrap_with_workflow_image(frame1_metadata),
        detections=frame1_detections,
    )
    frame2_result = byte_tracker_block.run(
        image=_wrap_with_workflow_image(frame2_metadata),
        detections=frame2_detections,
    )
    frame3_result = byte_tracker_block.run(
        image=_wrap_with_workflow_image(frame3_metadata),
        detections=frame3_detections,
    )

    # then
    assert (
        len(set(frame1_result["tracked_detections"].tracker_id.tolist())) == 4
    ), "Expected 4 unique tracking ids"
    assert (
        frame1_result["tracked_detections"].tracker_id.tolist()[:3]
        == frame2_result["tracked_detections"].tracker_id.tolist()
    ), "Expected the same 3 first objects in second frame"
    assert (
        frame1_result["tracked_detections"].tracker_id.tolist()[:3]
        == frame3_result["tracked_detections"].tracker_id.tolist()
    ), "Expected the same 3 first objects in third frame"
    assert (
        frame1_result["tracked_detections"].tracker_id.tolist()
        == frame1_result["new_instances"].tracker_id.tolist()
    ), "Expected all detections to fall into new instances category for frame 1"
    assert (
        len(frame1_result["already_seen_instances"]) == 0
    ), "Expected no instances marked as already seen"
    assert (
        len(frame2_result["new_instances"]) == 0
    ), "Expected no instances marked as new for frame 2"
    assert (
        frame2_result["tracked_detections"].tracker_id.tolist()
        == frame2_result["already_seen_instances"].tracker_id.tolist()
    ), "Expected all detections to fall into already seen category for frame 2"
    assert (
        len(frame3_result["new_instances"]) == 0
    ), "Expected no instances marked as new for frame 3"
    assert (
        frame3_result["tracked_detections"].tracker_id.tolist()
        == frame3_result["already_seen_instances"].tracker_id.tolist()
    ), "Expected all detections to fall into already seen category for frame 3"


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
    byte_tracker_block = ByteTrackerBlockV3()

    # when
    frame1_result = byte_tracker_block.run(
        image=_wrap_with_workflow_image(frame1_metadata),
        detections=frame1_detections,
    )
    frame2_result = byte_tracker_block.run(
        image=_wrap_with_workflow_image(frame2_metadata),
        detections=frame2_detections,
    )
    frame3_result = byte_tracker_block.run(
        image=_wrap_with_workflow_image(frame3_metadata),
        detections=frame3_detections,
    )

    # then
    assert (
        len(set(frame1_result["tracked_detections"].tracker_id.tolist())) == 4
    ), "Expected 4 unique tracking ids"
    assert (
        frame1_result["tracked_detections"].tracker_id.tolist()[:3]
        == frame2_result["tracked_detections"].tracker_id.tolist()
    ), "Expected the same 3 first objects in second frame"
    assert (
        frame1_result["tracked_detections"].tracker_id.tolist()[:3]
        == frame3_result["tracked_detections"].tracker_id.tolist()
    ), "Expected the same 3 first objects in third frame"
    assert (
        frame1_result["tracked_detections"].tracker_id.tolist()
        == frame1_result["new_instances"].tracker_id.tolist()
    ), "Expected all detections to fall into new instances category for frame 1"
    assert (
        len(frame1_result["already_seen_instances"]) == 0
    ), "Expected no instances marked as already seen"
    assert (
        len(frame2_result["new_instances"]) == 0
    ), "Expected no instances marked as new for frame 2"
    assert (
        frame2_result["tracked_detections"].tracker_id.tolist()
        == frame2_result["already_seen_instances"].tracker_id.tolist()
    ), "Expected all detections to fall into already seen category for frame 2"
    assert (
        len(frame3_result["new_instances"]) == 0
    ), "Expected no instances marked as new for frame 3"
    assert (
        frame3_result["tracked_detections"].tracker_id.tolist()
        == frame3_result["already_seen_instances"].tracker_id.tolist()
    ), "Expected all detections to fall into already seen category for frame 3"


def _wrap_with_workflow_image(metadata: VideoMetadata) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


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
    result_for_one = cache.record_instance(tracker_id=2)

    # then
    assert (
        result_for_zero is False
    ), "Expected id=0 not to be seen in cache previously, as the flush should happen"
    assert (
        result_for_one is True
    ), "Expected id=2 to still be in cache - as this is supposed to be the first non flushed id"
