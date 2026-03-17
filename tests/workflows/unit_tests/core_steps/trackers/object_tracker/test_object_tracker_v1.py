import datetime

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.trackers._utils import InstanceCache
from inference.core.workflows.core_steps.trackers.object_tracker.v1 import (
    ObjectTrackerBlockV1,
)
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


def _make_metadata(frame_number: int, fps: float = 1, comes_from_video_file: bool = True, timestamp_offset: int = 0) -> VideoMetadata:
    return VideoMetadata(
        video_identifier="vid_1",
        frame_number=frame_number,
        fps=fps,
        frame_timestamp=datetime.datetime.fromtimestamp(
            1726570875 + timestamp_offset
        ).astimezone(tz=datetime.timezone.utc),
        comes_from_video_file=comes_from_video_file,
    )


# ---------------------------------------------------------------------------
# Shared detection fixtures
# ---------------------------------------------------------------------------

_FRAME1_XYXY = np.array(
    [[10, 10, 20, 20], [21, 10, 31, 20], [31, 10, 41, 20], [100, 100, 110, 110]]
)
_FRAME2_XYXY = np.array(
    [[12, 10, 22, 20], [23, 10, 33, 20], [33, 10, 43, 20], [110, 100, 120, 110]]
)
_FRAME3_XYXY = np.array([[14, 10, 24, 20], [25, 10, 35, 20], [35, 10, 45, 20]])


def _detections(xyxy: np.ndarray, confidence: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full(len(xyxy), confidence),
        class_id=np.ones(len(xyxy), dtype=int),
    )


# ---------------------------------------------------------------------------
# Helper: run 3-frame sequence and return (frame2_result, frame3_result)
# ---------------------------------------------------------------------------

def _run_three_frames(tracker_type: str, **extra_kwargs):
    block = ObjectTrackerBlockV1()
    block.run(
        image=_wrap_with_workflow_image(_make_metadata(10)),
        detections=_detections(_FRAME1_XYXY),
        tracker_type=tracker_type,
        minimum_consecutive_frames=1,
        **extra_kwargs,
    )
    frame2_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(11)),
        detections=_detections(_FRAME2_XYXY),
        tracker_type=tracker_type,
        minimum_consecutive_frames=1,
        **extra_kwargs,
    )
    frame3_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(12)),
        detections=_detections(_FRAME3_XYXY),
        tracker_type=tracker_type,
        minimum_consecutive_frames=1,
        **extra_kwargs,
    )
    return frame2_result, frame3_result


# ---------------------------------------------------------------------------
# Per-algorithm tracker tests
# ---------------------------------------------------------------------------

def test_object_tracker_bytetrack() -> None:
    # given / when
    frame2_result, frame3_result = _run_three_frames("bytetrack")

    # then
    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3, "Expected 3 unique tracking ids in frame 2"
    assert len(frame3_ids) == 3, "Expected 3 unique tracking ids in frame 3"
    assert frame3_ids == frame2_ids, "Expected the same 3 IDs in frame 3 as frame 2"

    assert len(frame2_result["new_instances"]) == 3, "Expected 3 new instances in frame 2"
    assert len(frame2_result["already_seen_instances"]) == 0, "Expected no already-seen in frame 2"
    assert len(frame3_result["new_instances"]) == 0, "Expected no new instances in frame 3"
    assert len(frame3_result["already_seen_instances"]) == 3, "Expected 3 already-seen in frame 3"


def test_object_tracker_sort() -> None:
    # given / when
    frame2_result, frame3_result = _run_three_frames("sort")

    # then
    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3, "Expected 3 unique tracking ids in frame 2"
    assert len(frame3_ids) == 3, "Expected 3 unique tracking ids in frame 3"
    assert frame3_ids == frame2_ids, "Expected the same 3 IDs in frame 3 as frame 2"

    assert len(frame2_result["new_instances"]) == 3, "Expected 3 new instances in frame 2"
    assert len(frame2_result["already_seen_instances"]) == 0, "Expected no already-seen in frame 2"
    assert len(frame3_result["new_instances"]) == 0, "Expected no new instances in frame 3"
    assert len(frame3_result["already_seen_instances"]) == 3, "Expected 3 already-seen in frame 3"


def test_object_tracker_ocsort() -> None:
    # given / when — use high_conf_det_threshold=0.5 so 0.9-confidence detections pass
    frame2_result, frame3_result = _run_three_frames(
        "ocsort", high_conf_det_threshold=0.5
    )

    # then
    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3, "Expected 3 unique tracking ids in frame 2"
    assert len(frame3_ids) == 3, "Expected 3 unique tracking ids in frame 3"
    assert frame3_ids == frame2_ids, "Expected the same 3 IDs in frame 3 as frame 2"

    assert len(frame2_result["new_instances"]) == 3, "Expected 3 new instances in frame 2"
    assert len(frame2_result["already_seen_instances"]) == 0, "Expected no already-seen in frame 2"
    assert len(frame3_result["new_instances"]) == 0, "Expected no new instances in frame 3"
    assert len(frame3_result["already_seen_instances"]) == 3, "Expected 3 already-seen in frame 3"


def test_object_tracker_independent_state_per_algorithm() -> None:
    # given — same video_id, two different tracker_type values on the same block
    block = ObjectTrackerBlockV1()

    for tracker_type in ("bytetrack", "sort"):
        block.run(
            image=_wrap_with_workflow_image(_make_metadata(10)),
            detections=_detections(_FRAME1_XYXY),
            tracker_type=tracker_type,
            minimum_consecutive_frames=1,
        )

    bytetrack_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(11)),
        detections=_detections(_FRAME2_XYXY),
        tracker_type="bytetrack",
        minimum_consecutive_frames=1,
    )
    sort_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(11)),
        detections=_detections(_FRAME2_XYXY),
        tracker_type="sort",
        minimum_consecutive_frames=1,
    )

    # then — both should have confirmed IDs (state is independent)
    assert len(bytetrack_result["tracked_detections"]) > 0, "ByteTrack should have confirmed tracks"
    assert len(sort_result["tracked_detections"]) > 0, "SORT should have confirmed tracks"


def test_object_tracker_not_video_file() -> None:
    # given — comes_from_video_file=False with varying timestamps
    block = ObjectTrackerBlockV1()
    block.run(
        image=_wrap_with_workflow_image(_make_metadata(10, comes_from_video_file=False, timestamp_offset=0)),
        detections=_detections(_FRAME1_XYXY),
        tracker_type="bytetrack",
        minimum_consecutive_frames=1,
    )
    frame2_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(11, comes_from_video_file=False, timestamp_offset=1)),
        detections=_detections(_FRAME2_XYXY),
        tracker_type="bytetrack",
        minimum_consecutive_frames=1,
    )
    frame3_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(12, comes_from_video_file=False, timestamp_offset=2)),
        detections=_detections(_FRAME3_XYXY),
        tracker_type="bytetrack",
        minimum_consecutive_frames=1,
    )

    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3, "Expected 3 unique tracking ids in frame 2"
    assert len(frame3_ids) == 3, "Expected 3 unique tracking ids in frame 3"
    assert frame3_ids == frame2_ids, "Expected the same 3 IDs in frame 3 as frame 2"


def test_object_tracker_missing_fps() -> None:
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
    block = ObjectTrackerBlockV1()

    # when / then — must not raise
    result = block.run(
        image=_wrap_with_workflow_image(metadata),
        detections=detections,
        tracker_type="bytetrack",
        minimum_consecutive_frames=1,
    )
    assert "tracked_detections" in result
    assert "new_instances" in result
    assert "already_seen_instances" in result


# ---------------------------------------------------------------------------
# InstanceCache tests
# ---------------------------------------------------------------------------

def test_instance_cache_when_new_tracker_id_is_added() -> None:
    cache = InstanceCache(size=10)
    result = cache.record_instance(tracker_id=0)
    assert result is False, "Expected id not to be seen in cache previously"


def test_instance_cache_when_already_seen_tracker_id_is_added() -> None:
    cache = InstanceCache(size=10)
    _ = cache.record_instance(tracker_id=0)
    result = cache.record_instance(tracker_id=0)
    assert result is True, "Expected id to be seen in cache previously"


def test_instance_cache_flushing_on_overflow() -> None:
    cache = InstanceCache(size=10)
    for i in range(11):
        _ = cache.record_instance(tracker_id=i)
    result_for_zero = cache.record_instance(tracker_id=0)
    result_for_two = cache.record_instance(tracker_id=2)
    assert result_for_zero is False, "Expected id=0 not to be in cache after flush"
    assert result_for_two is True, "Expected id=2 to still be in cache"
