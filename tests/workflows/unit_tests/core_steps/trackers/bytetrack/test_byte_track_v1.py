import datetime

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.trackers.bytetrack.v1 import (
    ByteTrackBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import VideoMetadata
from tests.workflows.unit_tests.core_steps.trackers.conftest import (
    FRAME1_XYXY,
    FRAME2_XYXY,
    FRAME3_XYXY,
    make_detections,
    make_metadata,
    wrap_with_workflow_image,
)


def test_byte_track_block() -> None:
    block = ByteTrackBlockV1()
    block.run(
        image=wrap_with_workflow_image(make_metadata(10)),
        detections=make_detections(FRAME1_XYXY),
        minimum_consecutive_frames=1,
    )
    frame2_result = block.run(
        image=wrap_with_workflow_image(make_metadata(11)),
        detections=make_detections(FRAME2_XYXY),
        minimum_consecutive_frames=1,
    )
    frame3_result = block.run(
        image=wrap_with_workflow_image(make_metadata(12)),
        detections=make_detections(FRAME3_XYXY),
        minimum_consecutive_frames=1,
    )

    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3
    assert len(frame3_ids) == 3
    assert frame3_ids == frame2_ids

    assert len(frame2_result["new_instances"]) == 3
    assert len(frame2_result["already_seen_instances"]) == 0
    assert len(frame3_result["new_instances"]) == 0
    assert len(frame3_result["already_seen_instances"]) == 3


def test_byte_track_missing_fps() -> None:
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
    block = ByteTrackBlockV1()

    result = block.run(
        image=wrap_with_workflow_image(metadata),
        detections=detections,
        minimum_consecutive_frames=1,
    )
    assert "tracked_detections" in result
    assert "new_instances" in result
    assert "already_seen_instances" in result


def test_byte_track_not_video_file() -> None:
    block = ByteTrackBlockV1()
    block.run(
        image=wrap_with_workflow_image(make_metadata(10, comes_from_video_file=False, timestamp_offset=0)),
        detections=make_detections(FRAME1_XYXY),
        minimum_consecutive_frames=1,
    )
    frame2_result = block.run(
        image=wrap_with_workflow_image(make_metadata(11, comes_from_video_file=False, timestamp_offset=1)),
        detections=make_detections(FRAME2_XYXY),
        minimum_consecutive_frames=1,
    )
    frame3_result = block.run(
        image=wrap_with_workflow_image(make_metadata(12, comes_from_video_file=False, timestamp_offset=2)),
        detections=make_detections(FRAME3_XYXY),
        minimum_consecutive_frames=1,
    )

    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3
    assert len(frame3_ids) == 3
    assert frame3_ids == frame2_ids