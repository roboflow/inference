import datetime

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.trackers.bytetrack.v1 import (
    ByteTrackBlockV1,
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


def test_byte_track_block() -> None:
    block = ByteTrackBlockV1()
    block.run(
        image=_wrap_with_workflow_image(_make_metadata(10)),
        detections=_detections(_FRAME1_XYXY),
        minimum_consecutive_frames=1,
    )
    frame2_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(11)),
        detections=_detections(_FRAME2_XYXY),
        minimum_consecutive_frames=1,
    )
    frame3_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(12)),
        detections=_detections(_FRAME3_XYXY),
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
        image=_wrap_with_workflow_image(metadata),
        detections=detections,
        minimum_consecutive_frames=1,
    )
    assert "tracked_detections" in result
    assert "new_instances" in result
    assert "already_seen_instances" in result


def test_byte_track_not_video_file() -> None:
    block = ByteTrackBlockV1()
    block.run(
        image=_wrap_with_workflow_image(_make_metadata(10, comes_from_video_file=False, timestamp_offset=0)),
        detections=_detections(_FRAME1_XYXY),
        minimum_consecutive_frames=1,
    )
    frame2_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(11, comes_from_video_file=False, timestamp_offset=1)),
        detections=_detections(_FRAME2_XYXY),
        minimum_consecutive_frames=1,
    )
    frame3_result = block.run(
        image=_wrap_with_workflow_image(_make_metadata(12, comes_from_video_file=False, timestamp_offset=2)),
        detections=_detections(_FRAME3_XYXY),
        minimum_consecutive_frames=1,
    )

    frame2_ids = set(frame2_result["tracked_detections"].tracker_id.tolist())
    frame3_ids = set(frame3_result["tracked_detections"].tracker_id.tolist())

    assert len(frame2_ids) == 3
    assert len(frame3_ids) == 3
    assert frame3_ids == frame2_ids
