import datetime

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.trackers.sort.v1 import SortTrackerBlockV1
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


def test_sort_tracker_v1_basic() -> None:
    frame1_dets = sv.Detections(
        xyxy=np.array(
            [[10, 10, 20, 20], [21, 10, 31, 20], [31, 10, 41, 20], [100, 100, 110, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
    )
    frame2_dets = sv.Detections(
        xyxy=np.array(
            [[12, 10, 22, 20], [23, 10, 33, 20], [33, 10, 43, 20], [110, 100, 120, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
    )
    frame3_dets = sv.Detections(
        xyxy=np.array([[14, 10, 24, 20], [25, 10, 35, 20], [35, 10, 45, 20]]),
        confidence=np.array([0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1]),
    )

    meta1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(tz=datetime.timezone.utc),
        comes_from_video_file=True,
    )
    meta2 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=11,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570876).astimezone(tz=datetime.timezone.utc),
        comes_from_video_file=True,
    )
    meta3 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=12,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570877).astimezone(tz=datetime.timezone.utc),
        comes_from_video_file=True,
    )

    block = SortTrackerBlockV1()

    res1 = block.run(
        image=_wrap_with_workflow_image(meta1),
        detections=frame1_dets,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_consecutive_frames=1,
        minimum_iou_threshold=0.3,
        instances_cache_size=16,
    )
    res2 = block.run(
        image=_wrap_with_workflow_image(meta2),
        detections=frame2_dets,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_consecutive_frames=1,
        minimum_iou_threshold=0.3,
        instances_cache_size=16,
    )
    res3 = block.run(
        image=_wrap_with_workflow_image(meta3),
        detections=frame3_dets,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_consecutive_frames=1,
        minimum_iou_threshold=0.3,
        instances_cache_size=16,
    )

    assert len(set(res1["tracked_detections"].tracker_id.tolist())) == 4
    assert res1["tracked_detections"].tracker_id.tolist()[:3] == res2["tracked_detections"].tracker_id.tolist()[:3]
    assert res1["tracked_detections"].tracker_id.tolist()[:3] == res3["tracked_detections"].tracker_id.tolist()[:3]
    assert res1["tracked_detections"].tracker_id.tolist() == res1["new_instances"].tracker_id.tolist()
    assert len(res1["already_seen_instances"]) == 0
    assert len(res2["new_instances"]) <= 1
    assert res2["tracked_detections"].tracker_id.tolist()[:3] == res2["already_seen_instances"].tracker_id.tolist()[:3]
    assert len(res3["new_instances"]) == 0
    assert res3["tracked_detections"].tracker_id.tolist()[:3] == res3["already_seen_instances"].tracker_id.tolist()[:3]
