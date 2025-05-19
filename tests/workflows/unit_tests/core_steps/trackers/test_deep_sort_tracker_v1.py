import datetime
from typing import Any

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.trackers.deep_sort.v1 import (
    DeepSortTrackerBlockV1,
)
from inference.core.workflows.core_steps.trackers.base import InstanceCache
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


class _DummyReIDModel:
    def extract_features(self, image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        return np.ones((len(detections), 8), dtype=np.float32)


def _wrap(metadata: VideoMetadata) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def test_deep_sort_tracker(monkeypatch) -> None:
    monkeypatch.setattr(
        "inference.core.workflows.core_steps.trackers.base.BaseReIDTrackerBlock._get_reid_model",
        lambda self, model_name, device: _DummyReIDModel(),
    )

    frame1 = sv.Detections(
        xyxy=np.array(
            [[10, 10, 20, 20], [21, 10, 31, 20], [31, 10, 41, 20], [100, 100, 110, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
    )
    frame2 = sv.Detections(
        xyxy=np.array(
            [[12, 10, 22, 20], [23, 10, 33, 20], [33, 10, 43, 20], [110, 100, 120, 110]]
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9]),
        class_id=np.array([1, 1, 1, 1]),
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

    block = DeepSortTrackerBlockV1()

    res1 = block.run(
        image=_wrap(meta1),
        detections=frame1,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        embedding_model="dummy",
        appearance_threshold=0.7,
        appearance_weight=0.5,
        distance_metric="cosine",
        minimum_iou_threshold=0.3,
        minimum_consecutive_frames=1,
        instances_cache_size=16,
    )
    res2 = block.run(
        image=_wrap(meta2),
        detections=frame2,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        embedding_model="dummy",
        appearance_threshold=0.7,
        appearance_weight=0.5,
        distance_metric="cosine",
        minimum_iou_threshold=0.3,
        minimum_consecutive_frames=1,
        instances_cache_size=16,
    )

    assert res1["tracked_detections"].embedding.shape == (4, 8)
    assert len(set(res1["tracked_detections"].tracker_id.tolist())) == 4
    assert res1["tracked_detections"].tracker_id.tolist() == res1["new_instances"].tracker_id.tolist()
    assert len(res2["new_instances"]) <= 1
    assert res2["tracked_detections"].tracker_id.tolist()[:3] == res2["already_seen_instances"].tracker_id.tolist()[:3]


def test_instance_cache_basic() -> None:
    cache = InstanceCache(size=2)
    assert cache.record(0) is False
    assert cache.record(0) is True
    cache.record(1)
    cache.record(2)
    assert cache.record(0) is False
    assert cache.record(2) is True
