"""Detection event log: thresholds, stale removal, times, classes and state."""

from datetime import datetime, timedelta

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics.detection_event_log import v1 as module
from roboflow_workflows.v2.blocks.analytics.detection_event_log.v1 import DetectionEventLogBlock
from roboflow_workflows.v2.blocks.tracking.byte_tracker.v3 import TrackerCapacityError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView

from inference_models.models.base.object_detection import Detections

START = datetime(2026, 10, 13, 12, 0, 0)


def _image(frame, source="a", *, seconds=None):
    metadata = VideoMetadata(
        video_identifier=source,
        frame_number=frame,
        frame_timestamp=START
        + timedelta(seconds=frame if seconds is None else seconds),
        fps=10.0,
    )
    image = ImageData.from_tensor(
        torch.zeros((3, 4, 4), dtype=torch.uint8), video_metadata=metadata
    )

    return image


def _rows(*tracker_ids, class_ids=None, rows_metadata=None):
    count = len(tracker_ids)
    detections = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]] * count).reshape(-1, 4),
        class_id=torch.tensor(class_ids or [0] * count, dtype=torch.int64),
        confidence=torch.full((count,), 0.9),
        image_metadata={"class_names": {0: "car", 1: "truck"}},
        bboxes_metadata=rows_metadata
        or [{"tracker_id": tracker_id} for tracker_id in tracker_ids]
        or None,
    )

    return detections


def _log(block, image, prediction, **kwargs):
    parameters = {"frame_threshold": 2, "flush_interval": 1, "stale_frames": 1}
    output = block.run(image=image, predictions=prediction, **{**parameters, **kwargs})

    return output


def _ids(mapping):
    return sorted(mapping, key=int)


def test_tracks_move_from_pending_to_logged_and_complete_once_stale():
    block = DetectionEventLogBlock()
    frames = [(7,), (7,), (), (), ()]
    outputs = [
        _log(block, _image(frame), _rows(*ids))
        for frame, ids in enumerate(frames, start=1)
    ]

    states = [
        (
            _ids(output["event_log"]["pending"]),
            _ids(output["event_log"]["logged"]),
            _ids(output["complete_events"]),
        )
        for output in outputs
    ]
    assert states == [
        (["7"], [], []),
        ([], ["7"], []),
        ([], ["7"], []),
        ([], [], ["7"]),
        ([], [], []),
    ]
    assert [output["total_logged"] for output in outputs] == [0, 1, 1, 0, 0]
    assert [output["total_pending"] for output in outputs] == [1, 0, 0, 0, 0]
    assert outputs[3]["complete_events"]["7"] == {
        "tracker_id": 7,
        "class_name": "car",
        "first_seen_frame": 1,
        "first_seen_timestamp": (START + timedelta(seconds=1)).timestamp(),
        "last_seen_frame": 2,
        "last_seen_timestamp": (START + timedelta(seconds=2)).timestamp(),
        "first_seen_relative": 0.0,
        "last_seen_relative": 1.0,
        "frame_count": 2,
    }


def test_stale_pending_events_are_dropped_without_completing():
    block = DetectionEventLogBlock()
    _log(block, _image(1), _rows(3))
    _log(block, _image(2), _rows())
    output = _log(block, _image(3), _rows())

    assert output["event_log"] == {"logged": {}, "pending": {}}
    assert output["complete_events"] == {}


def test_removal_runs_every_flush_interval_before_the_frame_is_added():
    block = DetectionEventLogBlock()
    outputs = [
        _log(block, _image(frame), _rows(*ids), flush_interval=3, stale_frames=1)
        for frame, ids in enumerate([(1,), (), (), (), (1,)], start=1)
    ]

    # Removal at frames 4 (last flush 1); track 1 was unseen for 3 frames.
    assert [_ids(o["event_log"]["pending"]) for o in outputs] == [
        ["1"],
        ["1"],
        ["1"],
        [],
        ["1"],
    ]
    assert outputs[4]["event_log"]["pending"]["1"]["first_seen_frame"] == 5


def test_frame_threshold_change_applies_to_the_current_log():
    block = DetectionEventLogBlock()
    _log(block, _image(1), _rows(1), stale_frames=100)
    _log(block, _image(2), _rows(1), stale_frames=100)
    output = _log(block, _image(3), _rows(), frame_threshold=3, stale_frames=100)

    assert _ids(output["event_log"]["pending"]) == ["1"]


def test_rows_without_tracker_id_are_ignored_and_duplicates_count_once():
    block = DetectionEventLogBlock()
    prediction = _rows(
        1, 1, None, rows_metadata=[{"tracker_id": 1}, {"tracker_id": 1}, {}]
    )
    output = _log(block, _image(1), prediction, frame_threshold=5)

    assert list(output["event_log"]["pending"]) == ["1"]
    assert output["event_log"]["pending"]["1"]["frame_count"] == 1
    assert output["detections"] is prediction


def test_class_names_come_from_the_row_the_map_or_the_class_id():
    block = DetectionEventLogBlock()
    prediction = _rows(
        1,
        2,
        3,
        class_ids=[0, 1, 9],
        rows_metadata=[
            {"tracker_id": 1, "class": "bus"},
            {"tracker_id": 2},
            {"tracker_id": 3},
        ],
    )
    first = _log(block, _image(1), prediction, frame_threshold=5)
    # A later frame with another class keeps the first-seen name.
    later = _log(block, _image(2), _rows(2, class_ids=[0]), frame_threshold=5)

    names = {
        key: event["class_name"] for key, event in first["event_log"]["pending"].items()
    }
    assert names == {"1": "bus", "2": "truck", "3": "class_9"}
    assert later["event_log"]["pending"]["2"]["class_name"] == "truck"


def test_relative_time_counts_from_the_first_frame_of_the_source():
    block = DetectionEventLogBlock()
    _log(block, _image(1, seconds=100.0), _rows(1), frame_threshold=5)
    output = _log(block, _image(2, seconds=102.5), _rows(1), frame_threshold=5)

    event = output["event_log"]["pending"]["1"]
    assert event["first_seen_relative"] == 0.0
    assert event["last_seen_relative"] == 2.5
    assert (
        event["last_seen_timestamp"] == (START + timedelta(seconds=102.5)).timestamp()
    )


def test_images_without_video_metadata_use_fallback_fps_and_wall_clock(monkeypatch):
    monkeypatch.setattr(module.time, "time", lambda: 1234.5)
    block = DetectionEventLogBlock()
    image = ImageData.from_tensor(torch.zeros((3, 4, 4), dtype=torch.uint8))
    for _ in range(3):
        output = _log(block, image, _rows(1), frame_threshold=5, fallback_fps=2.0)

    event = output["event_log"]["pending"]["1"]
    assert list(block._sources) == [module.DEFAULT_SOURCE]
    assert event["last_seen_relative"] == 1.0
    assert event["last_seen_timestamp"] == 1234.5


def test_interleaved_sources_keep_separate_logs_with_equal_ids():
    block = DetectionEventLogBlock()
    _log(block, _image(1, "a"), _rows(1))
    _log(block, _image(1, "b"), _rows(1))
    a = _log(block, _image(2, "a"), _rows(1))
    b = _log(block, _image(2, "b"), _rows())

    assert _ids(a["event_log"]["logged"]) == ["1"]
    assert _ids(b["event_log"]["pending"]) == ["1"]


def test_capacity_rejects_a_new_source_before_creating_state():
    block = DetectionEventLogBlock()
    _log(block, _image(1, "a"), _rows(1), max_sources=1)

    with pytest.raises(TrackerCapacityError, match="max_sources=1"):
        _log(block, _image(1, "b"), _rows(1), max_sources=1)
    assert list(block._sources) == ["a"]


def test_other_context_failures_are_not_taken_for_a_plain_call(monkeypatch):
    def broken(self):
        raise RuntimeError("context broke")

    monkeypatch.setattr(DetectionEventLogBlock, "execution_context", property(broken))

    with pytest.raises(RuntimeError, match="context broke"):
        _log(DetectionEventLogBlock(), _image(1, ""), _rows(1))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"frame_threshold": 0}, "frame_threshold"),
        ({"flush_interval": 1.5}, "flush_interval"),
        ({"stale_frames": True}, "stale_frames"),
        ({"fallback_fps": 0.0}, "fallback_fps"),
        ({"fallback_fps": float("inf")}, "fallback_fps"),
    ],
)
def test_invalid_direct_calls_fail_before_state_changes(kwargs, match):
    block = DetectionEventLogBlock()

    with pytest.raises(ValueError, match=match):
        _log(block, _image(1), _rows(1), **kwargs)
    assert block._sources == {}


def test_view_stays_lazy_and_is_returned_as_given(monkeypatch):
    def forbidden(self):
        raise AssertionError("full_res must not run")

    monkeypatch.setattr(SegmentationView, "full_res", forbidden)
    view = SegmentationView(
        detections=_rows(4),
        scores=torch.zeros((1, 4, 4)),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((4, 4)),
    )

    output = _log(DetectionEventLogBlock(), _image(1), view)

    assert output["detections"] is view
    assert view.materialization_count == 0
    assert _ids(output["event_log"]["pending"]) == ["4"]
