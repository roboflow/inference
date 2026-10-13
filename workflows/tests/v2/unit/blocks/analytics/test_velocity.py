"""Velocity: time units, smoothing, unit conversion, gaps, state and carriers."""

from datetime import datetime, timedelta

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics import velocity as velocity_module
from roboflow_workflows.v2.blocks.analytics.velocity import VelocityBlock
from roboflow_workflows.v2.blocks.tracking.base import TrackerCapacityError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

START = datetime(2026, 10, 13, 12, 0, 0)


def _image(frame, source="a", *, fps=10.0, from_file=True, seconds=None):
    metadata = VideoMetadata(
        video_identifier=source,
        frame_number=frame,
        frame_timestamp=START
        + timedelta(seconds=frame if seconds is None else seconds),
        fps=fps,
        comes_from_video_file=from_file,
    )
    image = ImageData.from_tensor(
        torch.zeros((3, 40, 60), dtype=torch.uint8), video_metadata=metadata
    )

    return image


def _rows(*rows):
    """Detections from (tracker_id, centre_x, centre_y) tuples; 10 px boxes."""
    detections = Detections(
        xyxy=torch.tensor(
            [[x - 5.0, y - 5.0, x + 5.0, y + 5.0] for _, x, y in rows]
        ).reshape(-1, 4),
        class_id=torch.zeros(len(rows), dtype=torch.int64),
        confidence=torch.full((len(rows),), 0.9),
        image_metadata={"class_names": {0: "car"}},
        bboxes_metadata=[{"tracker_id": row[0], "custom": row[0]} for row in rows]
        or None,
    )

    return detections


def _measure(block, image, prediction, *, alpha=0.5, ppm=1.0, **kwargs):
    output = block.run(
        image=image,
        predictions=prediction,
        smoothing_alpha=alpha,
        pixels_per_meter=ppm,
        **kwargs,
    )["velocity_detections"]

    return output


def _values(output, key):
    values = [row[key] for row in output.bboxes_metadata]

    return values


def test_video_file_time_is_frame_number_over_fps_and_outputs_are_m_per_s():
    block = VelocityBlock()
    outputs = [
        _measure(block, _image(frame), _rows((1, 3.0 * frame, 50.0)), ppm=10.0)
        for frame in range(1, 6)
    ]

    # 3 px per 0.1 s = 30 px/s = 3 m/s at 10 px/m; smoothed approaches it as
    # 1 - (1 - a) ** (k - 1).
    for k, output in enumerate(outputs, start=1):
        row = output.bboxes_metadata[0]
        expected = 3.0 if k > 1 else 0.0
        assert row["velocity"] == pytest.approx([expected, 0.0])
        assert row["speed"] == pytest.approx(expected)
        smoothed = 3.0 * (1 - 0.5 ** (k - 1))
        assert row["smoothed_velocity"] == pytest.approx([smoothed, 0.0])
        assert row["smoothed_speed"] == pytest.approx(smoothed)


def test_stream_time_is_the_frame_timestamp():
    block = VelocityBlock()
    _measure(block, _image(1, from_file=False, seconds=0.0), _rows((1, 0.0, 0.0)))
    output = _measure(
        block, _image(2, from_file=False, seconds=0.5), _rows((1, 3.0, 4.0))
    )

    # 5 px in 0.5 s, ignoring frame numbers and fps.
    assert _values(output, "velocity") == [pytest.approx([6.0, 8.0])]
    assert _values(output, "speed") == [pytest.approx(10.0)]


@pytest.mark.parametrize("fps", [None, 0.0, -5.0, float("nan")])
def test_video_file_without_usable_fps_falls_back_to_timestamps(fps):
    block = VelocityBlock()
    _measure(block, _image(1, fps=fps, seconds=0.0), _rows((1, 0.0, 0.0)))
    output = _measure(block, _image(2, fps=fps, seconds=2.0), _rows((1, 4.0, 0.0)))

    assert _values(output, "velocity") == [pytest.approx([2.0, 0.0])]


def test_unchanged_time_reports_zero_and_decays_the_smoothed_value():
    block = VelocityBlock()
    _measure(block, _image(1), _rows((1, 0.0, 0.0)))
    _measure(block, _image(2), _rows((1, 2.0, 0.0)))
    repeated = _measure(block, _image(2), _rows((1, 9.0, 0.0)))

    assert _values(repeated, "velocity") == [pytest.approx([0.0, 0.0])]
    # 20 px/s smoothed to 10, then (1 - a) * 10.
    assert _values(repeated, "smoothed_velocity") == [pytest.approx([5.0, 0.0])]


def test_a_returning_track_is_measured_across_its_gap():
    block = VelocityBlock()
    _measure(block, _image(1), _rows((1, 0.0, 0.0), (2, 0.0, 20.0)))
    _measure(block, _image(2), _rows((2, 1.0, 20.0)))
    _measure(block, _image(3), _rows((2, 2.0, 20.0)))
    returned = _measure(block, _image(4), _rows((1, 6.0, 0.0)))

    # 6 px over 3 frames of 0.1 s.
    assert _values(returned, "velocity") == [pytest.approx([20.0, 0.0])]


def test_parameter_changes_apply_at_once_and_state_keeps_px_per_s():
    block = VelocityBlock()
    _measure(block, _image(1), _rows((1, 0.0, 0.0)), ppm=1.0)
    _measure(block, _image(2), _rows((1, 3.0, 4.0)), ppm=1.0)
    output = _measure(block, _image(3), _rows((1, 6.0, 8.0)), ppm=5.0, alpha=1.0)

    assert _values(output, "velocity") == [pytest.approx([6.0, 8.0])]
    assert _values(output, "smoothed_speed") == [pytest.approx(10.0)]


def test_interleaved_sources_have_separate_tracks_with_equal_ids():
    block = VelocityBlock()
    for frame in (1, 2):
        a = _measure(block, _image(frame, "a"), _rows((1, 1.0 * frame, 0.0)))
        b = _measure(block, _image(frame, "b"), _rows((1, 0.0, -4.0 * frame)))

    assert _values(a, "velocity") == [pytest.approx([10.0, 0.0])]
    assert _values(b, "velocity") == [pytest.approx([0.0, -40.0])]


def test_capacity_rejects_a_new_source_before_creating_state():
    block = VelocityBlock()
    _measure(block, _image(1, "a"), _rows((1, 0.0, 0.0)), max_sources=1)

    with pytest.raises(TrackerCapacityError, match="max_sources=1"):
        _measure(block, _image(1, "b"), _rows((1, 0.0, 0.0)), max_sources=1)
    assert list(block._sources) == ["a"]


def test_least_recently_seen_tracks_beyond_the_cap_are_forgotten(monkeypatch):
    monkeypatch.setattr(velocity_module, "MAX_TRACKS_PER_SOURCE", 2)
    block = VelocityBlock()
    _measure(block, _image(1), _rows((1, 0.0, 0.0)))
    _measure(block, _image(2), _rows((2, 0.0, 0.0)))
    _measure(block, _image(3), _rows((3, 0.0, 0.0)))
    kept = list(block._sources["a"].rows)
    returned = _measure(block, _image(4), _rows((1, 5.0, 0.0), (2, 5.0, 0.0)))

    # Track 1 was dropped at frame 3 and starts again; track 2 (frame 2) moved
    # 5 px in 0.2 s.
    assert kept == [2, 3]
    assert _values(returned, "velocity") == [
        pytest.approx([0.0, 0.0]),
        pytest.approx([25.0, 0.0]),
    ]


def test_a_frame_above_the_cap_measures_every_row_and_keeps_its_last(monkeypatch):
    monkeypatch.setattr(velocity_module, "MAX_TRACKS_PER_SOURCE", 2)
    block = VelocityBlock()
    _measure(block, _image(1), _rows((9, 0.0, 0.0), (1, 0.0, 10.0)))
    crowded = _measure(
        block, _image(2), _rows((1, 1.0, 10.0), (2, 1.0, 20.0), (3, 1.0, 30.0))
    )
    state = block._sources["a"]

    # All three rows are measured; only the last two (2, 3) are remembered,
    # each aligned with its own centre.
    assert _values(crowded, "custom") == [1, 2, 3]
    assert _values(crowded, "velocity") == [
        pytest.approx([10.0, 0.0]),
        pytest.approx([0.0, 0.0]),
        pytest.approx([0.0, 0.0]),
    ]
    assert list(state.rows.items()) == [(2, 0), (3, 1)]
    assert list(state.times) == [2, 3]
    assert state.positions.tolist() == [[1.0, 20.0], [1.0, 30.0]]
    assert tuple(state.smoothed.shape) == (2, 2)

    revisit = _measure(
        block, _image(3), _rows((9, 3.0, 0.0), (1, 3.0, 10.0), (2, 3.0, 20.0))
    )

    # Track 9 was the oldest and track 1 was measured but not kept: both start
    # again. Track 2 moved 2 px in 0.1 s.
    assert _values(revisit, "velocity") == [
        pytest.approx([0.0, 0.0]),
        pytest.approx([0.0, 0.0]),
        pytest.approx([20.0, 0.0]),
    ]
    assert _values(revisit, "smoothed_velocity")[2] == pytest.approx([10.0, 0.0])


def test_one_frame_with_more_tracks_than_the_real_cap_keeps_the_cap():
    count = velocity_module.MAX_TRACKS_PER_SOURCE + 1
    block = VelocityBlock()
    output = _measure(block, _image(1), _rows(*[(i, 0.0, 0.0) for i in range(count)]))
    state = block._sources["a"]

    assert len(output) == count
    assert list(state.rows) == list(range(1, count))
    assert len(state.times) == state.positions.shape[0] == count - 1
    assert state.smoothed.shape[0] == count - 1


def test_plain_call_without_video_identifier_uses_the_default_source():
    block = VelocityBlock()
    _measure(block, _image(1, ""), _rows((1, 0.0, 0.0)))
    output = _measure(block, _image(2, ""), _rows((1, 1.0, 0.0)))

    assert list(block._sources) == [velocity_module.DEFAULT_SOURCE]
    assert _values(output, "velocity") == [pytest.approx([10.0, 0.0])]


def test_other_context_failures_are_not_taken_for_a_plain_call(monkeypatch):
    def broken(self):
        raise RuntimeError("context broke")

    monkeypatch.setattr(VelocityBlock, "execution_context", property(broken))

    with pytest.raises(RuntimeError, match="context broke"):
        _measure(VelocityBlock(), _image(1, ""), _rows((1, 0.0, 0.0)))


def test_empty_prediction_is_returned_as_given_without_state_change():
    block = VelocityBlock()
    _measure(block, _image(1), _rows((1, 0.0, 0.0)))
    empty = _rows()

    assert _measure(block, _image(2), empty) is empty
    later = _measure(block, _image(3), _rows((1, 2.0, 0.0)))
    assert _values(later, "velocity") == [pytest.approx([10.0, 0.0])]


def test_invalid_calls_fail_before_state_changes():
    block = VelocityBlock()
    untracked = _rows((1, 0.0, 0.0))
    untracked.bboxes_metadata = [{}]
    no_video = ImageData.from_tensor(torch.zeros((3, 4, 4), dtype=torch.uint8))

    with pytest.raises(ValueError, match="every row needs a tracker_id"):
        _measure(block, _image(1), untracked)
    with pytest.raises(ValueError, match="needs video_metadata"):
        _measure(block, no_video, _rows((1, 0.0, 0.0)))
    with pytest.raises(ValueError, match="smoothing_alpha"):
        _measure(block, _image(1), _rows((1, 0.0, 0.0)), alpha=0.0)
    with pytest.raises(ValueError, match="pixels_per_meter"):
        _measure(block, _image(1), _rows((1, 0.0, 0.0)), ppm=0.0)
    assert block._sources == {}


def test_view_stays_lazy_and_keypoints_and_identity_are_kept(monkeypatch):
    def forbidden(self):
        raise AssertionError("full_res must not run")

    monkeypatch.setattr(SegmentationView, "full_res", forbidden)
    boxes = with_row_identity(_rows((1, 5.0, 5.0)), namespace="rows")
    view = SegmentationView(
        detections=boxes,
        scores=torch.zeros((1, 20, 30)),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((20, 30)),
    )
    points = KeyPoints(
        xy=torch.zeros((1, 2, 2)),
        class_id=torch.zeros(1, dtype=torch.int64),
        confidence=torch.ones((1, 2)),
    )
    block = VelocityBlock()

    measured_view = _measure(block, _image(1, "v"), view)
    measured_points = _measure(block, _image(1, "k"), (points, boxes))

    assert isinstance(measured_view, SegmentationView)
    assert measured_view.scores is view.scores
    assert measured_view.materialization_count == 0
    assert measured_view.detections.bboxes_metadata[0]["custom"] == 1
    assert measured_view.detections.bboxes_metadata[0]["speed"] == 0.0
    assert measured_points[0] is points
    assert measured_points[1].xyxy is boxes.xyxy
    assert row_record(measured_points[1]).namespace == "rows"
    assert boxes.bboxes_metadata == [{"tracker_id": 1, "custom": 1}]
