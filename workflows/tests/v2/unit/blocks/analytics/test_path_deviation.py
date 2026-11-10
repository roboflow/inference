"""Path deviation: Fréchet values, anchors, path changes, state and carriers."""

import math
from datetime import datetime

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics.path_deviation_analytics import v2 as module
from roboflow_workflows.v2.blocks.analytics.path_deviation_analytics.v2 import PathDeviationBlock
from roboflow_workflows.v2.blocks.tracking.byte_tracker.v3 import TrackerCapacityError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

LINE = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]


def _image(frame, source="a"):
    metadata = VideoMetadata(
        video_identifier=source,
        frame_number=frame,
        frame_timestamp=datetime(2026, 10, 13),
        fps=10.0,
        comes_from_video_file=True,
    )
    image = ImageData.from_tensor(
        torch.zeros((3, 40, 60), dtype=torch.uint8), video_metadata=metadata
    )

    return image


def _rows(*rows):
    """Detections from (tracker_id, centre_x, centre_y) tuples; 4 x 2 px boxes."""
    detections = Detections(
        xyxy=torch.tensor(
            [[x - 2.0, y - 1.0, x + 2.0, y + 1.0] for _, x, y in rows]
        ).reshape(-1, 4),
        class_id=torch.zeros(len(rows), dtype=torch.int64),
        confidence=torch.full((len(rows),), 0.9),
        bboxes_metadata=[{"tracker_id": row[0], "custom": row[0]} for row in rows]
        or None,
    )

    return detections


def _measure(block, image, prediction, path=LINE, **kwargs):
    output = block.run(
        image=image, predictions=prediction, reference_path=path, **kwargs
    )["path_deviation_detections"]

    return output


def _deviations(output):
    values = [row["path_deviation"] for row in output.bboxes_metadata]

    return values


def _frechet(path, reference):
    """Independent full discrete Fréchet table, filled row by row."""
    table = np.zeros((len(path), len(reference)))
    for i, p in enumerate(path):
        for j, r in enumerate(reference):
            distance = math.dist(p, r)
            if i == 0 and j == 0:
                table[i, j] = distance
            elif i == 0:
                table[i, j] = max(table[i, j - 1], distance)
            elif j == 0:
                table[i, j] = max(table[i - 1, j], distance)
            else:
                best = min(table[i - 1, j], table[i - 1, j - 1], table[i, j - 1])
                table[i, j] = max(best, distance)

    return table[-1, -1]


def test_track_shifted_from_the_path_reports_the_shift_at_the_end():
    block = PathDeviationBlock()
    values = [
        _deviations(_measure(block, _image(frame), _rows((1, x, 3.0))))[0]
        for frame, x in enumerate([0.0, 10.0, 20.0], start=1)
    ]

    assert values == pytest.approx([math.hypot(20, 3), math.hypot(10, 3), 3.0])


def test_values_match_the_full_table_for_random_tracks_and_a_path_change():
    generator = np.random.default_rng(7)
    first = generator.uniform(0, 100, (5, 2)).tolist()
    second = generator.uniform(0, 100, (3, 2)).tolist()
    # Whole-pixel centres, so float32 boxes hold them exactly.
    points = generator.integers(0, 100, (12, 2)).astype(float).tolist()
    block = PathDeviationBlock()

    for frame, (x, y) in enumerate(points, start=1):
        reference = first if frame <= 6 else second
        value = _deviations(_measure(block, _image(frame), _rows((4, x, y)), reference))
        assert value == [pytest.approx(_frechet(points[:frame], reference), abs=1e-9)]


def test_long_tracks_do_not_hit_the_recursion_limit():
    block = PathDeviationBlock()
    for frame in range(1, 1501):
        output = _measure(block, _image(frame), _rows((1, frame % 20, 1.0)))

    track = block._sources["a"][1]
    assert len(track.path) == 1500
    assert track.row.shape == (3,)
    assert _deviations(output)[0] == pytest.approx(_frechet(track.path, LINE), abs=1e-9)


@pytest.mark.parametrize(
    "anchor,point",
    [
        ("CENTER", (10.0, 5.0)),
        ("CENTER_LEFT", (8.0, 5.0)),
        ("CENTER_RIGHT", (12.0, 5.0)),
        ("TOP_CENTER", (10.0, 4.0)),
        ("TOP_LEFT", (8.0, 4.0)),
        ("TOP_RIGHT", (12.0, 4.0)),
        ("BOTTOM_LEFT", (8.0, 6.0)),
        ("BOTTOM_CENTER", (10.0, 6.0)),
        ("BOTTOM_RIGHT", (12.0, 6.0)),
    ],
)
def test_each_anchor_follows_its_box_point(anchor, point):
    # A two-point path at the anchor itself: deviation is its distance from
    # the far end, 30 px to the right.
    path = [list(point), [point[0] + 30.0, point[1]]]
    output = _measure(
        PathDeviationBlock(),
        _image(1),
        _rows((1, 10.0, 5.0)),
        path,
        triggering_anchor=anchor,
    )

    assert _deviations(output) == [30.0]


def test_sources_and_tracks_are_separate_and_missing_frames_add_no_point():
    block = PathDeviationBlock()
    _measure(block, _image(1, "a"), _rows((1, 0.0, 0.0), (2, 0.0, 5.0)))
    _measure(block, _image(1, "b"), _rows((1, 20.0, 9.0)))
    _measure(block, _image(2, "a"), _rows((2, 10.0, 5.0)))
    output = _measure(block, _image(3, "a"), _rows((1, 20.0, 0.0), (2, 20.0, 5.0)))

    tracks = block._sources["a"]
    assert tracks[1].path == [(0.0, 0.0), (20.0, 0.0)]
    assert block._sources["b"][1].path == [(20.0, 9.0)]
    # Track 1 skipped (10, 0): from (0, 0) to (20, 0), the middle reference
    # point is 10 px from either. Track 2 is 5 px off at every step.
    assert _deviations(output) == [10.0, 5.0]
    assert [row["custom"] for row in output.bboxes_metadata] == [1, 2]


def test_capacity_rejects_a_new_source_before_creating_state():
    block = PathDeviationBlock()
    _measure(block, _image(1, "a"), _rows((1, 0.0, 0.0)), max_sources=1)

    with pytest.raises(TrackerCapacityError, match="max_sources=1"):
        _measure(block, _image(1, "b"), _rows((1, 0.0, 0.0)), max_sources=1)
    assert list(block._sources) == ["a"]


def test_least_recently_seen_tracks_beyond_the_cap_are_forgotten(monkeypatch):
    monkeypatch.setattr(module, "MAX_TRACKS_PER_SOURCE", 2)
    block = PathDeviationBlock()
    _measure(block, _image(1), _rows((1, 0.0, 0.0), (2, 0.0, 0.0)))
    _measure(block, _image(2), _rows((1, 10.0, 0.0)))
    _measure(block, _image(3), _rows((3, 0.0, 0.0)))

    assert list(block._sources["a"]) == [1, 3]


def test_plain_call_without_video_identifier_uses_the_default_source():
    block = PathDeviationBlock()
    no_video = ImageData.from_tensor(torch.zeros((3, 4, 4), dtype=torch.uint8))
    _measure(block, no_video, _rows((1, 0.0, 0.0)))
    output = _measure(block, _image(2, ""), _rows((1, 20.0, 0.0)))

    assert list(block._sources) == [module.DEFAULT_SOURCE]
    assert _deviations(output) == [10.0]


def test_other_context_failures_are_not_taken_for_a_plain_call(monkeypatch):
    def broken(self):
        raise RuntimeError("context broke")

    monkeypatch.setattr(PathDeviationBlock, "execution_context", property(broken))

    with pytest.raises(RuntimeError, match="context broke"):
        _measure(PathDeviationBlock(), _image(1, ""), _rows((1, 0.0, 0.0)))


def test_empty_prediction_is_returned_as_given_and_registers_the_source():
    block = PathDeviationBlock()
    empty = _rows()

    assert _measure(block, _image(1), empty) is empty
    assert len(block._sources["a"]) == 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"triggering_anchor": "CENTER_OF_MASS"}, "triggering_anchor must be one"),
        ({"triggering_anchor": 3}, "triggering_anchor must be one"),
        ({"path": [[0, 0]]}, "at least 2"),
        ({"path": [[0, 0], [1]]}, "at least 2"),
        ({"path": [[0, 0], [1, float("nan")]]}, "finite"),
        ({"path": [[0, 0], [True, 1]]}, "finite numbers"),
        ({"path": "0,0;1,1"}, "at least 2"),
    ],
)
def test_invalid_calls_fail_before_state_changes(kwargs, match):
    block = PathDeviationBlock()

    with pytest.raises(ValueError, match=match):
        _measure(block, _image(1), _rows((1, 0.0, 0.0)), **kwargs)
    assert block._sources == {}


def test_untracked_rows_fail_before_state_changes():
    block = PathDeviationBlock()
    untracked = _rows((1, 0.0, 0.0), (2, 0.0, 0.0))
    untracked.bboxes_metadata = [{"tracker_id": 1}, {}]

    with pytest.raises(ValueError, match="every row needs a tracker_id"):
        _measure(block, _image(1), untracked)
    assert block._sources == {}


def test_dense_masks_and_noncontiguous_boxes_follow_rows_unchanged():
    xyxy = torch.tensor([[8.0, 4.0, 12.0, 6.0], [-2.0, -1.0, 2.0, 1.0]]).t()
    xyxy = xyxy.contiguous().t()  # non-contiguous view, same values
    masks = torch.zeros((2, 10, 20), dtype=torch.bool)
    masks[1, 0, 0] = True
    rows = InstanceDetections(
        xyxy=xyxy,
        class_id=torch.zeros(2, dtype=torch.int64),
        confidence=torch.ones(2),
        mask=masks,
        bboxes_metadata=[{"tracker_id": 5}, {"tracker_id": 6}],
    )

    output = _measure(PathDeviationBlock(), _image(1), rows)

    assert not xyxy.is_contiguous()
    assert output.mask is masks
    assert output.xyxy is xyxy
    assert _deviations(output) == pytest.approx([math.hypot(10, 5), 20.0])
    assert rows.bboxes_metadata == [{"tracker_id": 5}, {"tracker_id": 6}]


def test_view_stays_lazy_and_row_identity_is_kept(monkeypatch):
    def forbidden(self):
        raise AssertionError("full_res must not run")

    monkeypatch.setattr(SegmentationView, "full_res", forbidden)
    boxes = with_row_identity(_rows((1, 20.0, 0.0)), namespace="rows")
    view = SegmentationView(
        detections=boxes,
        scores=torch.zeros((1, 20, 30)),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((20, 30)),
    )

    measured = _measure(PathDeviationBlock(), _image(1), view)

    assert isinstance(measured, SegmentationView)
    assert measured.scores is view.scores
    assert measured.materialization_count == 0
    assert measured.detections.bboxes_metadata[0]["path_deviation"] == 20.0
    assert row_record(measured.detections).namespace == "rows"
    assert boxes.bboxes_metadata == [{"tracker_id": 1, "custom": 1}]
