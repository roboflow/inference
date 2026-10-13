"""Real compiled tracker -> line counter / time in zone, two sources per batch.

120 x 120 frames of a 10 fps file; 20 x 20 px boxes at x = 60. Line
``(0, 60) -> (120, 60)`` (moving up crosses in); zone: the band below y = 70::

    step              1    2    3    4    5    6
    cam-a centre y    35   45   55   65   75   85    moving down
    cam-b centre y    85   75   65   55   45   35    moving up

With the CENTER anchor both cross on step 4. With all four corners (the
default) a box crosses once its top or bottom edge has passed: step 5.
"""

from datetime import datetime

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.analytics.line_counter import LineCounterBlock
from roboflow_workflows.v2.blocks.analytics.time_in_zone import TimeInZoneBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

from inference_models.models.base.object_detection import Detections

BLOCKS = (LineCounterBlock, TimeInZoneBlock)
SOURCES = ("cam-b", "cam-a")
LINE = [[0, 60], [120, 60]]
ZONE = [[0, 70], [120, 70], [120, 120], [0, 120]]


def _catalogue():
    catalogue = create_catalogue()
    missing = [block for block in BLOCKS if catalogue.find(block.type) is None]

    return catalogue.with_blocks(missing)


def _image(source, frame):
    image = ImageData.from_tensor(
        torch.zeros((3, 120, 120), dtype=torch.uint8),
        video_metadata=VideoMetadata(
            video_identifier=source,
            frame_number=frame,
            frame_timestamp=datetime(2026, 10, 13, 12, 0, frame),
            fps=10.0,
            comes_from_video_file=True,
        ),
    )

    return image


def _box(y):
    detections = Detections(
        xyxy=torch.tensor([[50.0, y - 10.0, 70.0, y + 10.0]]),
        class_id=torch.tensor([0], dtype=torch.int64),
        confidence=torch.tensor([0.9]),
        image_metadata={"class_names": {0: "person"}},
    )

    return detections


def _frame(step):
    frame = {"cam-a": _box(25.0 + 10 * step), "cam-b": _box(95.0 - 10 * step)}

    return frame


def _empty():
    empty = select_predictions(_box(0.0), indices=np.array([], dtype=np.int64))

    return empty


def _definition(*, line=None, zone=None):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
            {"type": "WorkflowParameter", "name": "line", "kind": ["list_of_values"]},
            {"type": "WorkflowParameter", "name": "zone", "kind": ["list_of_values"]},
        ],
        "steps": [
            {
                "type": "v2/byte_tracker@v1",
                "name": "track",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "frame_rate": 10.0,
            },
            {
                "type": "v2/line_counter@v2",
                "name": "line",
                "image": "$inputs.image",
                "predictions": "$steps.track.tracked_detections",
                "line_segment": "$inputs.line",
                **(line or {}),
            },
            {
                "type": "v2/time_in_zone@v3",
                "name": "zone",
                "image": "$inputs.image",
                "predictions": "$steps.track.tracked_detections",
                "zone": "$inputs.zone",
                **(zone or {}),
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "in", "selector": "$steps.line.count_in"},
            {"type": "JsonField", "name": "out", "selector": "$steps.line.count_out"},
            {
                "type": "JsonField",
                "name": "crossed_out",
                "selector": "$steps.line.detections_out",
            },
            {
                "type": "JsonField",
                "name": "timed",
                "selector": "$steps.zone.timed_detections",
            },
            {
                "type": "JsonField",
                "name": "in_zone",
                "selector": "$steps.zone.count_in_zone",
            },
        ],
    }

    return definition


def _plan(**kwargs):
    plan = compile_workflow(_definition(**kwargs), catalogue=_catalogue())

    return plan


def _run(session, frame, predictions, *, line=LINE, zone=ZONE):
    rows = session.run(
        {
            "image": [_image(source, frame) for source in SOURCES],
            "predictions": [predictions[source] for source in SOURCES],
            "line": line,
            "zone": zone,
        }
    ).rows()
    by_source = dict(zip(SOURCES, rows))

    return by_source


def _run_steps(plan, steps):
    session = plan.create_session()
    try:
        frames = [_run(session, step, _frame(step)) for step in steps]
    finally:
        session.close()

    return frames


def _counts(frames, source):
    counts = [(frame[source]["in"], frame[source]["out"]) for frame in frames]

    return counts


def _times(row):
    times = [
        (metadata["tracker_id"], metadata["time_in_zone"])
        for metadata in row["timed"].bboxes_metadata or []
    ]

    return times


def test_tracked_sources_cross_the_line_and_are_timed_in_the_zone():
    frames = _run_steps(_plan(line={"triggering_anchor": "CENTER"}), range(1, 7))

    assert _counts(frames, "cam-a") == [(0, 0)] * 3 + [(0, 1)] * 3
    assert _counts(frames, "cam-b") == [(0, 0)] * 3 + [(1, 0)] * 3
    assert [len(frame["cam-a"]["crossed_out"]) for frame in frames] == [
        0,
        0,
        0,
        1,
        0,
        0,
    ]
    assert frames[3]["cam-a"]["crossed_out"].xyxy.tolist() == [[50, 55, 70, 75]]
    # cam-a enters the zone on step 5; cam-b leaves it after step 2.
    assert [_times(frame["cam-a"]) for frame in frames[4:]] == [
        [(1, 0.0)],
        [(1, pytest.approx(0.1))],
    ]
    assert [_times(frame["cam-b"]) for frame in frames[:3]] == [
        [(1, 0.0)],
        [(1, pytest.approx(0.1))],
        [],
    ]
    assert [frame["cam-b"]["in_zone"] for frame in frames] == [1, 1, 0, 0, 0, 0]


def test_default_anchors_need_the_whole_box_across():
    frames = _run_steps(_plan(), range(1, 7))

    assert _counts(frames, "cam-a") == [(0, 0)] * 4 + [(0, 1)] * 2
    assert _counts(frames, "cam-b") == [(0, 0)] * 4 + [(1, 0)] * 2


def test_out_of_zone_rows_are_kept_when_asked():
    zone = {
        "remove_out_of_zone_detections": False,
        "reset_out_of_zone_detections": False,
    }
    frames = _run_steps(_plan(zone=zone), range(1, 4))

    # cam-a stays above the zone: every row is kept with time 0.
    assert [_times(frame["cam-a"]) for frame in frames] == [[(1, 0.0)]] * 3
    assert [frame["cam-a"]["in_zone"] for frame in frames] == [0, 0, 0]
    assert _times(frames[2]["cam-b"]) == [(1, 0.0)]


def test_selected_line_change_restarts_counts_and_empty_frames_keep_them():
    session = _plan(line={"triggering_anchor": "CENTER"}).create_session()
    try:
        for step in range(1, 5):
            _run(session, step, _frame(step))
        empty = _run(session, 5, {"cam-a": _empty(), "cam-b": _empty()})
        moved = _run(session, 6, _frame(6), line=[[0, 100], [120, 100]], zone=[])
    finally:
        session.close()

    assert (empty["cam-a"]["in"], empty["cam-a"]["out"]) == (0, 1)
    assert len(empty["cam-a"]["crossed_out"]) == 0
    assert empty["cam-a"]["in_zone"] == 0
    # A new line starts from zero counts; no row is in an empty zone list.
    assert (moved["cam-a"]["in"], moved["cam-a"]["out"]) == (0, 0)
    assert moved["cam-a"]["in_zone"] == 0 and len(moved["cam-a"]["timed"]) == 0


def test_processing_reset_drops_counts_and_entries():
    plan = _plan(line={"triggering_anchor": "CENTER"})
    session = plan.create_session()
    try:
        for step in range(1, 6):
            before = _run(session, step, _frame(step))
        session.apply_update(session.prepare_update(plan, reset=True))
        after = _run(session, 6, _frame(6))
    finally:
        session.close()

    assert (before["cam-a"]["in"], before["cam-a"]["out"]) == (0, 1)
    assert (after["cam-a"]["in"], after["cam-a"]["out"]) == (0, 0)
    # A fresh entry on step 6, although cam-a entered the zone on step 5.
    assert _times(after["cam-a"]) == [(1, 0.0)]


@pytest.mark.parametrize(
    "step,params,match",
    [
        ("line", {"line_segment": [[0, 0]]}, "line_segment"),
        ("line", {"line_segment": [[5, 5], [5, 5]]}, "line_segment"),
        ("line", {"line_segment": [[0, 0], [True, 1]]}, "line_segment"),
        ("line", {"triggering_anchor": "MIDDLE"}, "triggering_anchor"),
        ("line", {"max_sources": 0}, "max_sources"),
        ("zone", {"zone": [[0, 0], [1, 1]]}, "zone"),
        ("zone", {"zone": [[[0, 0], [1, 0], [1, float("nan")]]]}, "zone"),
        ("zone", {"remove_out_of_zone_detections": 1}, "remove_out_of_zone"),
        ("zone", {"max_sources": 1.5}, "max_sources"),
    ],
)
def test_invalid_literals_fail_at_compile(step, params, match):
    with pytest.raises(ParamsValidationError, match=match):
        _plan(**{step: params})


@pytest.mark.parametrize(
    "inputs,match",
    [
        ({"line": [[0, 0]]}, "line_segment"),
        ({"zone": [[0, 0], [1, 1]]}, "zone"),
    ],
)
def test_selected_geometry_is_checked_per_run(inputs, match):
    session = _plan().create_session()
    try:
        with pytest.raises(StepExecutionError, match=match):
            _run(session, 1, _frame(1), **inputs)
    finally:
        session.close()


@pytest.mark.parametrize("step", ["line", "zone"])
def test_each_counter_bounds_sources(step):
    session = _plan(**{step: {"max_sources": 1}}).create_session()
    try:
        with pytest.raises(StepExecutionError, match="max_sources=1"):
            _run(session, 1, _frame(1))
    finally:
        session.close()
