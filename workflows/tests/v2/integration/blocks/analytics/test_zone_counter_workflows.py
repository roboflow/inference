"""Real compiled tracker -> line counter / time in zone, two sources per batch.

120 x 120 frames of a 10 fps file; 20 x 20 px boxes at x = 60. Line
``(0, 60) -> (120, 60)`` (moving up crosses in); zone: the band below y = 70::

    step              0    1    2    3    4    5    6
    cam-a centre y    25   35   45   55   65   75   85    moving down
    cam-b centre y    95   85   75   65   55   45   35    moving up

With the CENTER anchor both cross on step 4. With all four corners (the
default) the whole box must be across, so its trailing edge decides: step 5.

The tracker, ``v2/trackers_bytetrack@v1``, does not report a track on the
frame that creates it. Sessions therefore start with step 0, which only
creates the tracks; the counters see the tracks from step 1. Tracker ids come
from a process-wide counter, so tests read them from the outputs.
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
from roboflow_workflows.v2.engine.plan import CompileOptions

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
                "type": "v2/trackers_bytetrack@v1",
                "name": "track",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "frame_rate": 10.0,
                # Explicit: confidence 0.9 starts a track; 20 px boxes one
                # step (10 px) apart overlap with IoU 1/3.
                "minimum_iou_threshold": 0.2,
                "minimum_consecutive_frames": 2,
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


@pytest.fixture(params=("run", "phases"), autouse=True)
def block_execution(request, monkeypatch):
    """Exercise every compiled case in both supported local execution modes."""
    compile_original = compile_workflow

    def compile_in_mode(*args, **kwargs):
        kwargs["options"] = CompileOptions(block_execution=request.param)
        return compile_original(*args, **kwargs)

    monkeypatch.setitem(globals(), "compile_workflow", compile_in_mode)


def _plan(**kwargs):
    plan = compile_workflow(_definition(**kwargs), catalogue=_catalogue())

    return plan


def _run(session, frame, predictions, *, line=LINE, zone=ZONE, **parameters):
    rows = session.run(
        {
            "image": [_image(source, frame) for source in SOURCES],
            "predictions": [predictions[source] for source in SOURCES],
            "line": line,
            "zone": zone,
            **parameters,
        }
    ).rows()
    by_source = dict(zip(SOURCES, rows))

    return by_source


def _create_tracks(session, step=0, predictions=None, **parameters):
    created = _run(session, step, predictions or _frame(step), **parameters)
    for source in SOURCES:
        assert (created[source]["in"], created[source]["out"]) == (0, 0)
        assert len(created[source]["timed"]) == 0


def _run_steps(plan, steps):
    session = plan.create_session()
    try:
        _create_tracks(session)
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


def _track_id(row):
    (track_id,) = {track_id for track_id, _ in _times(row)}

    return track_id


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
    a_id, b_id = _track_id(frames[4]["cam-a"]), _track_id(frames[0]["cam-b"])
    assert a_id != b_id
    assert frames[3]["cam-a"]["crossed_out"].bboxes_metadata[0]["tracker_id"] == a_id
    assert [_times(frame["cam-a"]) for frame in frames[4:]] == [
        [(a_id, 0.0)],
        [(a_id, pytest.approx(0.1))],
    ]
    assert [_times(frame["cam-b"]) for frame in frames[:3]] == [
        [(b_id, 0.0)],
        [(b_id, pytest.approx(0.1))],
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
    a_id, b_id = _track_id(frames[0]["cam-a"]), _track_id(frames[0]["cam-b"])
    assert [_times(frame["cam-a"]) for frame in frames] == [[(a_id, 0.0)]] * 3
    assert [frame["cam-a"]["in_zone"] for frame in frames] == [0, 0, 0]
    assert _times(frames[2]["cam-b"]) == [(b_id, 0.0)]


def test_selected_line_change_restarts_counts_and_empty_frames_keep_them():
    session = _plan(line={"triggering_anchor": "CENTER"}).create_session()
    try:
        _create_tracks(session)
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


def test_selected_anchor_change_restarts_counts_without_a_static_crossing():
    definition = _definition(line={"triggering_anchor": "$inputs.anchor"})
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "anchor", "kind": ["string"]}
    )
    plan = compile_workflow(definition, catalogue=_catalogue())
    # Both sources move up. At y = 60 the box straddles the line: its top
    # edge (y 50) is above it, its bottom edge (y 70) below. The anchor
    # switches there while the box stands still.
    steps = [(80, "TOP_CENTER"), (75, "TOP_CENTER"), (70, "TOP_CENTER")]
    steps += [(65, "TOP_CENTER"), (60, "TOP_CENTER"), (60, "BOTTOM_CENTER")]
    steps += [(55, "BOTTOM_CENTER"), (50, "BOTTOM_CENTER"), (45, "BOTTOM_CENTER")]
    session = plan.create_session()
    try:
        _create_tracks(session, 0, {s: _box(85) for s in SOURCES}, anchor="TOP_CENTER")
        frames = [
            _run(session, step, {s: _box(y) for s in SOURCES}, anchor=anchor)
            for step, (y, anchor) in enumerate(steps, start=1)
        ]
    finally:
        session.close()

    # Top edge crosses on step 4; the switch restarts; bottom edge crosses on 9.
    expected = [(0, 0)] * 3 + [(1, 0)] * 2 + [(0, 0)] * 3 + [(1, 0)]
    assert _counts(frames, "cam-a") == _counts(frames, "cam-b") == expected
    assert [len(frame["cam-a"]["crossed_out"]) for frame in frames] == [0] * 9


def test_processing_reset_drops_counts_and_entries():
    plan = _plan(line={"triggering_anchor": "CENTER"})
    session = plan.create_session()
    try:
        _create_tracks(session)
        for step in range(1, 6):
            before = _run(session, step, _frame(step))
        session.apply_update(session.prepare_update(plan, reset=True))
        # The new tracker creates its tracks on step 6 and reports step 7.
        _create_tracks(session, 6)
        after = _run(session, 7, _frame(7))
    finally:
        session.close()

    assert (before["cam-a"]["in"], before["cam-a"]["out"]) == (0, 1)
    assert (after["cam-a"]["in"], after["cam-a"]["out"]) == (0, 0)
    # A fresh entry on step 7, although cam-a entered the zone on step 5.
    a_id = _track_id(after["cam-a"])
    assert a_id != _track_id(before["cam-a"])
    assert _times(after["cam-a"]) == [(a_id, 0.0)]


def test_processing_reset_drops_entries_for_reused_external_ids():
    # Fixed external ids expose stale dwell entries that global tracker ids hide.
    definition = _definition()
    definition["steps"] = definition["steps"][1:]
    for step in definition["steps"]:
        step["predictions"] = "$inputs.predictions"
    plan = compile_workflow(definition, catalogue=_catalogue())
    ids = {"cam-a": 7, "cam-b": 8}

    def run(frame):
        predictions = _frame(frame)
        for source, prediction in predictions.items():
            prediction.bboxes_metadata = [{"tracker_id": ids[source]}]
        return _run(session, frame, predictions)

    session = plan.create_session()
    try:
        entered = run(5)
        before = run(6)
        session.apply_update(session.prepare_update(plan, reset=True))
        after = run(7)
    finally:
        session.close()

    assert _times(entered["cam-a"]) == [(ids["cam-a"], 0.0)]
    assert _times(before["cam-a"]) == [(ids["cam-a"], pytest.approx(0.1))]
    # Same id stays inside the zone: fresh entry now, not 0.2 s since step 5.
    assert _times(after["cam-a"]) == [(ids["cam-a"], 0.0)]
    assert after["cam-a"]["in_zone"] == 1
    assert _times(after["cam-b"]) == []
    assert after["cam-b"]["in_zone"] == 0


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
