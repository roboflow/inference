"""Real compiled tracker -> path deviation / event log, and overlap filtering.

Two sources share every run, interleaved in one batch. Expected values are
analytic; 20 x 20 px boxes, reference path ``(10, 50) -> (50, 50)``::

    cam-a   centre (10 * step, 50)   on the path line
    cam-b   centre (30, 80)          still; hypot(20, 30) px from both ends

The tracker, ``v2/trackers_bytetrack@v1``, does not report a track on the
frame that creates it. Sessions therefore start with step 0, which only
creates the tracks; consumers see the tracks from step 1. Tracker ids come
from a process-wide counter, so tests read them from the outputs.
"""

import math
from datetime import datetime

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.analytics.event_log import DetectionEventLogBlock
from roboflow_workflows.v2.blocks.analytics.overlap_filter import OverlapFilterBlock
from roboflow_workflows.v2.blocks.analytics.path_deviation import PathDeviationBlock
from roboflow_workflows.v2.data.batch import EntryMetadata, InputValue, SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

from inference_models.models.base.object_detection import Detections

BLOCKS = (PathDeviationBlock, DetectionEventLogBlock, OverlapFilterBlock)
SOURCES = ("cam-b", "cam-a")
PATH = [[10, 50], [50, 50]]
STILL = math.hypot(20, 30)
CLASS_NAMES = {0: "person", 1: "bicycle"}
# Explicit tracker knobs: confidence 0.9 starts a track; cam-a's 20 px box
# moves 10 px per step, so consecutive boxes overlap with IoU 1/3.
TRACKER = {
    "type": "v2/trackers_bytetrack@v1",
    "name": "track",
    "image": "$inputs.image",
    "frame_rate": 10.0,
    "minimum_iou_threshold": 0.2,
    "minimum_consecutive_frames": 2,
}


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


def _boxes(*rows):
    """Detections from (class_id, centre_x, centre_y) tuples, 20 x 20 px."""
    detections = Detections(
        xyxy=torch.tensor(
            [[x - 10.0, y - 10.0, x + 10.0, y + 10.0] for _, x, y in rows]
        ).reshape(-1, 4),
        class_id=torch.tensor([row[0] for row in rows], dtype=torch.int64),
        confidence=torch.full((len(rows),), 0.9),
        image_metadata={"class_names": CLASS_NAMES},
    )

    return detections


def _frame(step):
    return {"cam-a": _boxes((0, 10.0 * step, 50.0)), "cam-b": _boxes((0, 30.0, 80.0))}


def _empty():
    empty = select_predictions(_boxes((0, 0, 0)), indices=np.array([], dtype=np.int64))

    return empty


def _definition(*, deviation=None, log=None):
    steps = [
        {**TRACKER, "predictions": "$inputs.predictions"},
        {
            "type": "v2/path_deviation_analytics@v2",
            "name": "deviation",
            "image": "$inputs.image",
            "predictions": "$steps.track.tracked_detections",
            "reference_path": "$inputs.path",
            "triggering_anchor": "$inputs.anchor",
            **(deviation or {}),
        },
        {
            "type": "v2/detection_event_log@v1",
            "name": "log",
            "image": "$inputs.image",
            "predictions": "$steps.track.tracked_detections",
            "frame_threshold": "$inputs.threshold",
            "flush_interval": 1,
            "stale_frames": 1,
            **(log or {}),
        },
    ]
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
            {"type": "WorkflowParameter", "name": "path", "kind": ["list_of_values"]},
            {"type": "WorkflowParameter", "name": "anchor", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "threshold", "kind": ["integer"]},
        ],
        "steps": steps,
        "outputs": [
            {
                "type": "JsonField",
                "name": "deviation",
                "selector": "$steps.deviation.path_deviation_detections",
            },
            {"type": "JsonField", "name": "log", "selector": "$steps.log.event_log"},
            {
                "type": "JsonField",
                "name": "complete",
                "selector": "$steps.log.complete_events",
            },
        ],
    }

    return definition


def _plan(**kwargs):
    plan = compile_workflow(_definition(**kwargs), catalogue=_catalogue())

    return plan


def _run(session, frame, predictions, *, path=PATH, anchor="CENTER", threshold=3):
    rows = session.run(
        {
            "image": [_image(source, frame) for source in SOURCES],
            "predictions": [predictions[source] for source in SOURCES],
            "path": path,
            "anchor": anchor,
            "threshold": threshold,
        }
    ).rows()
    by_source = dict(zip(SOURCES, rows))

    return by_source


def _deviation(row):
    values = [
        (metadata["tracker_id"], metadata["path_deviation"])
        for metadata in row["deviation"].bboxes_metadata or []
    ]

    return values


def _track_id(row):
    (track_id,) = {track_id for track_id, _ in _deviation(row)}

    return track_id


def _create_tracks(session, frame=0, **inputs):
    created = _run(session, frame, _frame(frame), **inputs)
    assert all(len(created[source]["deviation"]) == 0 for source in SOURCES)
    assert all(created[source]["log"]["pending"] == {} for source in SOURCES)


def test_tracked_sources_get_path_deviation_and_event_logs():
    session = _plan().create_session()
    try:
        _create_tracks(session)
        frames = [_run(session, step, _frame(step)) for step in range(1, 6)]
    finally:
        session.close()

    a_id, b_id = _track_id(frames[0]["cam-a"]), _track_id(frames[0]["cam-b"])
    assert a_id != b_id
    # cam-a on the path: 40 px from its end on frame 1; on frame 5 the centre
    # (30, 50) is 20 px from both path points.
    a = [_deviation(frame["cam-a"]) for frame in frames]
    assert a[0] == [(a_id, 40.0)] and a[-1] == [(a_id, 20.0)]
    assert [_deviation(frame["cam-b"]) for frame in frames] == [
        [(b_id, pytest.approx(STILL))]
    ] * 5
    key = str(a_id)
    logs = [frame["cam-a"]["log"] for frame in frames]
    assert [sorted(log["pending"]) for log in logs] == [[key], [key], [], [], []]
    assert logs[-1]["logged"][key]["frame_count"] == 5
    assert logs[-1]["logged"][key]["class_name"] == "person"
    # Relative times count from the source's first frame. That is step 0,
    # which the log processed with no row, so the track spans 1 s to 5 s.
    assert logs[-1]["logged"][key]["first_seen_relative"] == 1.0
    assert logs[-1]["logged"][key]["last_seen_relative"] == 5.0


def test_missing_frames_skip_and_empty_frames_age_events():
    session = _plan().create_session()
    try:
        _create_tracks(session)
        for step in range(1, 4):
            before = _run(session, step, _frame(step))
        missing = _run(session, 4, {**_frame(4), "cam-a": None})
        returned = _run(session, 5, _frame(4))
        empty = [
            _run(session, frame, {**_frame(4), "cam-a": _empty()})
            for frame in (6, 7, 8)
        ]
    finally:
        session.close()

    a_id, b_id = _track_id(before["cam-a"]), _track_id(before["cam-b"])
    key = str(a_id)
    assert all(value is None for value in missing["cam-a"].values())
    assert _deviation(missing["cam-b"]) == [(b_id, pytest.approx(STILL))]
    # The missing frame added no point: the path is x = 10, 20, 30, 40.
    assert _deviation(returned["cam-a"]) == [(a_id, 20.0)]
    assert returned["cam-a"]["log"]["logged"][key]["frame_count"] == 4
    # Two unseen frames make the event stale; it completes once.
    assert [sorted(row["cam-a"]["complete"]) for row in empty] == [[], [key], []]
    assert empty[-1]["cam-a"]["log"] == {"logged": {}, "pending": {}}
    assert len(empty[0]["cam-a"]["deviation"]) == 0


def test_selected_path_anchor_and_threshold_change_between_runs():
    session = _plan().create_session()
    try:
        _create_tracks(session)
        _run(session, 1, _frame(1))
        bottom = _run(session, 2, _frame(2), anchor="BOTTOM_CENTER", threshold=10)
        shifted = _run(session, 3, _frame(3), path=[[10, 60], [30, 60]], threshold=1)
    finally:
        session.close()

    a_id = _track_id(bottom["cam-a"])
    # BOTTOM_CENTER of frame 2 is (20, 60), 10 px below the path.
    assert _deviation(bottom["cam-a"]) == [(a_id, pytest.approx(math.hypot(30, 10)))]
    assert sorted(bottom["cam-a"]["log"]["pending"]) == [str(a_id)]
    # Points (10, 50), (20, 60), (30, 50) against (10, 60) -> (30, 60).
    assert _deviation(shifted["cam-a"]) == [(a_id, 10.0)]
    assert sorted(shifted["cam-a"]["log"]["logged"]) == [str(a_id)]


def test_processing_reset_drops_paths_and_logs():
    session = _plan().create_session()
    try:
        _create_tracks(session)
        for step in range(1, 4):
            before = _run(session, step, _frame(step))
        session.apply_update(session.prepare_update(_plan(), reset=True))
        # The new tracker creates its tracks on step 4 and reports step 5.
        _create_tracks(session, 4)
        after = _run(session, 5, _frame(5))
    finally:
        session.close()

    a_id = _track_id(after["cam-a"])
    assert a_id != _track_id(before["cam-a"])
    # A fresh path at (50, 50): 40 px from the path start.
    assert _deviation(after["cam-a"]) == [(a_id, 40.0)]
    # The log counted step 4 as a frame of the source, with no row.
    assert after["cam-a"]["log"]["pending"][str(a_id)]["first_seen_frame"] == 2


def test_processing_reset_drops_paths_and_logs_for_reused_external_ids():
    # Keep ids fixed across the reset so a leaked path or event is observable.
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
        for frame in (1, 2, 3):
            before = run(frame)
        session.apply_update(session.prepare_update(plan, reset=True))
        after = run(4)
    finally:
        session.close()

    for source in SOURCES:
        assert _track_id(before[source]) == _track_id(after[source]) == ids[source]
        assert before[source]["log"]["logged"][str(ids[source])]["frame_count"] == 3
        event = after[source]["log"]["pending"][str(ids[source])]
        assert event["first_seen_frame"] == event["frame_count"] == 1
        assert after[source]["log"]["logged"] == {}
    # Only (40, 50) remains: 30 px from the reference path's start.
    # A leaked path through x = 10, 20, 30, 40 has Fréchet distance 20 px instead.
    assert _deviation(after["cam-a"]) == [(ids["cam-a"], 30.0)]
    assert _deviation(after["cam-b"]) == [(ids["cam-b"], pytest.approx(STILL))]


@pytest.mark.parametrize(
    "step,params,match",
    [
        ("deviation", {"triggering_anchor": "MIDDLE"}, "triggering_anchor"),
        ("deviation", {"triggering_anchor": "CENTER_OF_MASS"}, "triggering_anchor"),
        ("deviation", {"reference_path": [[0, 0]]}, "reference_path"),
        ("deviation", {"reference_path": [[0, 0], [1]]}, "reference_path"),
        ("deviation", {"max_sources": 0}, "max_sources"),
        ("log", {"frame_threshold": 0}, "frame_threshold"),
        ("log", {"stale_frames": 1.5}, "stale_frames"),
        ("log", {"fallback_fps": 0}, "fallback_fps"),
    ],
)
def test_invalid_literals_fail_at_compile(step, params, match):
    with pytest.raises(ParamsValidationError, match=match):
        _plan(**{step: params})


@pytest.mark.parametrize(
    "inputs,match",
    [
        ({"path": [[0, 0]]}, "at least 2"),
        ({"anchor": "MIDDLE"}, "triggering_anchor must be one"),
        ({"threshold": 0}, "frame_threshold"),
    ],
)
def test_selected_values_are_checked_per_run(inputs, match):
    session = _plan().create_session()
    try:
        with pytest.raises(StepExecutionError, match=match):
            _run(session, 1, _frame(1), **inputs)
    finally:
        session.close()


def test_each_stateful_block_bounds_sources():
    for step in ("deviation", "log"):
        session = _plan(**{step: {"max_sources": 1}}).create_session()
        try:
            with pytest.raises(StepExecutionError, match="max_sources=1"):
                _run(session, 1, _frame(1))
        finally:
            session.close()


def test_images_without_video_identifier_use_their_own_source():
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowParameter",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
        ],
        "steps": [
            {
                "type": "v2/detection_event_log@v1",
                "name": "log",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "frame_threshold": 2,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "total",
                "selector": "$steps.log.total_logged",
            }
        ],
    }
    session = compile_workflow(definition, catalogue=_catalogue()).create_session()
    tracked = _boxes((0, 30.0, 30.0))
    tracked.bboxes_metadata = [{"tracker_id": 1}]
    try:
        totals = []
        for source in ("left", "right", "left"):
            image = InputValue(
                data=ImageData.from_tensor(torch.zeros((3, 8, 8), dtype=torch.uint8)),
                metadata=EntryMetadata(sample={(0,): SampleContext(source_id=source)}),
            )
            output = session.run({"image": image, "predictions": tracked}).rows()
            totals.append(output[0]["total"])
    finally:
        session.close()

    # "left" reaches 2 frames on its second call; "right" has only one.
    assert totals == [0, 0, 1]


def _riders_definition():
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
            {"type": "WorkflowParameter", "name": "reference", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": "v2/overlap_filter@v1",
                "name": "riders",
                "predictions": "$inputs.predictions",
                "overlap_class_name": "$inputs.reference",
                "overlap_type": "Any Overlap",
            },
            {**TRACKER, "predictions": "$steps.riders.overlaps"},
            {
                "type": "v2/detection_event_log@v1",
                "name": "log",
                "image": "$inputs.image",
                "predictions": "$steps.track.tracked_detections",
                "frame_threshold": 2,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "riders",
                "selector": "$steps.riders.overlaps",
            },
            {"type": "JsonField", "name": "log", "selector": "$steps.log.event_log"},
        ],
    }

    return definition


def test_filtered_riders_are_tracked_and_logged_per_source():
    plan = compile_workflow(_riders_definition(), catalogue=_catalogue())
    session = plan.create_session()
    # cam-a: a rider on a bicycle and a pedestrian; cam-b: a pedestrian only.
    scenes = {
        "cam-a": _boxes((0, 30.0, 30.0), (1, 30.0, 40.0), (0, 90.0, 90.0)),
        "cam-b": _boxes((0, 30.0, 30.0)),
    }
    try:
        # Frame 1 creates the track; frames 2 and 3 reach frame_threshold=2.
        for frame in (1, 2, 3):
            rows = session.run(
                {
                    "image": [_image(source, frame) for source in SOURCES],
                    "predictions": [scenes[source] for source in SOURCES],
                    "reference": "bicycle",
                }
            ).rows()
    finally:
        session.close()

    by_source = dict(zip(SOURCES, rows))
    assert by_source["cam-a"]["riders"].xyxy.tolist() == [[20.0, 20.0, 40.0, 40.0]]
    assert len(by_source["cam-b"]["riders"]) == 0
    (key,) = by_source["cam-a"]["log"]["logged"]
    assert by_source["cam-a"]["log"]["logged"][key]["class_name"] == "person"
    assert by_source["cam-a"]["log"]["pending"] == {}
    assert by_source["cam-b"]["log"] == {"logged": {}, "pending": {}}


def test_invalid_overlap_type_fails_at_compile():
    definition = _riders_definition()
    definition["steps"][0]["overlap_type"] = "Partial"

    with pytest.raises(ParamsValidationError, match="overlap_type"):
        compile_workflow(definition, catalogue=_catalogue())
