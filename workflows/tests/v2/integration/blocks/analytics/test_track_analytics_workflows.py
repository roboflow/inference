"""Real compiled tracker -> path deviation / event log, and overlap filtering.

Two sources share every run, interleaved in one batch. Expected values are
analytic; 20 x 20 px boxes, reference path ``(10, 50) -> (50, 50)``::

    cam-a   centre (10 * step, 50)   on the path line
    cam-b   centre (30, 80)          still; hypot(20, 30) px from both ends
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
        {
            "type": "v2/byte_tracker@v1",
            "name": "track",
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
            "frame_rate": 10.0,
        },
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


def test_tracked_sources_get_path_deviation_and_event_logs():
    session = _plan().create_session()
    try:
        frames = [_run(session, step, _frame(step)) for step in range(1, 6)]
    finally:
        session.close()

    # cam-a on the path: 40 px from its end on frame 1; on frame 5 the centre
    # (30, 50) is 20 px from both path points.
    a = [_deviation(frame["cam-a"]) for frame in frames]
    assert a[0] == [(1, 40.0)] and a[-1] == [(1, 20.0)]
    assert [_deviation(frame["cam-b"]) for frame in frames] == [
        [(1, pytest.approx(STILL))]
    ] * 5
    logs = [frame["cam-a"]["log"] for frame in frames]
    assert [sorted(log["pending"]) for log in logs] == [["1"], ["1"], [], [], []]
    assert logs[-1]["logged"]["1"]["frame_count"] == 5
    assert logs[-1]["logged"]["1"]["class_name"] == "person"
    assert logs[-1]["logged"]["1"]["last_seen_relative"] == 4.0


def test_missing_frames_skip_and_empty_frames_age_events():
    session = _plan().create_session()
    try:
        for step in range(1, 4):
            _run(session, step, _frame(step))
        missing = _run(session, 4, {**_frame(4), "cam-a": None})
        returned = _run(session, 5, _frame(4))
        empty = [
            _run(session, frame, {**_frame(4), "cam-a": _empty()})
            for frame in (6, 7, 8)
        ]
    finally:
        session.close()

    assert all(value is None for value in missing["cam-a"].values())
    assert _deviation(missing["cam-b"]) == [(1, pytest.approx(STILL))]
    # The missing frame added no point: the path is x = 10, 20, 30, 40.
    assert _deviation(returned["cam-a"]) == [(1, 20.0)]
    assert returned["cam-a"]["log"]["logged"]["1"]["frame_count"] == 4
    # Two unseen frames make the event stale; it completes once.
    assert [sorted(row["cam-a"]["complete"]) for row in empty] == [[], ["1"], []]
    assert empty[-1]["cam-a"]["log"] == {"logged": {}, "pending": {}}
    assert len(empty[0]["cam-a"]["deviation"]) == 0


def test_selected_path_anchor_and_threshold_change_between_runs():
    session = _plan().create_session()
    try:
        _run(session, 1, _frame(1))
        bottom = _run(session, 2, _frame(2), anchor="BOTTOM_CENTER", threshold=10)
        shifted = _run(session, 3, _frame(3), path=[[10, 60], [30, 60]], threshold=1)
    finally:
        session.close()

    # BOTTOM_CENTER of frame 2 is (20, 60), 10 px below the path.
    assert _deviation(bottom["cam-a"]) == [(1, pytest.approx(math.hypot(30, 10)))]
    assert sorted(bottom["cam-a"]["log"]["pending"]) == ["1"]
    # Points (10, 50), (20, 60), (30, 50) against (10, 60) -> (30, 60).
    assert _deviation(shifted["cam-a"]) == [(1, 10.0)]
    assert sorted(shifted["cam-a"]["log"]["logged"]) == ["1"]


def test_processing_reset_drops_paths_and_logs():
    session = _plan().create_session()
    try:
        for step in range(1, 4):
            _run(session, step, _frame(step))
        session.apply_update(session.prepare_update(_plan(), reset=True))
        after = _run(session, 4, _frame(4))
    finally:
        session.close()

    # A fresh path at (40, 50): 30 px from the path start.
    assert [value for _, value in _deviation(after["cam-a"])] == [30.0]
    assert after["cam-a"]["log"]["pending"]["1"]["first_seen_frame"] == 1


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
            {
                "type": "v2/byte_tracker@v1",
                "name": "track",
                "image": "$inputs.image",
                "predictions": "$steps.riders.overlaps",
                "frame_rate": 10.0,
            },
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
        for frame in (1, 2):
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
    assert list(by_source["cam-a"]["log"]["logged"]) == ["1"]
    assert by_source["cam-a"]["log"]["logged"]["1"]["class_name"] == "person"
    assert by_source["cam-b"]["log"] == {"logged": {}, "pending": {}}


def test_invalid_overlap_type_fails_at_compile():
    definition = _riders_definition()
    definition["steps"][0]["overlap_type"] = "Partial"

    with pytest.raises(ParamsValidationError, match="overlap_type"):
        compile_workflow(definition, catalogue=_catalogue())
