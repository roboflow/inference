"""Real compiled tracker -> class lock / stabilizer / velocity workflows.

Two sources share every run, interleaved in one batch. Each has one object
moving at a constant speed, so every expected value is analytic::

    cam-a   x1 = 3 * frame          30 px/s at 10 fps
    cam-b   y1 = 100 - 2 * frame    -20 px/s at 10 fps

The tracker, ``roboflow_core/trackers_bytetrack@v1``, does not report a track on the
frame that creates it. Every session therefore starts with frame 0, which only
creates the tracks: the consumers see zero rows there, and the analytic
values start on frame 1. Tracker ids come from a process-wide counter, so
tests read them from the outputs.
"""

from datetime import datetime

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.analytics.velocity.v1 import VelocityBlock
from roboflow_workflows.v2.blocks.tracking.track_class_lock.v1 import TrackClassLockBlock
from roboflow_workflows.v2.blocks.tracking.stabilize_detections.v1 import StabilizeDetectionsBlock
from roboflow_workflows.v2.data.batch import EntryMetadata, InputValue, SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

from inference_models.models.base.object_detection import Detections

CONSUMERS = (TrackClassLockBlock, StabilizeDetectionsBlock, VelocityBlock)
SOURCES = ("cam-b", "cam-a")
OUTPUTS = {
    "locked": "$steps.lock.tracked_detections",
    "stable": "$steps.stable.tracked_detections",
    "speed": "$steps.speed.velocity_detections",
}


def _catalogue():
    catalogue = create_catalogue()
    missing = [block for block in CONSUMERS if catalogue.find(block.type) is None]

    return catalogue.with_blocks(missing)


def _image(source, frame):
    image = ImageData.from_tensor(
        torch.zeros((3, 200, 200), dtype=torch.uint8),
        video_metadata=VideoMetadata(
            video_identifier=source,
            frame_number=frame,
            frame_timestamp=datetime(2026, 10, 13),
            fps=10.0,
            comes_from_video_file=True,
        ),
    )

    return image


def _box(source, frame, class_id=0):
    x, y = (3.0 * frame, 50.0) if source == "cam-a" else (50.0, 100.0 - 2.0 * frame)
    detections = Detections(
        xyxy=torch.tensor([[x, y, x + 40.0, y + 40.0]]),
        class_id=torch.tensor([class_id]),
        confidence=torch.tensor([0.9]),
        image_metadata={"class_names": {0: "car", 1: "truck"}},
    )

    return detections


def _empty():
    empty = select_predictions(_box("cam-a", 0), indices=np.array([], dtype=np.int64))

    return empty


def _definition(*, lock=None, stable=None, speed=None, outputs=OUTPUTS):
    # Explicit knobs: confidence 0.9 starts a track; a box moves at most 3 px
    # per frame, so consecutive 40 px boxes overlap with IoU above 0.8.
    track = {
        "type": "roboflow_core/trackers_bytetrack@v1",
        "name": "track",
        "frame_rate": 10.0,
        "minimum_iou_threshold": 0.3,
        "minimum_consecutive_frames": 2,
    }
    tracked = "$steps.track.tracked_detections"
    steps = [
        {**track, "image": "$inputs.image", "predictions": "$inputs.predictions"},
        {
            "type": "roboflow_core/track_class_lock@v1",
            "name": "lock",
            "image": "$inputs.image",
            "predictions": tracked,
            "min_votes": "$inputs.min_votes",
            "lead_margin": 1,
            "switch_after": 2,
            **(lock or {}),
        },
        {
            "type": "roboflow_core/stabilize_detections@v1",
            "name": "stable",
            "image": "$inputs.image",
            "predictions": tracked,
            "bbox_smoothing_coefficient": 0.5,
            **(stable or {}),
        },
        {
            "type": "roboflow_core/velocity@v1",
            "name": "speed",
            "image": "$inputs.image",
            "predictions": tracked,
            "pixels_per_meter": "$inputs.ppm",
            "smoothing_alpha": 1.0,
            **(speed or {}),
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
            {"type": "WorkflowParameter", "name": "min_votes", "kind": ["integer"]},
            {"type": "WorkflowParameter", "name": "ppm", "kind": ["float"]},
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def _plan(**kwargs):
    plan = compile_workflow(_definition(**kwargs), catalogue=_catalogue())

    return plan


def _run(session, frame, *, predictions=None, min_votes=2, ppm=10.0, classes=None):
    classes = classes or {}
    rows = session.run(
        {
            "image": [_image(source, frame) for source in SOURCES],
            "predictions": (
                [_box(source, frame, classes.get(source, 0)) for source in SOURCES]
                if predictions is None
                else predictions
            ),
            "min_votes": min_votes,
            "ppm": ppm,
        }
    ).rows()
    by_source = dict(zip(SOURCES, rows))

    return by_source


def _speed(row):
    values = [
        (metadata["tracker_id"], metadata["velocity"], metadata["speed"])
        for metadata in row["speed"].bboxes_metadata or []
    ]

    return values


def _track_ids(row):
    ids = {source: _speed(row[source])[0][0] for source in SOURCES}

    return ids


def _create_tracks(session, **inputs):
    created = _run(session, 0, **inputs)
    assert all(
        len(value) == 0 for source in SOURCES for value in created[source].values()
    )


def test_tracked_sources_get_analytic_velocity_smoothing_and_locks():
    session = _plan().create_session()
    try:
        _create_tracks(session)
        frames = [
            _run(session, frame, classes={"cam-a": cls})
            for frame, cls in zip(range(1, 5), [0, 0, 1, 1])
        ]
    finally:
        session.close()

    a, b = frames[-1]["cam-a"], frames[-1]["cam-b"]
    ids = _track_ids(frames[0])
    assert ids["cam-a"] != ids["cam-b"]
    assert _track_ids(frames[-1]) == ids
    # 30 px/s and -20 px/s at 10 px/m; the first reported frame of a track
    # reports 0.
    assert _speed(frames[0]["cam-a"]) == [(ids["cam-a"], [0.0, 0.0], 0.0)]
    assert _speed(a) == [(ids["cam-a"], pytest.approx([3.0, 0.0]), pytest.approx(3.0))]
    assert _speed(b) == [(ids["cam-b"], pytest.approx([0.0, -2.0]), pytest.approx(2.0))]
    # a = 0.5 keeps a seen box 1.5 px (cam-a) behind its raw position.
    assert a["stable"].xyxy.tolist() == [[10.5, 50.0, 50.5, 90.0]]
    assert b["stable"].xyxy.tolist() == [[50.0, 93.0, 90.0, 133.0]]
    # car locked at frame 2 (2 votes, lead 2); two truck frames move it.
    locks = [
        (
            row["cam-a"]["locked"].bboxes_metadata[0]["class"],
            row["cam-a"]["locked"].bboxes_metadata[0]["class_locked"],
        )
        for row in frames
    ]
    assert locks == [("car", False), ("car", True), ("car", True), ("truck", True)]
    assert b["locked"].bboxes_metadata[0]["class"] == "car"


def test_missing_frames_skip_empty_frames_age_and_returning_tracks_keep_time():
    session = _plan().create_session()
    try:
        _create_tracks(session)
        for frame in range(1, 5):
            before = _run(session, frame)
        missing = _run(session, 5, predictions=[_box("cam-b", 5), None])
        empty = _run(session, 6, predictions=[_box("cam-b", 6), _empty()])
        returned = _run(session, 7)
    finally:
        session.close()

    assert all(value is None for value in missing["cam-a"].values())
    assert _speed(missing["cam-b"])[0][1] == pytest.approx([0.0, -2.0])
    assert len(empty["cam-a"]["speed"]) == 0
    assert len(empty["cam-a"]["locked"]) == 0
    # The empty frame is a frame for the stabilizer: one gap row at the cached
    # frame-4 box advanced by a * 3 px.
    assert empty["cam-a"]["stable"].xyxy.tolist() == [[13.5, 50.0, 53.5, 90.0]]
    # Frame 4 -> 7 is 0.3 s by frame number for 9 px: still 3 m/s, and the
    # track keeps its id through the missing and the empty frame.
    assert _speed(returned["cam-a"]) == [
        (_track_ids(before)["cam-a"], pytest.approx([3.0, 0.0]), pytest.approx(3.0))
    ]


def test_selected_parameters_change_between_runs():
    session = _plan().create_session()
    try:
        _create_tracks(session, min_votes=5)
        _run(session, 1, min_votes=5)
        unlocked = _run(session, 2, min_votes=5)
        locked = _run(session, 3, min_votes=1, ppm=5.0)
    finally:
        session.close()

    assert unlocked["cam-a"]["locked"].bboxes_metadata[0]["class_locked"] is False
    assert locked["cam-a"]["locked"].bboxes_metadata[0]["class_locked"] is True
    # px/s state, so a new pixels_per_meter rescales at once: 30 / 5.
    assert _speed(locked["cam-a"])[0][2] == pytest.approx(6.0)


def test_processing_reset_drops_every_consumer_state():
    session = _plan().create_session()
    try:
        _create_tracks(session)
        for frame in range(1, 4):
            before = _run(session, frame)
        session.apply_update(session.prepare_update(_plan(), reset=True))
        # The new tracker creates its tracks on frame 4 and reports frame 5.
        recreated = _run(session, 4)
        after = _run(session, 5)
    finally:
        session.close()

    assert _speed(before["cam-a"])[0][2] == pytest.approx(3.0)
    assert all(len(value) == 0 for value in recreated["cam-a"].values())
    assert _speed(after["cam-a"])[0][2] == 0.0
    assert _track_ids(after)["cam-a"] != _track_ids(before)["cam-a"]
    assert after["cam-a"]["locked"].bboxes_metadata[0]["class_locked"] is False
    assert after["cam-a"]["stable"].xyxy.tolist() == [[15.0, 50.0, 55.0, 90.0]]


def test_processing_reset_drops_consumer_state_for_reused_external_ids():
    # External tracked rows may reuse ids. The built-in tracker's global ids
    # would hide stale per-track entries after a processing reset.
    definition = _definition()
    definition["steps"] = definition["steps"][1:]
    for step in definition["steps"]:
        step["predictions"] = "$inputs.predictions"
    plan = compile_workflow(definition, catalogue=_catalogue())
    ids = {"cam-a": 7, "cam-b": 8}

    def run(frame):
        predictions = [_box(source, frame) for source in SOURCES]
        for source, prediction in zip(SOURCES, predictions):
            prediction.bboxes_metadata = [{"tracker_id": ids[source]}]
        return _run(session, frame, predictions=predictions)

    session = plan.create_session()
    try:
        for frame in (1, 2, 3):
            before = run(frame)
        session.apply_update(session.prepare_update(plan, reset=True))
        after = run(4)
    finally:
        session.close()

    assert _track_ids(before) == _track_ids(after) == ids
    for source, speed in (("cam-a", 3.0), ("cam-b", 2.0)):
        assert _speed(before[source])[0][2] == pytest.approx(speed)
        assert before[source]["locked"].bboxes_metadata[0]["class_locked"] is True
        assert _speed(after[source]) == [(ids[source], [0.0, 0.0], 0.0)]
        assert after[source]["locked"].bboxes_metadata[0]["class_locked"] is False
        # A fresh stabilizer emits the raw frame-4 box with no cached smoothing.
        assert after[source]["stable"].xyxy.tolist() == _box(source, 4).xyxy.tolist()


@pytest.mark.parametrize("step", ["lock", "stable", "speed"])
def test_each_consumer_bounds_sources(step):
    session = _plan(**{step: {"max_sources": 1}}).create_session()
    try:
        with pytest.raises(StepExecutionError, match="max_sources=1"):
            _run(session, 1)
    finally:
        session.close()


@pytest.mark.parametrize(
    "step,params,match",
    [
        ("lock", {"reattach_window": 400}, "reattach_window"),
        ("lock", {"vote_confidence": 1.5}, "vote_confidence"),
        ("lock", {"state_ttl": 0}, "state_ttl"),
        ("stable", {"smoothing_window_size": 0}, "smoothing_window_size"),
        ("stable", {"bbox_smoothing_coefficient": 1.5}, "bbox_smoothing"),
        ("speed", {"smoothing_alpha": 0.0}, "smoothing_alpha"),
        ("speed", {"pixels_per_meter": -1.0}, "pixels_per_meter"),
        ("speed", {"max_sources": True}, "max_sources"),
    ],
)
def test_invalid_literals_fail_at_compile(step, params, match):
    with pytest.raises(ParamsValidationError, match=match):
        _plan(**{step: params})


def test_resolved_selectors_are_validated_per_run():
    session = _plan(lock={"state_ttl": "$inputs.min_votes"}).create_session()
    try:
        with pytest.raises(
            StepExecutionError, match="pixels_per_meter is invalid after resolution"
        ):
            _run(session, 1, ppm=0.0, min_votes=100)
        with pytest.raises(StepExecutionError, match="must not exceed state_ttl"):
            _run(session, 1, min_votes=2)
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
                "type": "roboflow_core/track_class_lock@v1",
                "name": "lock",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "min_votes": 2,
                "lead_margin": 1,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "locked",
                "selector": "$steps.lock.tracked_detections",
            }
        ],
    }
    session = compile_workflow(definition, catalogue=_catalogue()).create_session()
    tracked = _box("cam-a", 1)
    tracked.bboxes_metadata = [{"tracker_id": 1}]
    try:
        flags = []
        for source in ("left", "right", "left"):
            image = InputValue(
                data=ImageData.from_tensor(torch.zeros((3, 8, 8), dtype=torch.uint8)),
                metadata=EntryMetadata(sample={(0,): SampleContext(source_id=source)}),
            )
            output = session.run({"image": image, "predictions": tracked}).rows()
            flags.append(output[0]["locked"].bboxes_metadata[0]["class_locked"])
    finally:
        session.close()

    # "left" reaches 2 votes on its second frame; "right" has only one.
    assert flags == [False, False, True]
