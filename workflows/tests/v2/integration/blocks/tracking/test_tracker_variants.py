"""The four ``trackers`` package blocks in compiled workflows and sessions.

One workflow runs SORT, ByteTrack, OC-SORT and BoT-SORT on the same frames:
three objects A, B, C moving on a 120x240 frame, C with confidence 0.4.
"""

from datetime import datetime

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.tracking.trackers_botsort.v1 import BoTSORTTrackerBlock
from roboflow_workflows.v2.blocks.tracking.trackers_bytetrack.v1 import ByteTrackTrackerBlock
from roboflow_workflows.v2.blocks.tracking.trackers_ocsort.v1 import OCSORTTrackerBlock
from roboflow_workflows.v2.blocks.tracking.trackers_sort.v1 import SORTTrackerBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

from inference_models.models.base.object_detection import Detections

# Root registers these blocks in the shared catalogue; this test owns its own.
CATALOGUE = Catalogue(
    [SORTTrackerBlock, ByteTrackTrackerBlock, OCSORTTrackerBlock, BoTSORTTrackerBlock]
)
STEPS = {
    "sort": SORTTrackerBlock.type,
    "bytetrack": ByteTrackTrackerBlock.type,
    "ocsort": OCSORTTrackerBlock.type,
    "botsort": BoTSORTTrackerBlock.type,
}
OUTPUTS = tuple(SORTTrackerBlock.outputs)
START = torch.tensor(
    [[10, 10, 40, 40], [100, 20, 130, 50], [200, 60, 230, 90]], dtype=torch.float32
)
STEP = torch.tensor([[4, 0, 4, 0], [0, 3, 0, 3], [-5, 0, -5, 0]], dtype=torch.float32)
NAMES = "ABC"
MOVING = object()  # _run default: the moving fixture at the given frame.

# Tracked objects per frame 0..4 with V1 defaults, verified on trackers 2.4.0.
PATTERNS = {
    "sort": ["", "", "ABC", "ABC", "ABC"],
    "bytetrack": ["", "AB", "AB", "AB", "AB"],
    "ocsort": ["", "AB", "AB", "AB", "AB"],
    "botsort": ["", "AB", "AB", "AB", "AB"],
}


def _image(source="a", *, fps=30, pixels=None):
    image = ImageData.from_tensor(
        torch.zeros((3, 120, 240), dtype=torch.uint8) if pixels is None else pixels,
        video_metadata=VideoMetadata(
            video_identifier=source,
            fps=fps,
            frame_number=0,
            frame_timestamp=datetime(2026, 10, 13),
        ),
    )

    return image


def _predictions(frame):
    predictions = Detections(
        xyxy=START + frame * STEP,
        confidence=torch.tensor([0.9, 0.85, 0.4]),
        class_id=torch.tensor([0, 1, 2]),
        image_metadata={"parent_id": f"frame-{frame}"},
        bboxes_metadata=[{"name": name} for name in NAMES],
    )

    return predictions


def _definition(*, steps=tuple(STEPS), outputs=OUTPUTS, **params):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
        ],
        "steps": [
            {
                "type": STEPS[name],
                "name": name,
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                **params.get(name, {}),
            }
            for name in steps
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": f"{name}_{output}",
                "selector": f"$steps.{name}.{output}",
            }
            for name in steps
            for output in outputs
        ],
    }

    return definition


def _plan(definition=None):
    plan = compile_workflow(definition or _definition(), catalogue=CATALOGUE)

    return plan


def _run(session, frame, *, predictions=MOVING, **inputs):
    if predictions is MOVING:
        predictions = _predictions(frame)
    row = session.run(
        {
            "image": [_image()],
            "predictions": [predictions],
            **inputs,
        }
    ).rows()[0]

    return row


def _names(prediction):
    rows = [] if prediction is None else prediction.bboxes_metadata or []
    names = "".join(sorted(row["name"] for row in rows))

    return names


def _ids(prediction):
    ids = {row["name"]: row["tracker_id"] for row in prediction.bboxes_metadata or []}

    return ids


def test_four_variants_confirm_the_same_frames_differently():
    session = _plan().create_session()
    try:
        rows = [_run(session, frame) for frame in range(5)]
    finally:
        session.close()
    for name, pattern in PATTERNS.items():
        assert [_names(row[f"{name}_tracked_detections"]) for row in rows] == pattern
        first = pattern.index(pattern[-1])
        assert _names(rows[first][f"{name}_new_instances"]) == pattern[-1]
        assert _names(rows[-1][f"{name}_already_seen_instances"]) == pattern[-1]
        assert _names(rows[-1][f"{name}_new_instances"]) == ""
        ids = [_ids(row[f"{name}_tracked_detections"]) for row in rows[first:]]
        assert all(frame_ids == ids[0] for frame_ids in ids)


def test_demanding_one_output_still_advances_each_tracker():
    reference = _plan().create_session()
    limited = _plan(_definition(outputs=("already_seen_instances",))).create_session()
    try:
        for frame in range(4):
            expected = _run(reference, frame)
            actual = _run(limited, frame)
            for name in STEPS:
                key = f"{name}_already_seen_instances"
                assert _names(actual[key]) == _names(expected[key])
        assert _names(actual["sort_already_seen_instances"]) == "ABC"
    finally:
        reference.close()
        limited.close()


def test_missing_predictions_skip_and_empty_predictions_age_tracks():
    lost_fast = {name: {"lost_track_buffer": 1} for name in STEPS}
    session = _plan(_definition(**lost_fast)).create_session()
    try:
        for frame in range(3):
            before = _run(session, frame)
        for _ in range(3):
            skipped = _run(session, 3, predictions=None)
            assert all(value is None for value in skipped.values())
        kept = _run(session, 3)
        assert _ids(kept["sort_tracked_detections"]) == _ids(
            before["sort_tracked_detections"]
        )
        empty = select_predictions(_predictions(4), indices=[])
        for frame in range(4, 8):
            aged = _run(session, frame, predictions=empty)
            assert all(len(value) == 0 for value in aged.values())
        for frame in range(8, 12):
            returned = _run(session, frame)
        old = set(_ids(kept["sort_tracked_detections"]).values())
        new = set(_ids(returned["sort_tracked_detections"]).values())
        assert len(new) == 3 and not old & new
    finally:
        session.close()


def test_processing_reset_replaces_every_tracker():
    session = _plan().create_session()
    try:
        for frame in range(3):
            _run(session, frame)
        session.apply_update(session.prepare_update(_plan(), reset=True))
        rows = [_run(session, frame) for frame in range(3)]
    finally:
        session.close()
    assert [_names(row["sort_tracked_detections"]) for row in rows] == PATTERNS["sort"][
        :3
    ]
    assert _names(rows[2]["sort_new_instances"]) == "ABC"
    assert _names(rows[2]["sort_already_seen_instances"]) == ""


def test_source_without_fps_fails_the_step():
    session = _plan(_definition(steps=("ocsort",))).create_session()
    try:
        with pytest.raises(StepExecutionError, match="fps"):
            session.run({"image": [_image(fps=None)], "predictions": [_predictions(0)]})
    finally:
        session.close()


@pytest.mark.parametrize(
    "step,parameter,value",
    [
        ("sort", "minimum_iou_threshold", 1.5),
        ("sort", "frame_rate", 0.0),
        ("bytetrack", "high_conf_det_threshold", None),
        ("ocsort", "delta_t", 0),
        ("ocsort", "direction_consistency_weight", True),
        ("botsort", "cmc_method", "bogus"),
        ("botsort", "enable_cmc", 1),
        ("botsort", "cmc_downscale", 0),
    ],
)
def test_invalid_literal_knobs_fail_at_compile(step, parameter, value):
    with pytest.raises(ParamsValidationError, match=parameter):
        _plan(_definition(steps=(step,), **{step: {parameter: value}}))


def test_selected_knob_is_validated_after_resolution():
    definition = _definition(
        steps=("sort",), sort={"track_activation_threshold": "$inputs.knob"}
    )
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "knob", "kind": ["float"]}
    )
    session = _plan(definition).create_session()
    try:
        with pytest.raises(
            StepExecutionError,
            match="track_activation_threshold is invalid after resolution",
        ):
            _run(session, 0, knob=1.5)
        for frame in range(3):
            row = _run(session, frame, knob=0.5)
        # 0.5 keeps C (0.4) from starting a track; the default 0.25 would not.
        assert _names(row["sort_tracked_detections"]) == "AB"
    finally:
        session.close()


def _panning_inputs(frame):
    """Camera panning 24 px/frame over seeded texture; objects fixed in world."""
    world = cv2.GaussianBlur(
        np.random.default_rng(7).random((240, 800)) * 255, (5, 5), 0
    )
    window = torch.from_numpy(world.astype(np.uint8))[:, frame * 24 : frame * 24 + 320]
    xyxy = torch.tensor([[150, 60, 180, 90], [220, 120, 250, 150]], dtype=torch.float32)
    xyxy[:, [0, 2]] -= frame * 24
    predictions = Detections(
        xyxy=xyxy,
        confidence=torch.tensor([0.9, 0.9]),
        class_id=torch.tensor([0, 0]),
        bboxes_metadata=[{"name": "A"}, {"name": "B"}],
    )
    inputs = {
        "image": [_image(pixels=window.unsqueeze(0).repeat(3, 1, 1).contiguous())],
        "predictions": [predictions],
    }

    return inputs


@pytest.mark.parametrize("enable_cmc,expected", [(False, ""), (True, "AB")])
def test_botsort_cmc_selector_keeps_ids_on_a_panning_camera(enable_cmc, expected):
    definition = _definition(steps=("botsort",), botsort={"enable_cmc": "$inputs.cmc"})
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "cmc", "kind": ["boolean"]}
    )
    session = _plan(definition).create_session()
    try:
        rows = [
            session.run({**_panning_inputs(frame), "cmc": enable_cmc}).rows()[0]
            for frame in range(5)
        ]
    finally:
        session.close()
    tracked = [row["botsort_tracked_detections"] for row in rows[1:]]
    assert [_names(prediction) for prediction in tracked] == [expected] * 4
    assert all(_ids(prediction) == _ids(tracked[0]) for prediction in tracked)
