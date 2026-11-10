"""Real compiler/session integration for per-source tracking and demand.

The tracker is ``roboflow_core/trackers_bytetrack@v1``. It never reports a track on the
frame that creates it, so a source's first frame has zero rows and the two
fixture rows are first reported, as new, on its second frame. Ids come from
one counter shared by every ByteTrack tracker in the process: tests compare
ids, never their values.
"""

from datetime import datetime

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.tracking import base
from roboflow_workflows.v2.blocks.tracking.trackers_bytetrack.v1 import ByteTrackTrackerBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

from inference_models.models.base.instance_segmentation import InstanceDetections

OUTPUTS = tuple(ByteTrackTrackerBlock.outputs)
# Explicit, not defaults: both rows (confidence 0.9 and 0.8) are high-confidence
# and start tracks; the boxes do not move, so they match with IoU 1.
KNOBS = {
    "minimum_iou_threshold": 0.3,
    "minimum_consecutive_frames": 2,
    "high_conf_det_threshold": 0.5,
}


def _image(source):
    image = ImageData.from_tensor(
        torch.zeros((3, 40, 60), dtype=torch.uint8),
        video_metadata=VideoMetadata(
            video_identifier=source,
            fps=30,
            frame_number=0,
            frame_timestamp=datetime(2026, 10, 13),
        ),
    )

    return image


def _predictions():
    masks = torch.zeros((2, 40, 60), dtype=torch.bool)
    masks[0, :10, :10] = True
    masks[1, :10, 20:30] = True
    predictions = InstanceDetections(
        xyxy=torch.tensor([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=torch.float32),
        confidence=torch.tensor([0.9, 0.8]),
        class_id=torch.tensor([0, 1]),
        mask=masks,
        image_metadata={"parent_id": "frame"},
        bboxes_metadata=[{"row": 0}, {"row": 1}],
    )

    return predictions


def _definition(*, outputs=OUTPUTS, **params):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["instance_segmentation_prediction"],
            },
            {"type": "WorkflowParameter", "name": "activation", "kind": ["float"]},
        ],
        "steps": [
            {
                "type": "roboflow_core/trackers_bytetrack@v1",
                "name": "track",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "track_activation_threshold": "$inputs.activation",
                "frame_rate": 30.0,
                "lost_track_buffer": 1,
                **KNOBS,
                **params,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.track.{name}"}
            for name in outputs
        ],
    }

    return definition


def _plan(**params):
    plan = compile_workflow(_definition(**params), catalogue=create_catalogue())

    return plan


def _run(session, *, sources=("a",), predictions=None, activation=0.25):
    result = session.run(
        {
            "image": [_image(source) for source in sources],
            "predictions": (
                [_predictions() for _ in sources]
                if predictions is None
                else predictions
            ),
            "activation": activation,
        }
    ).rows()

    return result


def _ids(prediction):
    ids = [row["tracker_id"] for row in prediction.bboxes_metadata or []]

    return ids


def _input_rows(prediction):
    rows = [row["row"] for row in prediction.bboxes_metadata or []]

    return rows


def _without_ids(prediction):
    metadata = [
        {key: value for key, value in row.items() if key != "tracker_id"}
        for row in prediction.bboxes_metadata or []
    ]

    return metadata


def test_interleaved_sources_partition_and_reset():
    plan = _plan()
    session = plan.create_session()
    try:
        created = _run(session, sources=("b", "a"))
        assert all(len(value) == 0 for row in created for value in row.values())
        first = _run(session, sources=("a", "b"))
        assert [_input_rows(row["new_instances"]) for row in first] == [[0, 1]] * 2
        ids = dict(zip("ab", (_ids(row["new_instances"]) for row in first)))
        # One shared counter: the two sources never share an id.
        assert len(set(ids["a"]) | set(ids["b"])) == 4
        second = _run(session, sources=("b", "a"))
        assert [_ids(row["already_seen_instances"]) for row in second] == [
            ids["b"],
            ids["a"],
        ]
        assert all(len(row["new_instances"]) == 0 for row in second)
        session.apply_update(session.prepare_update(_plan(), reset=True))
        recreated = _run(session, sources=("b", "a"))
        assert all(len(row["tracked_detections"]) == 0 for row in recreated)
        reset = _run(session, sources=("b", "a"))
        assert [_input_rows(row["new_instances"]) for row in reset] == [[0, 1]] * 2
        # A reset drops every source but does not rewind the shared counter.
        reset_ids = _ids(reset[0]["new_instances"]) + _ids(reset[1]["new_instances"])
        assert not set(reset_ids) & (set(ids["a"]) | set(ids["b"]))
    finally:
        session.close()


def test_missing_predictions_skip_but_empty_predictions_age_tracks():
    session = _plan().create_session()
    try:
        _run(session)
        first = _run(session)[0]
        assert _input_rows(first["new_instances"]) == [0, 1]
        for _ in range(4):
            missing = _run(session, predictions=[None])[0]
            assert all(value is None for value in missing.values())
        seen = _run(session)[0]
        assert _ids(seen["already_seen_instances"]) == _ids(first["new_instances"])
        empty = select_predictions(_predictions(), indices=np.array([], dtype=np.int64))
        for _ in range(3):
            result = _run(session, predictions=[empty])[0]
            assert all(len(value) == 0 for value in result.values())
        _run(session)
        returned = _run(session)[0]
        assert not set(_ids(returned["tracked_detections"])) & set(
            _ids(first["tracked_detections"])
        )
        assert _ids(returned["tracked_detections"])
    finally:
        session.close()


def test_capacity_and_resolved_selector_validation():
    session = _plan(max_sources=1).create_session()
    try:
        _run(session)
        ids = _ids(_run(session)[0]["new_instances"])
        assert len(ids) == 2
        with pytest.raises(StepExecutionError, match="max_sources=1"):
            _run(session, sources=("b",))
        with pytest.raises(
            StepExecutionError,
            match="track_activation_threshold is invalid after resolution",
        ):
            _run(session, activation=1.5)
        assert _ids(_run(session)[0]["already_seen_instances"]) == ids
    finally:
        session.close()


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("track_activation_threshold", 1.5),
        ("track_activation_threshold", True),
        ("lost_track_buffer", 0),
        ("lost_track_buffer", True),
        ("minimum_iou_threshold", -0.1),
        ("minimum_consecutive_frames", 0),
        ("high_conf_det_threshold", 1.5),
        ("instances_cache_size", 0),
        ("instances_cache_size", True),
    ],
)
def test_invalid_literal_parameters_fail_at_compile(parameter, value):
    with pytest.raises(ParamsValidationError, match=parameter):
        _plan(**{parameter: value})


def test_legacy_matching_knob_is_rejected_not_renamed():
    # supervision's maximum matching cost has no meaning for a minimum IoU.
    with pytest.raises(ParamsValidationError, match="minimum_matching_threshold"):
        _plan(minimum_matching_threshold=0.8)


@pytest.mark.parametrize("demanded", OUTPUTS)
def test_engine_only_gathers_demanded_payload_and_state_matches_all_outputs(
    demanded, monkeypatch
):
    selected_rows = []
    original = base._tracked_rows

    def observe(prediction, *, rows, ids):
        selected_rows.append(rows.tolist())
        return original(prediction, rows=rows, ids=ids)

    reference = _plan().create_session()
    limited = _plan(outputs=(demanded,)).create_session()
    try:
        reported = 0
        for _ in range(3):
            expected = _run(reference)[0][demanded]
            selected_rows.clear()
            monkeypatch.setattr(base, "_tracked_rows", observe)
            actual = _run(limited)[0][demanded]
            monkeypatch.setattr(base, "_tracked_rows", original)
            assert len(selected_rows) == 1
            assert torch.equal(actual.xyxy, expected.xyxy)
            assert torch.equal(actual.mask, expected.mask)
            # The sessions draw ids from one counter; compare the rest.
            assert _without_ids(actual) == _without_ids(expected)
            assert len(set(_ids(actual))) == len(actual)
            reported += len(actual)
        assert reported == (4 if demanded == "tracked_detections" else 2)
    finally:
        reference.close()
        limited.close()


@pytest.mark.parametrize(
    "parameter,value,kind",
    [
        ("track_activation_threshold", 1.5, "float"),
        ("lost_track_buffer", 0, "integer"),
        ("minimum_iou_threshold", -0.1, "float"),
        ("minimum_consecutive_frames", 0, "integer"),
        ("high_conf_det_threshold", 1.5, "float"),
    ],
)
def test_each_tracker_knob_accepts_and_validates_a_resolved_selector(
    parameter, value, kind
):
    definition = _definition()
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "knob", "kind": [kind]}
    )
    definition["steps"][0][parameter] = "$inputs.knob"
    plan = compile_workflow(definition, catalogue=create_catalogue())
    session = plan.create_session()
    try:
        with pytest.raises(
            StepExecutionError, match=f"{parameter} is invalid after resolution"
        ):
            session.run(
                {
                    "image": [_image("a")],
                    "predictions": [_predictions()],
                    "activation": 0.25,
                    "knob": value,
                }
            )
        valid = 0.25 if kind == "float" else 1
        for _ in range(2):
            rows = session.run(
                {
                    "image": [_image("a")],
                    "predictions": [_predictions()],
                    "activation": 0.25,
                    "knob": valid,
                }
            ).rows()
        assert _input_rows(rows[0]["new_instances"]) == [0, 1]
    finally:
        session.close()
