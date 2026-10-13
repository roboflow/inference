"""Real compiler/session integration for per-source tracking and demand."""

from datetime import datetime

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.tracking import base
from roboflow_workflows.v2.blocks.tracking.byte_track import ByteTrackerBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

from inference_models.models.base.instance_segmentation import InstanceDetections

OUTPUTS = tuple(ByteTrackerBlock.outputs)


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
                "type": "v2/byte_tracker@v1",
                "name": "track",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "track_activation_threshold": "$inputs.activation",
                "frame_rate": 30.0,
                "lost_track_buffer": 1,
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


def test_interleaved_sources_partition_and_reset():
    plan = _plan()
    session = plan.create_session()
    try:
        first = _run(session, sources=("b", "a"))
        assert [_ids(row["new_instances"]) for row in first] == [[1, 2], [1, 2]]
        second = _run(session, sources=("a", "b"))
        assert [_ids(row["already_seen_instances"]) for row in second] == [
            [1, 2],
            [1, 2],
        ]
        assert all(len(row["new_instances"]) == 0 for row in second)
        session.apply_update(session.prepare_update(_plan(), reset=True))
        reset = _run(session, sources=("b", "a"))
        assert [_ids(row["new_instances"]) for row in reset] == [[1, 2], [1, 2]]
    finally:
        session.close()


def test_missing_predictions_skip_but_empty_predictions_age_tracks():
    session = _plan().create_session()
    try:
        first = _run(session)[0]
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
        with pytest.raises(StepExecutionError, match="max_sources=1"):
            _run(session, sources=("b",))
        with pytest.raises(
            StepExecutionError,
            match="track_activation_threshold is invalid after resolution",
        ):
            _run(session, activation=1.5)
        assert _ids(_run(session)[0]["already_seen_instances"]) == [1, 2]
    finally:
        session.close()


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("track_activation_threshold", 1.5),
        ("track_activation_threshold", True),
        ("lost_track_buffer", 0),
        ("lost_track_buffer", True),
        ("minimum_matching_threshold", -0.1),
        ("minimum_consecutive_frames", 0),
        ("instances_cache_size", 0),
        ("instances_cache_size", True),
    ],
)
def test_invalid_literal_parameters_fail_at_compile(parameter, value):
    with pytest.raises(ParamsValidationError, match=parameter):
        _plan(**{parameter: value})


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
        for _ in range(2):
            expected = _run(reference)[0][demanded]
            selected_rows.clear()
            monkeypatch.setattr(base, "_tracked_rows", observe)
            actual = _run(limited)[0][demanded]
            monkeypatch.setattr(base, "_tracked_rows", original)
            assert len(selected_rows) == 1
            assert torch.equal(actual.xyxy, expected.xyxy)
            assert torch.equal(actual.mask, expected.mask)
            assert actual.bboxes_metadata == expected.bboxes_metadata
    finally:
        reference.close()
        limited.close()


@pytest.mark.parametrize(
    "parameter,value,kind",
    [
        ("track_activation_threshold", 1.5, "float"),
        ("lost_track_buffer", 0, "integer"),
        ("minimum_matching_threshold", -0.1, "float"),
        ("minimum_consecutive_frames", 0, "integer"),
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
        rows = session.run(
            {
                "image": [_image("a")],
                "predictions": [_predictions()],
                "activation": 0.25,
                "knob": valid,
            }
        ).rows()
        assert _ids(rows[0]["new_instances"]) == [1, 2]
    finally:
        session.close()
