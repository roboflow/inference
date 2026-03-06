"""
Integration tests for detection → continue_if → email/CSV workflows.

Workflow JSON definitions live in the workflow_definitions/ subdirectory.
Detection is mocked with configurable per-image (or per-slice) counts so we get
deterministic output matching real-model behaviour. Scenarios:

1) Batch of 4 images: indices 1 and 3 have detections (2 and 1). Tests (a)-(d) for email with/without count and with/without image names.
2) Sliced (no stitch): image_slicer -> detection -> continue_if -> email per slice. First 3 images -> 4 slices each, last -> 8 slices (20 total). Detections in slice indices 6, 7, 16 only. One email per slice with detections; (a) and (b) only (no image_names - they do not propagate to slices).
3) Like 1) with enable_email gate: assert behaviour when True/False.
4) Like 1) but CSV sink; assert rows and content in temp dir.
"""

import json
import os
from glob import glob
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

_WORKFLOW_DEFINITIONS_DIR = Path(__file__).resolve().parent / "workflow_definitions"


BATCH_4_IMAGE_NAMES = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
BATCH_4_DETECTION_COUNTS = [0, 2, 0, 1]
SLICED_DETECTION_COUNTS = [
    0, 0, 0, 0,  # image 0
    0, 0, 1, 1,  # image 1 (slices 2,3)
    0, 0, 0, 0,  # image 2
    0, 0, 0, 0, 0, 1, 0, 0,  # image 3 (slice 5)
]


def _load_workflow_definition(name: str) -> dict:
    path = _WORKFLOW_DEFINITIONS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow definition file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _make_person_prediction() -> ObjectDetectionPrediction:
    return ObjectDetectionPrediction(
        x=100.0,
        y=100.0,
        width=50.0,
        height=50.0,
        confidence=0.9,
        class_id=0,
        **{"class": "person"},
    )


def make_mock_detection_responses(
    counts: List[int],
) -> callable:
    """Build a mock that returns one ObjectDetectionInferenceResponse per request image with len(predictions) == counts[i]."""

    def mock_fn(model_id: str, request: ObjectDetectionInferenceRequest):
        images = request.image if isinstance(request.image, list) else [request.image]
        n = len(images)
        if n != len(counts):
            raise ValueError(
                f"Mock expected {len(counts)} images but got {n}. "
                "Check batch/slice counts for this scenario."
            )
        responses = []
        for i in range(n):
            k = counts[i]
            preds = [_make_person_prediction() for _ in range(k)]
            responses.append(
                ObjectDetectionInferenceResponse(
                    image=InferenceResponseImage(width=640, height=480),
                    predictions=preds,
                )
            )
        return responses

    return mock_fn


@pytest.fixture
def model_manager() -> ModelManager:
    from inference.core.registries.roboflow import RoboflowModelRegistry
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    manager = ModelManager(model_registry=registry)

    return manager


def _run_workflow(
    workflow_definition: dict,
    runtime_parameters: dict,
    model_manager,
    detection_counts: List[int],
):
    init_params = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=init_params,
        max_concurrent_steps=1,
    )
    mock_fn = make_mock_detection_responses(detection_counts)
    with patch.object(ModelManager, "add_model"):
        with patch.object(
            ModelManager,
            "infer_from_request_sync",
            side_effect=mock_fn,
        ):
            return engine.run(runtime_parameters=runtime_parameters)


# ---------- Scenario 1: Batch of 4 images, detections at 1 and 3 ----------


def _batch_4_images():
    """Four images (same shape for non-sliced scenario)."""
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]


@pytest.mark.parametrize(
    "workflow_name,expected_at_1,expected_at_3",
    [
        (
            "with_email_message_params",
            "2 detection",
            "1 detection",
        ),
        (
            "without_email_message_params",
            "Detection(s) found",
            "Detection(s) found",
        ),
        (
            "with_image_names_and_email_message_params",
            ("2 detection", "img1.jpg"),
            ("1 detection", "img3.jpg"),
        ),
        (
            "with_image_names_and_without_email_message_params",
            ("img1.jpg", "detection"),
            ("img3.jpg", "detection"),
        ),
    ],
    ids=["with_email_message_params", "without_email_message_params", "with_image_names_and_email_message_params", "without_image_names_and_email_message_params"],
)
def test_scenario_1_email_messages(
    workflow_name: str,
    expected_at_1,
    expected_at_3,
    model_manager,
) -> None:
    """1) Batch of 4 images; images 1 and 3 have detections. Assert email message content (a-d)."""
    workflow = _load_workflow_definition(workflow_name)

    runtime = {
        "image": _batch_4_images(),
        "dry_run": True,
    }

    inputs = {inp["name"] for inp in workflow.get("inputs", [])}
    if "image_names" in inputs:
        runtime["image_names"] = BATCH_4_IMAGE_NAMES

    result = _run_workflow(
        workflow, runtime, model_manager, detection_counts=BATCH_4_DETECTION_COUNTS
    )

    assert len(result) == 4

    assert result[0].get("email_message") is None
    assert result[2].get("email_message") is None

    msg1 = str(result[1]["email_message"])
    msg3 = str(result[3]["email_message"])

    if isinstance(expected_at_1, tuple):
        for part in expected_at_1:
            assert part in msg1, f"Expected {part!r} in message: {msg1!r}"
        for part in expected_at_3:
            assert part in msg3, f"Expected {part!r} in message: {msg3!r}"
    else:
        assert expected_at_1 in msg1, f"Expected {expected_at_1!r} in: {msg1!r}"
        assert expected_at_3 in msg3, f"Expected {expected_at_3!r} in: {msg3!r}"


# ---------- Scenario 2: One email per slice with detections ----------

# Per-image slice indices with detections: image 1 has detections in slices 2,3; image 3 in slice 4 (0-based)
# So we get 3 emails total: result[1]["email_message"][2], [1][3], result[3]["email_message"][3]
SLICED_EMAIL_IMAGE_INDEX_AND_SLICE = [(1, 2), (1, 3), (3, 5)]


def _sliced_4_images():
    """Three images that slice into 4 slices each, one that slices into 8. With 640x640, 0.2 overlap, stride 512: 2x2 needs 1152; 2x4 needs 2176x1152."""
    # 1152x1152 -> 2x2 = 4 slices; 2176x1152 -> 4x2 = 8 slices
    img_4 = np.zeros((1152, 1152, 3), dtype=np.uint8)
    img_8 = np.zeros((1152, 2176, 3), dtype=np.uint8)
    return [img_4, img_4, img_4, img_8]


@pytest.mark.parametrize(
    "workflow_name,expected_in_message",
    [
        ("sliced_image_with_email_message_params", "1 detection"),
        ("sliced_image_without_email_message_params", "Detection(s) found"),
    ],
    ids=["sliced_image_with_email_message_params", "sliced_image_without_email_message_params"],
)
def test_scenario_2_sliced_email_messages(
    workflow_name: str,
    expected_in_message: str,
    model_manager,
) -> None:
    """2) Sliced (no stitch): one email per slice with detections. Output is 4 batch elements (per image), each email_message a list per slice."""
    workflow = _load_workflow_definition(workflow_name)

    runtime = {
        "image": _sliced_4_images(),
        "dry_run": True,
    }

    result = _run_workflow(
        workflow,
        runtime,
        model_manager,
        detection_counts=SLICED_DETECTION_COUNTS,
    )

    assert len(result) == 4, "4 input images"
    for image_idx, slice_idx in SLICED_EMAIL_IMAGE_INDEX_AND_SLICE:
        messages = result[image_idx].get("email_message")
        assert messages is not None
        assert isinstance(messages, list)
        msg = messages[slice_idx]
        assert msg is not None
        assert expected_in_message in str(msg)


# ---------- Scenario 3: enable_email gate ----------


def test_scenario_3_email_gate_enabled(
    model_manager,
) -> None:
    """3) enable_email=True: same as scenario 1 with count and image names."""
    workflow = _load_workflow_definition("with_email_gate_and_with_email_message_params")
    runtime = {
        "image": _batch_4_images(),
        "image_names": BATCH_4_IMAGE_NAMES,
        "dry_run": True,
        "enable_email": True,
    }

    result = _run_workflow(
        workflow, runtime, model_manager, detection_counts=BATCH_4_DETECTION_COUNTS
    )

    assert len(result) == 4
    assert result[0].get("email_message") is None
    assert result[2].get("email_message") is None
    msg1 = str(result[1]["email_message"])
    msg3 = str(result[3]["email_message"])
    assert "2 detection" in msg1 and "img1.jpg" in msg1
    assert "1 detection" in msg3 and "img3.jpg" in msg3


def test_scenario_3_email_gate_disabled(
    model_manager,
) -> None:
    """3) enable_email=False: email step never runs."""
    workflow = _load_workflow_definition("with_email_gate_and_without_email_message_params")
    runtime = {
        "image": _batch_4_images(),
        "image_names": BATCH_4_IMAGE_NAMES,
        "dry_run": True,
        "enable_email": False,
    }

    result = _run_workflow(
        workflow, runtime, model_manager, detection_counts=BATCH_4_DETECTION_COUNTS
    )

    assert len(result) == 4
    for i in range(4):
        assert result[i].get("email_message") is None


# ---------- Scenario 4: CSV created and saved ----------


def test_scenario_4_csv_sink(
    model_manager,
    empty_directory: str,
) -> None:
    """4) Batch of 4 images, only 1 and 3 have detections. CSV has 2 rows: num_detections 2 and 1."""
    workflow = _load_workflow_definition("with_csv_sink_and_with_detection_input")
    runtime = {
        "image": _batch_4_images(),
        "dry_run": False,
        "output_directory": empty_directory,
    }

    result = _run_workflow(
        workflow, runtime, model_manager, detection_counts=BATCH_4_DETECTION_COUNTS
    )

    assert len(result) == 4
    # Only last batch index gets the aggregated CSV message
    assert result[3].get("save_message") == "Data saved successfully"

    csv_files = glob(os.path.join(empty_directory, "detection_log_*.csv"))
    assert len(csv_files) >= 1
    df = pd.read_csv(csv_files[0])

    assert "num_detections" in df.columns
    assert "timestamp" in df.columns
    assert len(df) == 2
    assert df["num_detections"].tolist() == [2, 1]


def test_scenario_4_csv_sink_with_image_names(
    model_manager,
    empty_directory: str,
) -> None:
    """4) Same but workflow with image_names; CSV rows have correct image_name."""
    workflow = _load_workflow_definition("with_csv_sink_and_without_detection_input")
    runtime = {
        "image": _batch_4_images(),
        "image_names": BATCH_4_IMAGE_NAMES,
        "dry_run": False,
        "output_directory": empty_directory,
    }

    result = _run_workflow(
        workflow, runtime, model_manager, detection_counts=BATCH_4_DETECTION_COUNTS
    )

    assert len(result) == 4
    assert result[3].get("save_message") == "Data saved successfully"

    csv_files = glob(os.path.join(empty_directory, "detection_log_*.csv"))
    assert len(csv_files) >= 1
    df = pd.read_csv(csv_files[0])

    assert "image_name" in df.columns
    assert "timestamp" in df.columns
    assert len(df) == 2
    assert df["image_name"].tolist() == ["img1.jpg", "img3.jpg"]
