"""Integration tests for detection → continue_if → email/CSV workflows. Workflow definitions in ./workflow_definitions."""

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

    with patch.object(
        ModelManager,
        "add_model",
    ):
        with patch.object(
            ModelManager,
            "infer_from_request_sync",
            side_effect=mock_fn,
        ):
            return engine.run(
                runtime_parameters=runtime_parameters,
            )


def _batch_4_images():
    """Four images (same shape for non-sliced scenario)."""
    return [
        np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )
        for _ in range(4)
    ]


@patch(
    "inference.core.workflows.core_steps.sinks.email_notification.v2.send_email_via_roboflow_proxy"
)
@pytest.mark.parametrize(
    "workflow_name,receiver_email,subject,expected_message_parameters",
    [
        # (
        #     "with_email_message_params",
        #     "noreply@example.com",
        #     "Detections found",
        #     ({"num_detections": 2}, {"num_detections": 1}),
        # ),
        (
            "without_email_message_params",
            "noreply@example.com",
            "Detections found",
            ({}, {}),
        ),
        # (
        #     "with_image_names_and_email_message_params",
        #     "noreply@example.com",
        #     "Detections found",
        #     ({"num_detections": 2, "image_name": "img1.jpg"}, {"num_detections": 1, "image_name": "img3.jpg"}),
        # ),
        # (
        #     "with_image_names_and_without_email_message_params",
        #     "noreply@example.com",
        #     "Detections found",
        #     ({"image_name": "img1.jpg"}, {"image_name": "img3.jpg"}),
        # ),
    ],
    ids=[
        # "with_email_message_params",
        "without_email_message_params",
        # "with_image_names_and_email_message_params",
        # "without_image_names_and_email_message_params",
    ],
)
def test_scenario_1(
    send_email_mock,
    workflow_name: str,
    receiver_email: str,
    subject: str,
    expected_message_parameters: tuple[dict[str, str | int], dict[str, str | int]],
    model_manager,
) -> None:
    """Batch of 4 images; images 1 and 3 have detections (2 and 1 respectively).
    Assert email message content and that the send path was invoked with expected arguments."""
    send_email_mock.return_value = (False, "Notification sent successfully")
    workflow_definition = _load_workflow_definition(workflow_name)

    runtime_parameters = {"image": _batch_4_images()}
    inputs = {inp["name"] for inp in workflow_definition.get("inputs", [])}
    if "image_names" in inputs:
        runtime_parameters["image_names"] = BATCH_4_IMAGE_NAMES

    result = _run_workflow(
        workflow_definition,
        runtime_parameters,
        model_manager,
        detection_counts=BATCH_4_DETECTION_COUNTS,
    )

    assert send_email_mock.call_count == 2
    for i, call in enumerate(send_email_mock.call_args_list):
        assert call.kwargs["receiver_email"] == [receiver_email]
        assert call.kwargs["subject"] == subject

        params = expected_message_parameters[i]

        if not params:
            assert call.kwargs["message_parameters"] == {}
            continue

        for param_name, param_value in params.items():
            if param_name == "num_detections":
                    assert len(call.kwargs["message_parameters"][param_name]) == param_value
            else:
                assert call.kwargs["message_parameters"][param_name] == param_value


    assert len(result) == 4
    assert result[0].get("email_message") is None
    assert result[2].get("email_message") is None
    assert result[1].get("email_message") is not None
    assert result[3].get("email_message") is not None



# SLICED_EMAIL_IMAGE_INDEX_AND_SLICE = [
#     (1, 2),
#     (1, 3),
#     (3, 5),
# ]


# def _sliced_4_images():
#     """Three images that slice into 4 slices each, one into 8. 1152x1152 -> 2x2 = 4 slices;
#     2176x1152 -> 4x2 = 8 slices. 640x640, 0.2 overlap, stride 512."""
#     img_4 = np.zeros(
#         (1152, 1152, 3),
#         dtype=np.uint8,
#     )
#     img_8 = np.zeros(
#         (1152, 2176, 3),
#         dtype=np.uint8,
#     )
#     return [img_4, img_4, img_4, img_8]


# @patch(
#     "inference.core.workflows.core_steps.sinks.email_notification.v2.send_email_via_roboflow_proxy"
# )
# @pytest.mark.parametrize(
#     "workflow_name,expected_in_message",
#     [
#         (
#             "sliced_image_with_email_message_params",
#             "1 detection",
#         ),
#         (
#             "sliced_image_without_email_message_params",
#             "Detection(s) found",
#         ),
#     ],
#     ids=[
#         "sliced_image_with_email_message_params",
#         "sliced_image_without_email_message_params",
#     ],
# )
# def test_scenario_2_sliced_email_messages(
#     send_email_mock,
#     workflow_name: str,
#     expected_in_message: str,
#     model_manager,
# ) -> None:
#     """Sliced (no stitch): image_slicer -> detection -> continue_if -> email per slice.
#     First 3 images -> 4 slices each, last -> 8 (20 total). Detections in slice indices 6, 7, 16 only.
#     One email per slice with detections; (a) and (b) only (no image_names).
#     Per-image slice indices with detections: image 1 in slices 2,3; image 3 in slice 5 (0-based).
#     Output: 4 batch elements, each email_message a list per slice."""
#     send_email_mock.return_value = (False, "Notification sent successfully")
#     workflow = _load_workflow_definition(workflow_name)
#     runtime = {"image": _sliced_4_images()}

#     result = _run_workflow(
#         workflow,
#         runtime,
#         model_manager,
#         detection_counts=SLICED_DETECTION_COUNTS,
#     )

#     assert len(result) == 4, "4 input images"

#     for image_idx, slice_idx in SLICED_EMAIL_IMAGE_INDEX_AND_SLICE:
#         messages = result[image_idx].get("email_message")
#         assert messages is not None
#         assert isinstance(messages, list)
#         msg = messages[slice_idx]
#         assert msg is not None
#         assert expected_in_message in str(msg)

#     assert send_email_mock.call_count >= 1


# @patch(
#     "inference.core.workflows.core_steps.sinks.email_notification.v2.send_email_via_roboflow_proxy"
# )
# def test_scenario_3_email_gate_enabled(
#     send_email_mock,
#     model_manager,
# ) -> None:
#     """Like scenario 1 with enable_email gate True: count and image names in email."""
#     send_email_mock.return_value = (False, "Notification sent successfully")
#     workflow = _load_workflow_definition("with_email_gate_and_with_email_message_params")
#     runtime = {
#         "image": _batch_4_images(),
#         "image_names": BATCH_4_IMAGE_NAMES,
#         "enable_email": True,
#     }

#     result = _run_workflow(
#         workflow,
#         runtime,
#         model_manager,
#         detection_counts=BATCH_4_DETECTION_COUNTS,
#     )

#     assert len(result) == 4
#     assert result[0].get("email_message") is None
#     assert result[2].get("email_message") is None

#     msg1 = str(result[1]["email_message"])
#     msg3 = str(result[3]["email_message"])
#     assert "2 detection" in msg1 and "img1.jpg" in msg1
#     assert "1 detection" in msg3 and "img3.jpg" in msg3


# @patch(
#     "inference.core.workflows.core_steps.sinks.email_notification.v2.send_email_via_roboflow_proxy"
# )
# def test_scenario_3_email_gate_disabled(
#     send_email_mock,
#     model_manager,
# ) -> None:
#     """enable_email=False: email step never runs; no email_message and no send calls."""
#     workflow = _load_workflow_definition("with_email_gate_and_without_email_message_params")

#     runtime = {
#         "image": _batch_4_images(),
#         "image_names": BATCH_4_IMAGE_NAMES,
#         "enable_email": False,
#     }

#     result = _run_workflow(
#         workflow,
#         runtime,
#         model_manager,
#         detection_counts=BATCH_4_DETECTION_COUNTS,
#     )

#     assert len(result) == 4
#     for i in range(4):
#         assert result[i].get("email_message") is None
#     assert send_email_mock.call_count == 0


# def test_scenario_4_csv_sink(
#     model_manager,
#     empty_directory: str,
# ) -> None:
#     """Like scenario 1 but CSV sink. Batch of 4 images; only indices 1 and 3 have detections.
#     CSV has 2 rows (num_detections 2 and 1). Only the last batch index gets the aggregated save_message."""
#     workflow = _load_workflow_definition("with_csv_sink_and_with_detection_input")

#     runtime = {
#         "image": _batch_4_images(),
#         "output_directory": empty_directory,
#     }

#     result = _run_workflow(
#         workflow,
#         runtime,
#         model_manager,
#         detection_counts=BATCH_4_DETECTION_COUNTS,
#     )

#     assert len(result) == 4
#     assert result[3].get("save_message") == "Data saved successfully"

#     csv_files = glob(os.path.join(empty_directory, "detection_log_*.csv"))
#     assert len(csv_files) >= 1
#     df = pd.read_csv(csv_files[0])

#     assert "num_detections" in df.columns
#     assert "timestamp" in df.columns
#     assert len(df) == 2
#     assert df["num_detections"].tolist() == [2, 1]


# def test_scenario_4_csv_sink_with_image_names(
#     model_manager,
#     empty_directory: str,
# ) -> None:
#     """Like scenario 4 but workflow has image_names input; CSV rows have correct image_name (img1.jpg, img3.jpg)."""
#     workflow = _load_workflow_definition("with_csv_sink_and_without_detection_input")

#     runtime = {
#         "image": _batch_4_images(),
#         "image_names": BATCH_4_IMAGE_NAMES,
#         "output_directory": empty_directory,
#     }

#     result = _run_workflow(
#         workflow,
#         runtime,
#         model_manager,
#         detection_counts=BATCH_4_DETECTION_COUNTS,
#     )

#     assert len(result) == 4
#     assert result[3].get("save_message") == "Data saved successfully"

#     csv_files = glob(os.path.join(empty_directory, "detection_log_*.csv"))
#     assert len(csv_files) >= 1
#     df = pd.read_csv(csv_files[0])

#     assert "image_name" in df.columns
#     assert "timestamp" in df.columns
#     assert len(df) == 2
#     assert df["image_name"].tolist() == ["img1.jpg", "img3.jpg"]
