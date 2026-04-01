"""Integration tests for detection → continue_if → email/CSV workflows. Workflow definitions in ./workflow_definitions."""

import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import supervision as sv

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.fusion.detections_stitch.v1 import (
    DetectionsStitchBlockV1,
)
from inference.core.workflows.errors import (
    ControlFlowDefinitionError,
    StepInputLineageError,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

_WORKFLOW_DEFINITIONS_DIR = Path(__file__).resolve().parent / "workflow_definitions"

# Not testing the message interpolation here
SUCCESSFUL_EMAIL_MESSAGE_MOCK = "Notification sent successfully"

BATCH_4_IMAGE_NAMES = [
    "img0",
    "img1",
    "img2",
    "img3",
]

SLICED_NAMES = [
    # image 0
    "image_0_slice_0",
    "image_0_slice_1",
    "image_0_slice_2",
    "image_0_slice_3",
    # image 1
    "image_1_slice_0",
    "image_1_slice_1",
    "image_1_slice_2",
    "image_1_slice_3",
    # image 2
    "image_2_slice_0",
    "image_2_slice_1",
    "image_2_slice_2",
    "image_2_slice_3",
    # image 3
    "image_3_slice_0",
    "image_3_slice_1",
    "image_3_slice_2",
    "image_3_slice_3",
    "image_3_slice_4",
    "image_3_slice_5",
    "image_3_slice_6",
    "image_3_slice_7",
]

BATCH_4_DETECTION_COUNTS = [0, 2, 0, 1]
SLICED_DETECTION_COUNTS = [
    0,
    0,
    0,
    0,  # image 0
    0,
    0,
    1,
    1,  # image 1 (slices 2,3)
    0,
    0,
    0,
    0,  # image 2
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,  # image 3 (slice 5)
]


def _batch_4_images():
    """Four images (same shape for non-sliced scenario)."""
    return [
        np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )
        for _ in range(4)
    ]


def _sliced_4_images():
    """Three images that slice into 4 slices each, one into 8. 1152x1152 -> 2x2 = 4 slices;
    2176x1152 -> 4x2 = 8 slices. 640x640, 0.2 overlap, stride 512."""
    img_4 = np.zeros(
        (1152, 1152, 3),
        dtype=np.uint8,
    )
    img_8 = np.zeros(
        (1152, 2176, 3),
        dtype=np.uint8,
    )
    return [img_4, img_4, img_4, img_8]


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


def make_mock_detection_responses_per_model(
    model_id_to_counts: Dict[str, List[int]],
) -> callable:
    """Build a mock that returns different detection counts per model_id.

    Each value in model_id_to_counts is the full list of counts for all calls
    to that model in order (consumed sequentially per model_id).
    """

    # Mutable index per model_id for consuming counts
    model_id_index: Dict[str, int] = {mid: 0 for mid in model_id_to_counts}

    def mock_fn(model_id: str, request: ObjectDetectionInferenceRequest):
        if model_id not in model_id_to_counts:
            raise ValueError(
                f"Mock received unknown model_id={model_id!r}. "
                f"Known: {list(model_id_to_counts)}."
            )
        counts = model_id_to_counts[model_id]
        images = request.image if isinstance(request.image, list) else [request.image]
        n = len(images)
        start = model_id_index[model_id]
        end = start + n
        if end > len(counts):
            raise ValueError(
                f"Mock for model_id={model_id!r} expected at least {end} counts "
                f"but only {len(counts)} defined. Check detection_counts for this scenario."
            )
        chunk = counts[start:end]
        model_id_index[model_id] = end
        responses = []
        for k in chunk:
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
    detection_counts_per_model: Dict[str, List[int]],
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

    mock_fn = make_mock_detection_responses_per_model(detection_counts_per_model)

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


@patch(
    "inference.core.workflows.core_steps.sinks.email_notification.v2.send_email_via_roboflow_proxy"
)
@pytest.mark.parametrize(
    "image_gen_fn,\
    names,\
    detection_counts_per_model,\
    enable_email,\
    workflow_name,\
    expected_call_count,\
    expected_receiver_email,\
    expected_subject,\
    expected_message_parameters,\
    expected_result",
    [
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_email_message_params",
            2,
            "noreply@example.com",
            "Detections found",
            (
                {"num_detections": 2},
                {"num_detections": 1},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "without_email_message_params",
            2,
            "noreply@example.com",
            "Detections found",
            (
                {},
                {},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_image_names_and_email_message_params",
            2,
            "noreply@example.com",
            "Detections found",
            (
                {"num_detections": 2, "name": "img1"},
                {"num_detections": 1, "name": "img3"},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_image_names_and_without_email_message_params",
            2,
            "noreply@example.com",
            "Detections found",
            (
                {"name": "img1"},
                {"name": "img3"},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "sliced_image_with_email_message_params",
            3,
            "noreply@example.com",
            "Detections found",
            (
                {"num_detections": 1},
                {"num_detections": 1},
                {"num_detections": 1},
            ),
            (
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                    ]
                },
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        None,
                        None,
                    ]
                },
            ),
        ),
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "sliced_image_without_email_message_params",
            3,
            "noreply@example.com",
            "Detections found",
            (
                {},
                {},
                {},
            ),
            (
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                    ]
                },
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        None,
                        None,
                    ]
                },
            ),
        ),
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "sliced_image_with_email_message_params_and_area_size_step",
            3,
            "noreply@example.com",
            "Detections found",
            (
                {"num_detections": 1, "area_converted": 2500},
                {"num_detections": 1, "area_converted": 2500},
                {"num_detections": 1, "area_converted": 2500},
            ),
            (
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                    ]
                },
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        None,
                        None,
                    ]
                },
            ),
        ),
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "sliced_image_without_email_message_params_and_area_size_step",
            3,
            "noreply@example.com",
            "Detections found",
            (
                {"area_converted": 2500},
                {"area_converted": 2500},
                {"area_converted": 2500},
            ),
            (
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                    ]
                },
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        None,
                        None,
                    ]
                },
            ),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_email_gate_and_with_email_message_params",
            2,
            "noreply@example.com",
            "Detections found",
            (
                {"num_detections": 2},
                {"num_detections": 1},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_email_gate_and_without_email_message_params",
            2,
            "noreply@example.com",
            "Detections found",
            (
                {},
                {},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            False,
            "with_email_gate_and_without_email_message_params",
            0,
            "noreply@example.com",
            "Detections found",
            ({},),
            (
                {"email_message": None},
                {"email_message": None},
                {"email_message": None},
                {"email_message": None},
            ),
        ),
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "with_detection_collapse_right_after_slice",  # after the dim collapse we are dim=1
            4,  # In this scenario the continue_if step counts the number of slices for each image, so 4 calls to the email step
            "noreply@example.com",
            "Detections found",
            (
                {"num_slices": 4},
                {"num_slices": 4},
                {"num_slices": 4},
                {"num_slices": 8},
            ),
            (  # In this scenario the email step is called 4 times, once for each image, as each image has at least one slice
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "with_detection_collapse_right_after_slice_with_agg_operation",
            2,  # The continue-if correctly checks the number of detections for each image (given slices of that image)
            "noreply@example.com",
            "Detections found",
            (  # The operations of counting the detection are done after receiving the params, so here we get the slices
                {"num_slices": 4},
                {"num_slices": 8},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "with_detection_collapse_right_after_slice_with_agg_operation_without_message_params",
            2,  # The continue-if correctly checks the number of detections for each image (given slices of that image)
            "noreply@example.com",
            "Detections found",
            (
                {},
                {},
            ),
            (
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_detection_collapse_right_after_detect_with_agg_operation",
            1,  # The continue-if correctly checks the total number of detections in the batch
            "noreply@example.com",
            "Detections found",
            (  # The operations of counting the detection are done after receiving the params, so here we get the size of the batch
                {"num_batch_detections": 4},
            ),
            ({"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_detection_collapse_right_after_detect_with_agg_operation_without_message_params",
            1,  # The continue-if correctly checks the total number of detections in the batch
            "noreply@example.com",
            "Detections found",
            ({},),
            ({"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            True,
            "with_detection_collapse_after_continue_if",
            1,  # We aggregated the detection lists after continue_if
            "noreply@example.com",
            "Detections found",
            ({"num_batch_filtered_detections": 2},),  # Only two images had detections
            ({"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},),
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": [3, 0, 1, 2]},
            True,
            "with_two_continue_if",
            1,
            "noreply@example.com",
            "Detections found",
            ({},),
            (
                {"email_message": None},
                {"email_message": None},
                {"email_message": None},
                {"email_message": SUCCESSFUL_EMAIL_MESSAGE_MOCK},
            ),
        ),
        (
            _sliced_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            True,
            "with_two_continue_if_different_control_flow_lineage",
            3,  # The deepest control-flow-lineage is used, thus we are masking up to the slice level
            "noreply@example.com",
            "Detections found",
            (
                {},
                {},
                {},
            ),
            (
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                    ]
                },
                {"email_message": [None, None, None, None]},
                {
                    "email_message": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        SUCCESSFUL_EMAIL_MESSAGE_MOCK,
                        None,
                        None,
                    ]
                },
            ),
        ),
    ],
    ids=[
        "with_email_message_params",
        "without_email_message_params",
        "with_image_names_and_email_message_params",
        "without_image_names_and_email_message_params",
        "sliced_image_with_email_message_params",
        "sliced_image_without_email_message_params",
        "sliced_image_with_email_message_params_and_area_size_step",
        "sliced_image_without_email_message_params_and_area_size_step",
        "with_email_gate_and_with_email_message_params",
        "with_email_gate_and_without_email_message_params",
        "with_email_gate_and_without_email_message_params_and_email_disabled",
        "with_detection_collapse_right_after_slice",
        "with_detection_collapse_right_after_slice_with_agg_operation",
        "with_detection_collapse_right_after_slice_with_agg_operation_without_message_params",
        "with_detection_collapse_right_after_detect_with_agg_operation",
        "with_detection_collapse_right_after_detect_with_agg_operation_without_message_params",
        "with_detection_collapse_after_continue_if",
        "with_two_continue_if",
        "with_two_continue_if_different_control_flow_lineage",
    ],
)
def test_properly_running_side_effect_step_and_returning_results_in_different_data_lineage_control_lineage_scenarios(
    send_email_mock,
    image_gen_fn: callable,
    names: List[str],
    detection_counts_per_model: Dict[str, List[int]],
    enable_email: bool,
    workflow_name: str,
    expected_call_count: int,
    expected_receiver_email: str,
    expected_subject: str,
    expected_message_parameters: tuple[dict[str, str | int], dict[str, str | int]],
    expected_result: tuple[dict[str, str | int], dict[str, str | int]],
    model_manager,
) -> None:
    send_email_mock.return_value = (False, "Notification sent successfully")
    workflow_definition = _load_workflow_definition(workflow_name)

    runtime_parameters = {"image": image_gen_fn()}
    inputs = {inp["name"] for inp in workflow_definition.get("inputs", [])}
    if "names" in inputs:
        runtime_parameters["names"] = names
    if "enable_email" in inputs:
        runtime_parameters["enable_email"] = enable_email

    result = _run_workflow(
        workflow_definition,
        runtime_parameters,
        model_manager,
        detection_counts_per_model=detection_counts_per_model,
    )

    assert send_email_mock.call_count == expected_call_count
    for i, call in enumerate(send_email_mock.call_args_list):
        assert call.kwargs["receiver_email"] == [expected_receiver_email]
        assert call.kwargs["subject"] == expected_subject

        expected_params = expected_message_parameters[i]
        assert len(call.kwargs["message_parameters"]) == len(expected_params)

        for param_name, param_value in expected_params.items():
            actual = call.kwargs["message_parameters"][param_name]

            if param_name == "num_detections":
                assert isinstance(actual, sv.Detections)
                assert len(actual) == param_value
                continue

            if param_name in [
                "num_slices",
                "num_batch_detections",
                "num_batch_filtered_detections",
            ]:
                assert isinstance(actual, list)
                assert len(actual) == param_value
                continue

            if param_name == "area_converted":
                assert actual["area_converted"] == param_value
                continue

            assert actual == param_value

    assert len(result) == len(expected_result)
    for i, result in enumerate(result):
        assert result.get("email_message") == expected_result[i].get("email_message")


@patch(
    "inference.core.workflows.core_steps.sinks.email_notification.v2.send_email_via_roboflow_proxy"
)
@pytest.mark.parametrize(
    "image_gen_fn,\
    names,\
    detection_counts_per_model,\
    workflow_name",
    [
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            "sliced_image_with_email_message_params_with_slice_names",
        ),
    ],
    ids=[
        "sliced_image_with_email_message_params_with_slice_names",
    ],
)
def test_scenario_raises_step_input_lineage_error(
    send_email_mock,
    image_gen_fn: callable,
    names: List[str],
    detection_counts_per_model: Dict[str, List[int]],
    workflow_name: str,
    model_manager,
) -> None:
    send_email_mock.return_value = (False, "This will not send an email")
    workflow_definition = _load_workflow_definition(workflow_name)

    runtime_parameters = {"image": image_gen_fn()}
    inputs = {inp["name"] for inp in workflow_definition.get("inputs", [])}
    if "names" in inputs:
        runtime_parameters["names"] = names

    with pytest.raises(StepInputLineageError):
        _run_workflow(
            workflow_definition,
            runtime_parameters,
            model_manager,
            detection_counts_per_model=detection_counts_per_model,
        )

    assert send_email_mock.call_count == 0


@patch(
    "inference.core.workflows.core_steps.sinks.email_notification.v2.send_email_via_roboflow_proxy"
)
@pytest.mark.parametrize(
    "image_gen_fn,\
    names,\
    detection_counts_per_model,\
    workflow_name",
    [
        (
            _sliced_4_images,
            SLICED_NAMES,
            {"yolov8n-640": SLICED_DETECTION_COUNTS},
            "sliced_image_without_email_message_params_with_slice_names",
        ),
    ],
    ids=[
        "sliced_image_without_email_message_params_with_slice_names",
    ],
)
def test_scenario_raises_control_flow_definition_error(
    send_email_mock,
    image_gen_fn: callable,
    names: List[str],
    detection_counts_per_model: Dict[str, List[int]],
    workflow_name: str,
    model_manager,
) -> None:
    send_email_mock.return_value = (False, "This will not send an email")
    workflow_definition = _load_workflow_definition(workflow_name)

    runtime_parameters = {"image": image_gen_fn()}
    inputs = {inp["name"] for inp in workflow_definition.get("inputs", [])}
    if "names" in inputs:
        runtime_parameters["names"] = names

    with pytest.raises(ControlFlowDefinitionError):
        _run_workflow(
            workflow_definition,
            runtime_parameters,
            model_manager,
            detection_counts_per_model=detection_counts_per_model,
        )

    assert send_email_mock.call_count == 0


@pytest.mark.parametrize(
    "run_scalar_step, expect_result",
    [
        (True, "foobar"),
        (False, None),
    ],
    ids=["run_scalar_step_true", "run_scalar_step_false"],
)
@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_control_flow_lineage_using_workflow_with_scalar_only_block_parses_and_runs(
    get_plugin_modules_mock: MagicMock,
    run_scalar_step: bool,
    expect_result: str,
) -> None:
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.scalar_only_block_plugin",
    ]
    execution_engine = ExecutionEngine.init(
        workflow_definition=_load_workflow_definition("with_scalar_only_step"),
        init_parameters={
            "workflows_core.model_manager": None,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    result = execution_engine.run(
        runtime_parameters={"run_scalar_step": run_scalar_step},
    )
    assert len(result) == 1
    assert result[0]["result"] == expect_result


@pytest.mark.parametrize(
    "workflow_name, \
    names, \
    expect_result",
    [
        (
            "with_scalar_only_step_getting_batch_data",
            BATCH_4_IMAGE_NAMES,
            BATCH_4_IMAGE_NAMES,
        ),
        (
            "with_scalar_only_step_getting_batch_data_only_control_flow_lineage",
            ["img1", "img2", "img3", "not"],
            ["foobar"] * 3,
        ),
        (
            "with_scalar_only_step_getting_batch_data_only_control_flow_lineage",
            BATCH_4_IMAGE_NAMES,
            ["foobar"] * len(BATCH_4_IMAGE_NAMES),
        ),
    ],
    ids=[
        "with_scalar_only_step_getting_batch_data",
        "with_scalar_only_step_getting_batch_data_only_control_flow_lineage_with_not_all_elements_meeting_requirements",
        "with_scalar_only_step_getting_batch_data_only_control_flow_lineage",
    ],
)
@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_control_flow_lineage_using_workflow_with_scalar_only_block_that_gets_batch_data(
    get_plugin_modules_mock: MagicMock,
    workflow_name: str,
    names: List[str],
    expect_result: str,
) -> None:
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.scalar_only_block_plugin",
    ]
    execution_engine = ExecutionEngine.init(
        workflow_definition=_load_workflow_definition(workflow_name),
        init_parameters={
            "workflows_core.model_manager": None,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    result = execution_engine.run(
        runtime_parameters={"names": names},
    )
    assert len(result) == len(names)
    for i in range(len(expect_result)):
        assert result[i]["result"] == expect_result[i]


@pytest.mark.parametrize(
    "workflow_name, \
    names, \
    expect_result",
    [
        (
            "with_batch_only_step_with_batch_data",
            BATCH_4_IMAGE_NAMES,
            BATCH_4_IMAGE_NAMES,
        ),
    ],
    ids=[
        "with_batch_only_step_with_batch_data",
    ],
)
@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_control_flow_lineage_using_workflow_with_batch_only_block_that_gets_batch_data(
    get_plugin_modules_mock: MagicMock,
    workflow_name: str,
    names: List[str],
    expect_result: str,
) -> None:
    """Workflow with batch_only_echo (optional batch-only input) wired to $inputs.names runs and echoes batch."""
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.batch_only_block_plugin",
    ]
    execution_engine = ExecutionEngine.init(
        workflow_definition=_load_workflow_definition(workflow_name),
        init_parameters={
            "workflows_core.model_manager": None,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    result = execution_engine.run(
        runtime_parameters={"names": names},
    )
    assert len(result) == len(expect_result)
    for i in range(len(expect_result)):
        assert result[i]["result"] == expect_result[i]


@pytest.mark.parametrize(
    "image_gen_fn,\
    names,\
    detection_counts_per_model,\
    workflow_name,\
    expected_results,\
    expected_num_files,\
    expected_columns,\
    expected_num_rows,\
    expected_names,\
    expected_num_detections",
    [
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            "with_csv_sink_and_with_detection_input",
            [
                {"save_message": None},
                {"save_message": None},
                {"save_message": None},
                {"save_message": "Data saved successfully"},
            ],
            1,
            ["num_detections", "name", "timestamp"],
            2,
            ["img1", "img3"],
            [2, 1],
        ),
        (
            _batch_4_images,
            BATCH_4_IMAGE_NAMES,
            {"yolov8n-640": BATCH_4_DETECTION_COUNTS},
            "with_csv_sink_and_without_detection_input",
            [
                {"save_message": None},
                {"save_message": None},
                {"save_message": None},
                {"save_message": "Data saved successfully"},
            ],
            1,
            ["name", "timestamp"],
            2,
            ["img1", "img3"],
            [2, 1],
        ),
    ],
    ids=[
        "with_csv_sink_and_with_detection_input",
        "with_csv_sink_and_without_detection_input",
    ],
)
def test_control_flow_lineage_using_workflow_with_csv_sink_and_detection_input(
    image_gen_fn: callable,
    names: List[str],
    detection_counts_per_model: Dict[str, List[int]],
    workflow_name: str,
    expected_results: List[dict],
    expected_num_files: int,
    expected_columns: List[str],
    expected_num_rows: int,
    expected_names: List[str],
    expected_num_detections: List[int],
    model_manager,
    empty_directory,
) -> None:
    workflow_definition = _load_workflow_definition(workflow_name)

    runtime_parameters = {
        "image": image_gen_fn(),
        "output_directory": empty_directory,
    }

    inputs = {inp["name"] for inp in workflow_definition.get("inputs", [])}
    if "names" in inputs:
        runtime_parameters["names"] = names

    result = _run_workflow(
        workflow_definition,
        runtime_parameters,
        model_manager,
        detection_counts_per_model=detection_counts_per_model,
    )
    assert result == expected_results

    csv_files = glob(os.path.join(empty_directory, "detection_log_*.csv"))
    assert len(csv_files) == expected_num_files
    df = pd.read_csv(csv_files[0])

    assert set(df.columns) == set(expected_columns)
    assert len(df) == expected_num_rows

    if "num_detections" in expected_columns:
        assert df["num_detections"].tolist() == expected_num_detections
    if "name" in expected_columns:
        assert df["name"].tolist() == expected_names


@pytest.mark.parametrize(
    "image_gen_fn,\
    detection_counts_per_model,\
    workflow_name,\
    expected_call_count, \
    expected_results",
    [
        (
            _sliced_4_images,
            {
                "yolov8n-640": SLICED_DETECTION_COUNTS,
                "yolov8s-640": BATCH_4_DETECTION_COUNTS,
            },
            "with_two_continue_if_data_lineage_present",
            2,
            [
                {"stitched_predictions": (type(None),)},
                {"stitched_predictions": (sv.Detections, 2)},
                {"stitched_predictions": (type(None),)},
                {"stitched_predictions": (sv.Detections, 1)},
            ],
        ),
    ],
    ids=[
        "with_two_continue_if_data_lineage_present",
    ],
)
def test_side_effect_step_with_data_lineage_and_continue_if_zero_calls(
    image_gen_fn: callable,
    detection_counts_per_model: Dict[str, List[int]],
    workflow_name: str,
    expected_call_count: int,
    expected_results: List[dict],
    model_manager,
) -> None:
    workflow_definition = _load_workflow_definition(workflow_name)
    runtime_parameters = {"image": image_gen_fn()}

    stitch_run_call_count = []
    real_run = DetectionsStitchBlockV1.run

    def counting_run(self, *args, **kwargs):
        stitch_run_call_count.append(1)
        return real_run(self, *args, **kwargs)

    with patch.object(DetectionsStitchBlockV1, "run", counting_run):
        result = _run_workflow(
            workflow_definition,
            runtime_parameters,
            model_manager,
            detection_counts_per_model=detection_counts_per_model,
        )

    assert len(stitch_run_call_count) == expected_call_count, (
        f"DetectionsStitchBlockV1.run should be called {expected_call_count} times, "
        f"was {len(stitch_run_call_count)}"
    )
    assert len(result) == len(expected_results)
    for i, item in enumerate(result):
        assert isinstance(
            item["stitched_predictions"],
            expected_results[i]["stitched_predictions"][0],
        )

        if isinstance(item["stitched_predictions"], sv.Detections):
            assert (
                len(item["stitched_predictions"])
                == expected_results[i]["stitched_predictions"][1]
            )
