"""Ordering-correctness tests for remote stream pipelining on a real
ExecutionEngine.

The workflow is a remote object-detection model feeding a real ByteTrack
step. The mocked HTTP client answers with a distinct canned prediction per
frame and sleeps longer for earlier frames, so remote responses complete in
reverse submission order. Emissions must still come out strictly in frame
order, each with its own frame's detections, and the tracker must behave
exactly as in a sequential (depth 1) run.

Lives under unit_tests: everything remote is mocked (MagicMock model manager,
patched `InferenceHTTPClient`), unlike tests/workflows/integration_tests
which load real models.
"""

import base64
import time
from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock

import cv2
import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.model_handlers.workflows import (
    LookaheadPipelinedWorkflowRunner,
    WorkflowRunner,
    wrap_workflow_runner_for_stream_pipeline,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.object_detection import (
    v3 as object_detection_v3,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine

FRAMES_NUMBER = 8
IMAGE_SIZE = 240
PIXEL_VALUE_STEP = 25
STATIC_OBJECT_FIRST_FRAME = 4

WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "workspace/model/1",
        },
        {
            "type": "roboflow_core/byte_tracker@v3",
            "name": "tracker",
            "image": "$inputs.image",
            "detections": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "tracked_detections",
            "selector": "$steps.tracker.tracked_detections",
        }
    ],
}


def _make_frame(frame_id: int) -> VideoFrame:
    # Each frame carries a distinct uniform pixel value so the mocked HTTP
    # client can recover the frame index from the base64 payload it receives.
    return VideoFrame(
        image=np.full(
            (IMAGE_SIZE, IMAGE_SIZE, 3), frame_id * PIXEL_VALUE_STEP, dtype=np.uint8
        ),
        frame_id=frame_id,
        frame_timestamp=datetime.fromtimestamp(frame_id + 1),
        fps=30.0,
        measured_fps=None,
        source_id=0,
        comes_from_video_file=True,
    )


def _frame_index_of(base64_image: str) -> int:
    image = cv2.imdecode(
        np.frombuffer(base64.b64decode(base64_image), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    return int(round(float(image.mean()) / PIXEL_VALUE_STEP))


def _canned_box(cx: int, cy: int, detection_id: str) -> dict:
    return {
        "width": 20,
        "height": 20,
        "x": cx,
        "y": cy,
        "confidence": 0.9,
        "class_id": 0,
        "class": "car",
        "detection_id": detection_id,
        "parent_id": "image",
    }


def _moving_box_xyxy(frame_index: int) -> List[float]:
    cx = 40 + 2 * frame_index
    return [cx - 10, 90, cx + 10, 110]


def _canned_prediction(frame_index: int) -> dict:
    # One object drifting 2px per frame (uniquely identifying the frame by
    # its coordinates) plus a second, static object appearing mid-stream.
    boxes = [_canned_box(40 + 2 * frame_index, 100, f"moving-{frame_index}")]
    if frame_index >= STATIC_OBJECT_FIRST_FRAME:
        boxes.append(_canned_box(180, 180, f"static-{frame_index}"))
    return {
        "predictions": boxes,
        "image": {"width": IMAGE_SIZE, "height": IMAGE_SIZE},
        "time": 0.1,
    }


def _patch_remote_http_client(
    monkeypatch,
    delays: Optional[List[float]],
    completion_order: Optional[List[int]] = None,
) -> None:
    def _infer(inference_input, model_id):
        frame_index = _frame_index_of(inference_input[0])
        if delays is not None:
            time.sleep(delays[frame_index])
        if completion_order is not None:
            completion_order.append(frame_index)
        return _canned_prediction(frame_index)

    mock_client = MagicMock()
    mock_client.infer.side_effect = _infer
    monkeypatch.setattr(
        object_detection_v3,
        "InferenceHTTPClient",
        MagicMock(return_value=mock_client),
    )
    monkeypatch.setattr(object_detection_v3, "InferenceConfiguration", MagicMock())


def _init_wrapped_runner(monkeypatch, pipeline_depth: int):
    monkeypatch.setattr(
        object_detection_v3,
        "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH",
        pipeline_depth,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters={
            "workflows_core.model_manager": MagicMock(),
            "workflows_core.api_key": "key",
            "workflows_core.step_execution_mode": StepExecutionMode.REMOTE,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=execution_engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=execution_engine,
    )
    return execution_engine, runner


def test_remote_stream_pipelining_emits_in_frame_order_with_tracker_parity(
    monkeypatch,
) -> None:
    # given - remote responses for early frames land LAST: frame 0 is the
    # slowest, so with 4 requests in flight completion order is reversed
    completion_order: List[int] = []
    delays = [0.08, 0.06, 0.04, 0.02, 0.015, 0.01, 0.005, 0.002]
    _patch_remote_http_client(
        monkeypatch, delays=delays, completion_order=completion_order
    )
    execution_engine, runner = _init_wrapped_runner(monkeypatch, pipeline_depth=4)
    assert isinstance(runner, LookaheadPipelinedWorkflowRunner)
    model_block = execution_engine._engine._compiled_workflow.steps["model"].step

    try:
        # when
        emissions = []
        for frame_id in range(FRAMES_NUMBER):
            result = runner([_make_frame(frame_id)])
            if result is not None:
                emissions.append(result)
        flushed_results = runner.flush()
        emissions.extend(flushed_results or [])

        # then - the delays actually reversed completion of the in-flight batch
        assert completion_order.index(3) < completion_order.index(0)

        # then - every frame emitted exactly once, in strictly ascending order
        emitted_frame_ids = [
            emission.video_frames[0].frame_id for emission in emissions
        ]
        assert emitted_frame_ids == list(range(FRAMES_NUMBER))

        # then - each emission carries its own frame's detections
        for frame_index, emission in enumerate(emissions):
            tracked_detections = emission.predictions[0]["tracked_detections"]
            canned_xyxy = [_moving_box_xyxy(frame_index)] + (
                [[170, 170, 190, 190]]
                if frame_index >= STATIC_OBJECT_FIRST_FRAME
                else []
            )
            assert tracked_detections.xyxy[0].tolist() == _moving_box_xyxy(frame_index)
            for tracked_xyxy in tracked_detections.xyxy.tolist():
                assert tracked_xyxy in canned_xyxy

        # then - the pipelined block was engaged
        assert model_block._remote_pipeline is not None
    finally:
        runner.close()
    assert model_block._remote_pipeline is None

    # given - a sequential reference run: same canned responses, no delays,
    # pipelining disabled, fresh engine (fresh tracker state)
    _patch_remote_http_client(monkeypatch, delays=None)
    _, sequential_runner = _init_wrapped_runner(monkeypatch, pipeline_depth=1)
    assert isinstance(sequential_runner, WorkflowRunner)

    # when
    sequential_results = [
        sequential_runner([_make_frame(frame_id)]) for frame_id in range(FRAMES_NUMBER)
    ]

    # then - tracked boxes and tracker id sequences match the sequential run
    for emission, sequential_result in zip(emissions, sequential_results):
        pipelined_detections = emission.predictions[0]["tracked_detections"]
        sequential_detections = sequential_result[0]["tracked_detections"]
        assert pipelined_detections.xyxy.tolist() == sequential_detections.xyxy.tolist()
        assert (
            pipelined_detections.tracker_id.tolist()
            == sequential_detections.tracker_id.tolist()
        )
