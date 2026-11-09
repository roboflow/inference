"""Exercise delayed output through the actual workflow-to-WebRTC adapter."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from av import VideoFrame
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
)
from streamvision.stream.model_handlers.workflows import WorkflowRunner
from streamvision.webrtc_worker.utils import detect_image_output, process_frame


def test_example_warmup_autodetection_and_delayed_image(monkeypatch):
    # Only model inference is simulated. Compilation, step wiring, video
    # metadata, workflow runner, rendering and AV image conversion are real.
    def model_run(self, images, **kwargs):
        results = []
        for image in images:
            results.append(
                {
                    "timeline": (
                        [
                            ActionRecognitionPrediction(
                                start_frame_idx=1,
                                end_frame_idx=20,
                                class_name="wave",
                                class_id=0,
                            )
                        ]
                        if image.video_metadata.frame_number >= 151
                        else []
                    ),
                    "error_status": "",
                }
            )
        return results

    monkeypatch.setattr(ActionRecognitionModelBlockV1, "run", model_run)
    example = (
        Path(__file__).resolve().parents[4]
        / "examples/action-recognition-delay/workflow.json"
    )
    engine = ExecutionEngine.init(
        workflow_definition=json.loads(example.read_text()),
        init_parameters={
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
            "workflows_core.model_manager": None,
        },
    )
    runner = WorkflowRunner(
        workflows_parameters={"model_id": "test/1"},
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    pipeline = SimpleNamespace(_on_video_frame=runner)
    for number in [1, 1, *range(2, 302)]:
        frame = VideoFrame.from_ndarray(
            np.full((90, 160, 3), number % 256, dtype=np.uint8),
            format="bgr24",
        )
        result, rendered, errors = process_frame(
            frame=frame,
            frame_id=number,
            declared_fps=30,
            measured_fps=2,
            comes_from_video_file=False,
            inference_pipeline=pipeline,
            stream_output="image",
        )
        assert not errors
        assert detect_image_output(result) == "image"
        assert rendered is not None
        if number <= 300:
            assert result["status"] == "buffering"
        else:
            assert result["displayed_frame_number"] == 1
            assert result["image"].video_metadata.frame_number == 1
            assert result["actions"] == ["wave"]
            assert rendered.to_ndarray(format="bgr24")[-1, -1, 0] == 1
