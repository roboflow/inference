"""Exercise the example workflow through the actual workflow-to-WebRTC adapter."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from av import VideoFrame
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1,
)
from roboflow_workflows.core_steps.visualizations.action_recognition import (
    v1 as visualization_module,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
)
from streamvision.stream.model_handlers.workflows import WorkflowRunner
from streamvision.webrtc_worker.utils import detect_image_output, process_frame

from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)
from inference_models.models.base.action_recognition import VideoSampling

EXAMPLE = (
    Path(__file__).resolve().parents[4]
    / "examples/action-recognition-visualization/workflow.json"
)
FIRST_RESULT_FRAME = 151


def _example_engine(monkeypatch):
    def font_unavailable(font_family):
        raise RuntimeError("font unavailable in tests")

    monkeypatch.setattr(visualization_module, "resolve_font_path", font_unavailable)
    engine = ExecutionEngine.init(
        workflow_definition=json.loads(EXAMPLE.read_text()),
        init_parameters={
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
            "workflows_core.model_manager": None,
        },
    )
    return engine


def _stream(engine, *, display_mode, frame_numbers, measured_fps, after_frame=None):
    # Compilation, step wiring, video metadata, workflow runner, rendering and
    # AV image conversion are real.
    runner = WorkflowRunner(
        workflows_parameters={"model_id": "test/1", "display_mode": display_mode},
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    pipeline = SimpleNamespace(_on_video_frame=runner)
    outputs = []
    for number in frame_numbers:
        frame = VideoFrame.from_ndarray(
            np.full((180, 320, 3), number % 256, dtype=np.uint8),
            format="bgr24",
        )
        result, rendered, errors = process_frame(
            frame=frame,
            frame_id=number,
            declared_fps=30,
            measured_fps=measured_fps,
            comes_from_video_file=False,
            inference_pipeline=pipeline,
            stream_output="image",
        )
        assert not errors
        assert detect_image_output(result) == "image"
        outputs.append((number, result, rendered.to_ndarray(format="bgr24")))
        if after_frame is not None:
            after_frame()
    return outputs


def test_example_synchronized_display_puts_labels_on_their_frames(monkeypatch):
    # Only model inference is simulated.
    def model_run(self, images, **kwargs):
        results = []
        for image in images:
            analyzed = image.video_metadata.frame_number >= FIRST_RESULT_FRAME
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
                        if analyzed
                        else []
                    ),
                    "error_status": "",
                    # Null until the first result: the step after must still run.
                    "analyzed_until_frame": 120 if analyzed else None,
                }
            )
        return results

    monkeypatch.setattr(ActionRecognitionModelBlockV1, "run", model_run)
    outputs = _stream(
        _example_engine(monkeypatch),
        display_mode="synchronized",
        frame_numbers=[1, 1, *range(2, 172)],
        measured_fps=2,
    )

    for number, result, pixels in outputs:
        if number < FIRST_RESULT_FRAME:
            assert result["status"] == "analyzing"
            assert result["displayed_frame_number"] == 1
            continue

        shown = result["displayed_frame_number"]
        assert result["status"] == "synchronized"
        assert shown == number - (FIRST_RESULT_FRAME - 2)
        assert result["image"].video_metadata.frame_number == shown
        assert result["actions"] == (["wave"] if shown <= 20 else [])
        # Top-right corner: outside the badges and above the strip.
        assert pixels[1, -2, 0] == pytest.approx(shown % 256, abs=2)


class _InstantModel:
    """Stands in for a loaded model; every call reports one action."""

    class_names = ["wave"]
    supports_confidence = False
    supports_observed_duration = False

    def __init__(self):
        self.video_sampling = VideoSampling(
            window_seconds=5.0,
            sample_fps=2.0,
            min_frames=1,
        )
        self.calls = 0

    def infer(self, frames, class_names=None, fps=None, **kwargs):
        self.calls += 1
        return [
            ModelActionRecognitionPrediction(
                start_frame_idx=0,
                end_frame_idx=len(frames) - 1,
                class_name="wave",
            )
        ]


def test_example_live_display_runs_both_real_blocks_with_background_inference(
    monkeypatch,
):
    # The model block is real here: sampling, the worker thread, the merge and
    # the analyzed-frame mark. Only the weights are replaced.
    model = _InstantModel()
    monkeypatch.setattr(
        ActionRecognitionModelBlockV1, "_get_model", lambda self, model_id: model
    )
    engine = _example_engine(monkeypatch)
    model_block = engine._engine._compiled_workflow.steps["actions"].step

    def collect_result():
        pending = model_block._video_bookkeeping["default_source"].pending
        if pending is not None:
            # Let the worker finish, so the next frame collects its result.
            pending.future.result(timeout=5)

    try:
        outputs = _stream(
            engine,
            display_mode="live",
            frame_numbers=[1, 1, *range(2, 161)],
            measured_fps=30,
            after_frame=collect_result,
        )
    finally:
        model_block.close()

    for number, result, pixels in outputs:
        assert result["displayed_frame_number"] == number
        assert pixels[1, -2, 0] == number % 256
    results = [result for _, result, _ in outputs]
    # The stride is the model's 5-second window, so the first call fires on
    # frame 151; the worker's answer is collected on frame 152.
    assert model.calls == 1
    assert results[151]["status"] == "analyzing"
    assert results[151]["analyzed_until_frame"] is None
    assert results[152]["status"] == "live"
    assert results[152]["analyzed_until_frame"] == 151
    assert results[152]["actions"] == ["wave"]
    assert results[-1]["actions_frame_number"] == 151
    assert results[-1]["model_error"] == ""
