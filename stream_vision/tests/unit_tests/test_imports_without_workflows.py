"""`streamvision` runs its non-workflow parts without `roboflow_workflows`."""

import os
import subprocess
import sys

from roboflow_workflows.execution_engine.v1.executor.utils import resolve_futures
from streamvision.stream import pipeline

BLOCKER = """
import sys


class RoboflowWorkflowsBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "roboflow_workflows":
            raise ModuleNotFoundError("blocked by the test", name={missing_name!r})
        return None


sys.meta_path.insert(0, RoboflowWorkflowsBlocker())
"""

IMPORTS_AND_WORKFLOW_ENTRY_POINTS = r"""
import pytest
import streamvision.camera.video_source
import streamvision.stream.sinks
import streamvision.stream.utils
from streamvision.stream import pipeline
from streamvision.stream.exceptions import CannotInitialiseModelError

with pytest.raises(CannotInitialiseModelError, match=r"streamvision\[workflows\]"):
    pipeline.build_workflows_profiler(enabled=False, max_runs_in_buffer=1)
for profiler_argument in ({}, {"profiler": object()}):
    with pytest.raises(CannotInitialiseModelError, match=r"streamvision\[workflows\]"):
        pipeline.InferencePipeline.init_with_workflow(
            "TestPatternStreamProducer",
            workflow_specification={},
            workflow_init_parameters={},
            step_error_handler=None,
            **profiler_argument,
        )
predictions = object()
assert pipeline._resolve_prediction_futures(predictions) is predictions
assert "roboflow_workflows" not in sys.modules
"""

CUSTOM_LOGIC_PIPELINE = r"""
import threading

from streamvision.stream.pipeline import InferencePipeline

dispatched = []
enough_frames = threading.Event()


def on_video_frame(video_frames):
    return [{"frame_id": video_frame.frame_id} for video_frame in video_frames]


def on_prediction(prediction, video_frame):
    dispatched.append((prediction["frame_id"], video_frame.frame_id))
    if len(dispatched) >= 3:
        enough_frames.set()


inference_pipeline = InferencePipeline.init_with_custom_logic(
    video_reference="TestPatternStreamProducer",
    on_video_frame=on_video_frame,
    on_prediction=on_prediction,
)
inference_pipeline.start(use_main_thread=False)
try:
    assert enough_frames.wait(timeout=30), dispatched
finally:
    inference_pipeline.terminate()
    inference_pipeline.join()
assert all(predicted == frame for predicted, frame in dispatched), dispatched
assert "roboflow_workflows" not in sys.modules
"""

IMPORT_PIPELINE_EXPECTING_MISSING_TORCH = r"""
try:
    import streamvision.stream.pipeline
except ModuleNotFoundError as error:
    assert error.name == "torch", error
else:
    raise AssertionError("a missing dependency of roboflow_workflows was silenced")
"""


def _run_child(code: str, *, missing_name: str) -> subprocess.CompletedProcess:
    child_env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    completed = subprocess.run(
        [sys.executable, "-c", BLOCKER.format(missing_name=missing_name) + code],
        capture_output=True,
        text=True,
        timeout=120,
        env=child_env,
    )

    return completed


def test_stream_modules_import_and_workflow_entry_points_ask_for_the_extra() -> None:
    completed = _run_child(
        IMPORTS_AND_WORKFLOW_ENTRY_POINTS,
        missing_name="roboflow_workflows",
    )

    assert completed.returncode == 0, completed.stderr


def test_custom_logic_pipeline_processes_frames_without_roboflow_workflows() -> None:
    completed = _run_child(CUSTOM_LOGIC_PIPELINE, missing_name="roboflow_workflows")

    assert completed.returncode == 0, completed.stderr


def test_missing_dependency_of_roboflow_workflows_still_fails_the_import() -> None:
    completed = _run_child(
        IMPORT_PIPELINE_EXPECTING_MISSING_TORCH,
        missing_name="torch",
    )

    assert completed.returncode == 0, completed.stderr


def test_pipeline_resolves_futures_with_roboflow_workflows_when_installed() -> None:
    assert pipeline.resolve_futures is resolve_futures
