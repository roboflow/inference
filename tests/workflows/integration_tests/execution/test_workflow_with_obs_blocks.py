"""Workflow-level tests for the OBS Studio blocks and the RTSP Stream Watcher.

OBS itself is replaced by a fake websocket client and the RTSP camera by a fake
capture, the same stand-ins the unit tests use, so these run anywhere. What they
add over the unit tests is the whole path through the Execution Engine: block
compilation, the `obs_connection` kind flowing between blocks, the Class Router
feeding the OBS Action, and the watcher's background model surfacing detections
into a real workflow.
"""

import time

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.sinks.obs import client as obs_client
from inference.core.workflows.core_steps.transformations.rtsp_stream_watcher import (
    v1 as watcher_module,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)


class FakeOBSClient:
    """Stands in for obsws_python.ReqClient and records what the workflow asked for."""

    def __init__(self):
        self.calls = []

    def get_version(self):
        self.calls.append(("get_version",))
        return type("Version", (), {"obs_version": "32.0.0"})()

    def set_current_program_scene(self, scene_name):
        self.calls.append(("set_current_program_scene", scene_name))

    def disconnect(self):
        pass


@pytest.fixture
def fake_obs(monkeypatch):
    clients = []

    def fake_connect(host, port, password, timeout):
        client = FakeOBSClient()
        clients.append(client)
        return client

    obs_client.reset_clients()
    monkeypatch.setattr(obs_client, "_connect", fake_connect)
    yield clients
    obs_client.reset_clients()


SCENE_SWITCH_ON_DETECTED_CLASS = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {"type": "roboflow_core/obs_connection@v1", "name": "obs"},
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/detections_class_router@v1",
            "name": "scene_router",
            "predictions": "$steps.detector.predictions",
            "routes": {"dog": "Dogs", "cat": "Cats"},
            "confidence_threshold": 0.4,
        },
        {
            "type": "roboflow_core/obs_action@v1",
            "name": "switch_scene",
            "connection": "$steps.obs.connection",
            "action": "set_scene",
            "scene_name": "$steps.scene_router.value",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "scene", "selector": "$steps.scene_router.value"},
        {"type": "JsonField", "name": "obs", "selector": "$steps.switch_scene.message"},
        {
            "type": "JsonField",
            "name": "connection_note",
            "selector": "$steps.obs.message",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows driving OBS Studio",
    use_case_title="Switch the OBS Studio scene when a class is detected",
    use_case_description="""
Runs an object detection model on the input, maps the most confident detected class to an
OBS scene name with the **Detections Class Router** (`dog -> Dogs`, `cat -> Cats`), and
switches OBS to that scene with the **OBS Action** block over obs-websocket. Frames with
nothing routed hold the current scene, and repeated identical requests are deduplicated so
the sink costs nothing while nothing changes.

!!! note "Runs against a local OBS Studio"

    The OBS blocks talk to OBS Studio's websocket server on the machine running `inference`,
    so they are not available on Roboflow Hosted Serverless or Dedicated Deployments.
""",
    workflow_definition=SCENE_SWITCH_ON_DETECTED_CLASS,
    workflow_name_in_app="obs-scene-switch-on-detected-class",
)
def test_workflow_switches_obs_scene_on_detected_class(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
    fake_obs,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=SCENE_SWITCH_ON_DETECTED_CLASS,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    first = execution_engine.run(runtime_parameters={"image": [dogs_image]})
    again = execution_engine.run(runtime_parameters={"image": [dogs_image]})
    nothing_routed = execution_engine.run(runtime_parameters={"image": [crowd_image]})

    # then
    assert len(fake_obs) == 1, "one pooled connection is shared by both OBS blocks"
    scene_calls = [c for c in fake_obs[0].calls if c[0] == "set_current_program_scene"]
    assert first[0]["scene"] == "Dogs"
    assert first[0]["obs"] == "Switched OBS to scene 'Dogs'"
    assert (
        "authentication" in first[0]["connection_note"]
        or "password" in first[0]["connection_note"]
    ), "the connection reports where its credential came from"
    assert "skipped" in again[0]["obs"], "an unchanged scene costs no OBS call"
    assert nothing_routed[0]["scene"] is None
    # the engine does not run a step whose target is None, so the action
    # reports nothing and OBS keeps whatever scene it was on
    assert nothing_routed[0]["obs"] is None
    assert scene_calls == [("set_current_program_scene", "Dogs")]


class FakeCapture:
    """Stands in for cv2.VideoCapture, serving one frame over and over."""

    def __init__(self, frame: np.ndarray):
        self._frame = frame

    def isOpened(self):
        return True

    def grab(self):
        time.sleep(0.005)
        return True

    def retrieve(self):
        return True, self._frame.copy()

    def release(self):
        pass


CAMERA_WATCH_WITH_BACKGROUND_MODEL = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/rtsp_stream_watcher@v1",
            "name": "yard_camera",
            "stream_url": "rtsp://192.168.1.31:554/live/mainStream",
            "check_interval_seconds": 0.2,
            "model_id": "yolov8n-640",
            "confidence": 0.4,
            "class_filter": ["dog"],
        },
        {
            "type": "roboflow_core/detections_class_router@v1",
            "name": "camera_router",
            "predictions": "$steps.yard_camera.predictions",
            "routes": {"dog": "YardCam"},
            "confidence_threshold": 0.4,
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "preview",
            "image": "$inputs.image",
            "predictions": "$steps.yard_camera.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "scene",
            "selector": "$steps.camera_router.value",
        },
        {
            "type": "JsonField",
            "name": "status",
            "selector": "$steps.yard_camera.status",
        },
        {
            "type": "JsonField",
            "name": "available",
            "selector": "$steps.yard_camera.frame_available",
        },
        {"type": "JsonField", "name": "frame", "selector": "$steps.yard_camera.frame"},
    ],
}


@add_to_workflows_gallery(
    category="Workflows driving OBS Studio",
    use_case_title="Watch a second RTSP camera with a background model",
    use_case_description="""
The **RTSP Stream Watcher** keeps a second camera open in the background, samples it every
`check_interval_seconds`, runs `model_id` on the sample in its own thread and hands the
latest detections to the workflow on every run, so extra cameras cost the primary video path
no frame rate. Here a **Detections Class Router** turns a detected `dog` into the value
`YardCam`, which an **OBS Action** would use as a scene name.

!!! note "Runs against a camera on the local network"

    The watcher opens the RTSP stream from the machine running `inference`, so it is not
    available on Roboflow Hosted Serverless or Dedicated Deployments.
""",
    workflow_definition=CAMERA_WATCH_WITH_BACKGROUND_MODEL,
    workflow_name_in_app="rtsp-camera-watch-with-background-model",
)
def test_workflow_watches_rtsp_camera_with_background_model(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
    monkeypatch,
) -> None:
    # given
    watcher_module.stop_all_readers()
    monkeypatch.setattr(
        watcher_module._StreamReader, "_open", lambda self: FakeCapture(dogs_image)
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=CAMERA_WATCH_WITH_BACKGROUND_MODEL,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    try:
        # when: the first run starts the reader; the background model needs a
        # moment to load and run its first check
        result = execution_engine.run(runtime_parameters={"image": [crowd_image]})
        deadline = time.monotonic() + 60
        while result[0]["scene"] != "YardCam" and time.monotonic() < deadline:
            time.sleep(0.25)
            result = execution_engine.run(runtime_parameters={"image": [crowd_image]})

        # then
        assert result[0]["status"] == "streaming"
        assert result[0]["available"] is True
        assert result[0]["scene"] == "YardCam"
        assert result[0]["frame"].numpy_image.shape == dogs_image.shape
    finally:
        watcher_module.stop_all_readers()
