"""RTSP Stream Watcher: watch another camera from inside a Workflow.

The workflow's own input is one video source (a webcam, a file, a stream). This
block holds a background reader on *another* RTSP stream, samples it every
`check_interval_seconds`, optionally runs an object detection model on the
sample in that same background thread, and hands the Workflow the latest frame
and detections on every run. A Workflow can therefore react to what a different
camera sees - a person walking into a yard camera switching an OBS scene - with
no process outside the Workflow and no cost on the Workflow's own frame rate:
the model never runs on the request path.

Readers are shared per (stream URL, model) across every block instance in the
process, drain the stream continuously (RTSP buffers otherwise lag by seconds),
keep only the newest sample, reconnect on failure, and stop themselves when
nothing has asked for a frame for a while.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    filter_out_unwanted_classes_from_sv_detections_batch,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
    roboflow_platform_model,
)

PLACEHOLDER_SHAPE = (360, 640, 3)

Detector = Callable[[np.ndarray], sv.Detections]

LONG_DESCRIPTION = """
Watch a second camera from inside a Workflow. The block keeps a background reader open on an
RTSP stream, samples it every `check_interval_seconds`, and - if a `model_id` is set - runs
that model on the sample **in the background**. Every Workflow run then gets the camera's
latest frame and detections instantly, so a Workflow driven by one camera can react to what
another camera sees without slowing down.

## How This Block Works

1. On first use it opens the RTSP stream in a background thread shared by every block
   pointing at the same URL and model, and keeps draining it so the newest frame is always
   at hand
2. Every `check_interval_seconds` the thread keeps one frame and, when `model_id` is set,
   runs the model on it. The model runs on a spare core, never on the Workflow's own
   request, so extra cameras cost the primary video no frame rate
3. Each Workflow run returns the latest sample as `frame` and its detections as
   `predictions`. Between checks the same detections are returned again, so downstream
   logic sees a steady value rather than flicker
4. If the stream is down, still connecting, or its newest frame is older than
   `max_frame_age_seconds`, `frame_available` is `False`, `frame` is a blank image and
   `predictions` is empty, so scene logic holds instead of acting on a frozen picture
5. When no Workflow has asked for a frame for `idle_timeout_seconds`, the reader closes the
   stream; the next request reopens it

## Common Use Cases

- **Multi-camera scene switching**: a person on a yard camera switches OBS to that camera,
  while the Workflow's own input stays on the presenter's webcam
- **Cross-camera triggers**: act in one view on something seen in another
- **Low-rate monitoring** of extra cameras alongside a primary video workflow

## Connecting to Other Blocks

- **Then a Detections Class Router** on `predictions` to turn what the camera sees into a
  value, and **First Non Empty Or Default** to give the cameras priority over the primary
  input
- **Then an OBS Action**, or any other sink
- Leave `model_id` empty to use the block as a plain frame source and run any model on
  `frame` yourself; that model then runs on every Workflow run

## Requirements

The RTSP stream must be reachable from the machine running the Workflow. Cameras on a local
network are not reachable from Roboflow Hosted Serverless or Dedicated Deployments.
"""


def _redact(url: str) -> str:
    """Stream URLs often carry credentials; never let them reach a log line."""
    try:
        parts = urlsplit(url)
        if parts.username or parts.password:
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            return urlunsplit(
                (parts.scheme, f"***@{host}", parts.path, parts.query, "")
            )
    except ValueError:
        pass
    return url


def _reader_key(url: str, model_id: Optional[str]) -> str:
    return f"{url}#{model_id or ''}"


class _StreamReader:
    """One background reader per (stream URL, model), shared across block instances."""

    def __init__(self, url: str, reconnect_delay: float, idle_timeout: float):
        self.url = url
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self.idle_timeout = max(1.0, float(idle_timeout))
        self.check_interval = 3.0
        self.detector: Optional[Detector] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_time: float = 0.0
        self._detections: Optional[sv.Detections] = None
        self._status: str = "connecting"
        self._last_request: float = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- control ---------------------------------------------------------------

    def configure(self, check_interval: float, detector: Optional[Detector]) -> None:
        """Latest caller's settings win; readers are shared, so keep them consistent."""
        with self._lock:
            self.check_interval = max(0.05, float(check_interval))
            self.detector = detector

    def ensure_running(self) -> None:
        with self._lock:
            self._last_request = time.monotonic()
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._status = "connecting"
            self._thread = threading.Thread(
                target=self._run, name=f"rtsp-watcher:{_redact(self.url)}", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def latest(
        self,
    ) -> Tuple[Optional[np.ndarray], float, str, Optional[sv.Detections]]:
        with self._lock:
            return self._frame, self._frame_time, self._status, self._detections

    # -- reader loop -----------------------------------------------------------

    def _open(self) -> Any:
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _check(self, frame: np.ndarray, now: float) -> None:
        """Keep a sample and run the model on it; a model fault is not a stream fault."""
        with self._lock:
            detector = self.detector
        detections: Optional[sv.Detections] = None
        if detector is not None:
            try:
                detections = detector(frame)
            except Exception as error:  # noqa: BLE001 - keep watching, retry next check
                logger.warning(
                    "RTSP watcher model failed on %s: %s", _redact(self.url), error
                )
                detections = sv.Detections.empty()
        with self._lock:
            self._frame, self._frame_time, self._detections = frame, now, detections

    def _run(self) -> None:
        while not self._stop.is_set():
            if time.monotonic() - self._last_request > self.idle_timeout:
                with self._lock:
                    self._status = "idle"
                return
            capture = self._open()
            if not capture or not capture.isOpened():
                with self._lock:
                    self._status = "reconnecting"
                logger.warning(
                    "RTSP watcher could not open %s; retrying", _redact(self.url)
                )
                self._stop.wait(self.reconnect_delay)
                continue
            with self._lock:
                self._status = "streaming"
            last_check = 0.0
            try:
                while not self._stop.is_set():
                    if time.monotonic() - self._last_request > self.idle_timeout:
                        with self._lock:
                            self._status = "idle"
                        return
                    # grab() drains the stream so the kept frame is the newest one
                    if not capture.grab():
                        raise RuntimeError("stream read failed")
                    now = time.monotonic()
                    if now - last_check < self.check_interval:
                        continue
                    ok, frame = capture.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError("frame decode failed")
                    last_check = now
                    self._check(frame, now)
            except Exception as error:  # noqa: BLE001 - any stream fault -> reconnect
                with self._lock:
                    self._status = "reconnecting"
                logger.warning(
                    "RTSP watcher lost %s (%s); reconnecting", _redact(self.url), error
                )
                self._stop.wait(self.reconnect_delay)
            finally:
                try:
                    capture.release()
                except Exception:  # noqa: BLE001
                    pass


_READERS: Dict[str, _StreamReader] = {}
_READERS_LOCK = threading.Lock()


def get_reader(
    url: str,
    model_id: Optional[str],
    check_interval: float,
    detector: Optional[Detector],
    reconnect_delay: float,
    idle_timeout: float,
) -> _StreamReader:
    key = _reader_key(url, model_id)
    with _READERS_LOCK:
        reader = _READERS.get(key)
        if reader is None:
            reader = _StreamReader(url, reconnect_delay, idle_timeout)
            _READERS[key] = reader
    reader.configure(check_interval=check_interval, detector=detector)
    reader.ensure_running()
    return reader


def stop_all_readers() -> None:
    """Stop every reader. Used by tests."""
    with _READERS_LOCK:
        readers = list(_READERS.values())
        _READERS.clear()
    for reader in readers:
        reader.stop()


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "RTSP Stream Watcher",
            "version": "v1",
            "short_description": "Watch another RTSP camera and run a model on it in the background.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "search_keywords": [
                "rtsp",
                "camera",
                "stream",
                "watch",
                "second camera",
                "multi-camera",
                "monitor",
            ],
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-video",
                "blockPriority": 3,
                "popular": False,
            },
        }
    )
    type: Literal["roboflow_core/rtsp_stream_watcher@v1"]
    stream_url: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = Field(
        description="RTSP URL of the camera to watch. Pass it as a Workflow parameter or from a "
        "secrets provider if it carries credentials.",
        examples=["rtsp://192.168.1.17:554/live/mainStream", "$inputs.camera_url"],
    )
    check_interval_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=3.0,
        description="How often to sample the camera and run the model on it. A few seconds is "
        "plenty for scene switching and keeps the model off the Workflow's own frame rate.",
        examples=[3.0],
    )
    model_id: Optional[Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str]] = Field(
        default=None,
        title="Model",
        description="Object detection model to run on every check, in the background. Leave "
        "empty to only emit frames.",
        examples=["rfdetr-nano", "my_project/3", "$inputs.model"],
    )
    confidence: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.4,
        description="Confidence threshold for the background model.",
        examples=[0.4],
        json_schema_extra={"relevant_for": {"model_id": {"required": True}}},
    )
    class_filter: Optional[Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]] = (
        Field(
            default=None,
            description="Keep only these classes from the background model. Empty keeps all.",
            examples=[["person"], "$inputs.classes"],
            json_schema_extra={"relevant_for": {"model_id": {"required": True}}},
        )
    )
    max_frame_age_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=15.0,
        description="A sample older than this is treated as unavailable, so a frozen or dropped "
        "stream stops driving downstream logic instead of acting on a stale picture. Keep it "
        "well above `check_interval_seconds`.",
        examples=[15.0],
    )
    reconnect_delay_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=2.0,
        description="Wait between reconnection attempts when the stream drops.",
        examples=[2.0],
    )
    idle_timeout_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=60.0,
        description="Close the stream when no Workflow has asked for a frame for this long. "
        "The next request reopens it.",
        examples=[60.0],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return []

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="frame", kind=[IMAGE_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="frame_available", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
            OutputDefinition(name="frame_age_seconds", kind=[FLOAT_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        if isinstance(self.model_id, str) and not self.model_id.startswith("$"):
            return [roboflow_platform_model(model_id=self.model_id)]
        return None

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            RuntimeRestriction(
                severity=Severity.HARD,
                note=(
                    "Block opens an RTSP stream from the process running the Workflow, "
                    "normally a camera on a local network. Hosted Serverless and Roboflow "
                    "Dedicated Deployments cannot reach it."
                ),
                applies_to_runtimes=[
                    Runtime.HOSTED_SERVERLESS,
                    Runtime.DEDICATED_DEPLOYMENT,
                ],
            ),
        ]


class RTSPStreamWatcherBlockV1(WorkflowBlock):

    def __init__(self, model_manager: ModelManager, api_key: Optional[str]):
        self._model_manager = model_manager
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _build_detector(
        self,
        model_id: str,
        confidence: float,
        class_filter: Optional[List[str]],
        parent_id: str,
    ) -> Detector:
        """Same local path as the Roboflow Object Detection block, run off-request."""

        def detect(frame: np.ndarray) -> sv.Detections:
            image = WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id=parent_id),
                numpy_image=frame,
            )
            request = ObjectDetectionInferenceRequest(
                api_key=self._api_key,
                model_id=model_id,
                image=[image.to_inference_format(numpy_preferred=True)],
                confidence=confidence,
                class_filter=class_filter,
                source="workflow-execution",
            )
            self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
            predictions = self._model_manager.infer_from_request_sync(
                model_id=model_id, request=request
            )
            if not isinstance(predictions, list):
                predictions = [predictions]
            predictions = [
                e.model_dump(by_alias=True, exclude_none=True) for e in predictions
            ]
            predictions = convert_inference_detections_batch_to_sv_detections(
                predictions
            )
            predictions = attach_prediction_type_info_to_sv_detections_batch(
                predictions=predictions, prediction_type="object-detection"
            )
            predictions = filter_out_unwanted_classes_from_sv_detections_batch(
                predictions=predictions, classes_to_accept=class_filter
            )
            predictions = attach_parents_coordinates_to_batch_of_sv_detections(
                images=[image], predictions=predictions
            )
            return predictions[0]

        return detect

    def run(
        self,
        stream_url: str,
        check_interval_seconds: float,
        model_id: Optional[str],
        confidence: float,
        class_filter: Optional[List[str]],
        max_frame_age_seconds: float,
        reconnect_delay_seconds: float,
        idle_timeout_seconds: float,
    ) -> BlockResult:
        parent_id = f"rtsp-watcher:{_redact(stream_url)}"
        detector = None
        if model_id:
            detector = self._build_detector(
                model_id=model_id,
                confidence=confidence,
                class_filter=class_filter or None,
                parent_id=parent_id,
            )
        reader = get_reader(
            url=stream_url,
            model_id=model_id,
            check_interval=check_interval_seconds,
            detector=detector,
            reconnect_delay=reconnect_delay_seconds,
            idle_timeout=idle_timeout_seconds,
        )
        frame, frame_time, status, detections = reader.latest()
        age = (time.monotonic() - frame_time) if frame is not None else float("inf")
        available = frame is not None and age <= max_frame_age_seconds
        if not available:
            # a blank frame and empty detections make downstream logic hold rather
            # than react to a frozen picture (or crash on a missing image)
            frame = np.zeros(PLACEHOLDER_SHAPE, dtype=np.uint8)
            detections = None
            if status == "streaming" and frame_time:
                status = "stale"
        return {
            "frame": WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id=parent_id),
                numpy_image=frame,
            ),
            "predictions": (
                detections if detections is not None else sv.Detections.empty()
            ),
            "frame_available": available,
            "status": status,
            "frame_age_seconds": round(age, 3) if available else -1.0,
        }
