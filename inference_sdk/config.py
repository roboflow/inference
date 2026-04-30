import contextvars
import json
import os
import threading
from typing import Iterable, Optional, Tuple

from inference_sdk.utils.environment import str2bool

execution_id = contextvars.ContextVar("execution_id", default=None)


class RemoteProcessingTimeCollector:
    """Thread-safe collector for GPU processing times from remote execution responses.

    A single instance is shared across all threads handling a single request.
    Each entry stores a model_id alongside the processing time.

    Uses threading.Lock (not asyncio.Lock) because add() is only called from
    synchronous worker threads (ThreadPoolExecutor). The middleware reads via
    drain() after await call_next() returns, at which point all worker threads
    have completed — so there is no contention in the async context.
    """

    def __init__(self):
        self._entries: list = []  # list of (model_id, time) tuples
        self._model_ids: set = set()
        self._cold_start_entries: list = []  # list of (model_id, load_time) tuples
        self._cold_start_total_load_time: float = 0.0
        self._cold_start_count: int = 0
        self._lock = threading.Lock()

    def add(self, processing_time: float, model_id: str = "unknown") -> None:
        with self._lock:
            self._entries.append((model_id, processing_time))

    def add_model_id(self, model_id: Optional[str]) -> None:
        if model_id in (None, "", "unknown"):
            return
        with self._lock:
            self._model_ids.add(model_id)

    def add_model_ids(self, model_ids: Iterable[str]) -> None:
        filtered_ids = {
            model_id for model_id in model_ids if model_id not in (None, "", "unknown")
        }
        if not filtered_ids:
            return
        with self._lock:
            self._model_ids.update(filtered_ids)

    def record_cold_start(
        self,
        load_time: float,
        model_id: Optional[str] = None,
        count: int = 1,
    ) -> None:
        with self._lock:
            self._cold_start_total_load_time += load_time
            self._cold_start_count += count
            if model_id not in (None, "", "unknown"):
                self._cold_start_entries.append((model_id, load_time))
                self._model_ids.add(model_id)

    def drain(self) -> list:
        """Atomically return all entries and clear the internal list."""
        with self._lock:
            entries = self._entries
            self._entries = []
            return entries

    def snapshot_entries(self) -> list:
        with self._lock:
            return list(self._entries)

    def snapshot_model_ids(self) -> set:
        with self._lock:
            return set(self._model_ids)

    def snapshot_cold_start_entries(self) -> list:
        with self._lock:
            return list(self._cold_start_entries)

    def snapshot_cold_start_total_load_time(self) -> float:
        with self._lock:
            return self._cold_start_total_load_time

    def snapshot_cold_start_count(self) -> int:
        with self._lock:
            return self._cold_start_count

    def has_data(self) -> bool:
        with self._lock:
            return len(self._entries) > 0

    def has_cold_start_data(self) -> bool:
        with self._lock:
            return self._cold_start_count > 0

    def snapshot_summary(
        self, max_detail_bytes: int = 4096
    ) -> Tuple[float, Optional[str]]:
        """Return (total_time, entries_json_or_none) without clearing entries."""
        entries = self.snapshot_entries()
        total = sum(t for _, t in entries)
        detail = json.dumps([{"m": m, "t": t} for m, t in entries])
        if len(detail) > max_detail_bytes:
            detail = None
        return total, detail

    def summarize(self, max_detail_bytes: int = 4096) -> Tuple[float, Optional[str]]:
        """Atomically drain entries and return (total_time, entries_json_or_none).

        Returns the total processing time and a JSON string of individual entries.
        If the JSON exceeds max_detail_bytes, the detail string is omitted (None).
        """
        entries = self.drain()
        total = sum(t for _, t in entries)
        detail = json.dumps([{"m": m, "t": t} for m, t in entries])
        if len(detail) > max_detail_bytes:
            detail = None
        return total, detail


remote_processing_times = contextvars.ContextVar(
    "remote_processing_times", default=None
)

WORKFLOW_RUN_RETRIES_ENABLED = str2bool(
    os.getenv("WORKFLOW_RUN_RETRIES_ENABLED", "True")
)
EXECUTION_ID_HEADER = os.getenv("EXECUTION_ID_HEADER", "execution_id")
PROCESSING_TIME_HEADER = os.getenv("PROCESSING_TIME_HEADER", "X-Processing-Time")
INTERNAL_REMOTE_EXEC_REQ_HEADER = "X-Internal-Remote-Exec-Req"
INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER = "X-Internal-Remote-Exec-Req-Verified"
ENABLE_INTERNAL_REMOTE_EXEC_HEADER = os.getenv(
    "ENABLE_INTERNAL_REMOTE_EXEC_HEADER", "False"
).lower() in ("true", "1")

apply_duration_minimum = contextvars.ContextVar("apply_duration_minimum", default=False)


ALL_ROBOFLOW_API_URLS = {
    "https://detect.roboflow.com",
    "https://outline.roboflow.com",
    "https://classify.roboflow.com",
    "https://infer.roboflow.com",
    "https://serverless.roboflow.com",
    "https://serverless.roboflow.one",
    "https://asyncinfer.roboflow.com",
    "https://asyncinfer.roboflow.one",
}


# WebRTC configuration
WEBRTC_INITIAL_FRAME_TIMEOUT = float(os.getenv("WEBRTC_INITIAL_FRAME_TIMEOUT", "90.0"))
WEBRTC_VIDEO_QUEUE_MAX_SIZE = int(os.getenv("WEBRTC_VIDEO_QUEUE_MAX_SIZE", "8"))
WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT = float(
    os.getenv("WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT", "2.0")
)

# Video file upload via datachannel
WEBRTC_VIDEO_UPLOAD_CHUNK_SIZE = int(
    os.getenv("WEBRTC_VIDEO_UPLOAD_CHUNK_SIZE", "49152")
)  # 48KB - safe for WebRTC
WEBRTC_VIDEO_UPLOAD_BUFFER_LIMIT = int(
    os.getenv("WEBRTC_VIDEO_UPLOAD_BUFFER_LIMIT", "262144")
)  # 256KB max buffered before backpressure

# Roboflow API base URL for TURN config and other services
RF_API_BASE_URL = os.getenv("RF_API_BASE_URL", "https://api.roboflow.com")


class InferenceSDKDeprecationWarning(Warning):
    """Class used for warning of deprecated features in the Inference SDK"""

    pass
