import contextvars
import json
import os
import threading
from typing import Optional, Tuple

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
        self._lock = threading.Lock()

    def add(self, processing_time: float, model_id: str = "unknown") -> None:
        with self._lock:
            self._entries.append((model_id, processing_time))

    def drain(self) -> list:
        """Atomically return all entries and clear the internal list."""
        with self._lock:
            entries = self._entries
            self._entries = []
            return entries

    def has_data(self) -> bool:
        with self._lock:
            return len(self._entries) > 0

    def summarize(
        self, max_detail_bytes: int = 4096
    ) -> Tuple[float, Optional[str]]:
        """Atomically drain entries and return (total_time, entries_json_or_none).

        Returns the total processing time and a JSON string of individual entries.
        If the JSON exceeds max_detail_bytes, the detail string is omitted (None).
        """
        entries = self.drain()
        total = sum(t for _, t in entries)
        detail = json.dumps(
            [{"m": m, "t": t} for m, t in entries]
        )
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


ALL_ROBOFLOW_API_URLS = {
    "https://detect.roboflow.com",
    "https://outline.roboflow.com",
    "https://classify.roboflow.com",
    "https://infer.roboflow.com",
    "https://serverless.roboflow.com",
    "https://serverless.roboflow.one",
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
