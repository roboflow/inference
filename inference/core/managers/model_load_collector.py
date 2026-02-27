import contextvars
import json
import threading
from typing import Optional, Tuple


class ModelLoadCollector:
    """Thread-safe collector for model cold start events during a request.

    A single instance is shared across all threads handling a single request.
    Each entry stores a model_id alongside the load time.

    Mirrors the design of RemoteProcessingTimeCollector from inference_sdk.
    """

    def __init__(self):
        self._entries: list = []  # list of (model_id, load_time) tuples
        self._lock = threading.Lock()

    def record(self, model_id: str, load_time: float) -> None:
        with self._lock:
            self._entries.append((model_id, load_time))

    def has_data(self) -> bool:
        with self._lock:
            return len(self._entries) > 0

    def summarize(self, max_detail_bytes: int = 4096) -> Tuple[float, Optional[str]]:
        """Return (total_load_time, entries_json_or_none).

        Returns the total model load time and a JSON string of individual
        entries.  If the JSON exceeds *max_detail_bytes*, the detail string
        is omitted (None).
        """
        with self._lock:
            entries = list(self._entries)
        total = sum(t for _, t in entries)
        detail = json.dumps([{"m": m, "t": t} for m, t in entries])
        if len(detail) > max_detail_bytes:
            detail = None
        return total, detail


model_load_info: contextvars.ContextVar[Optional[ModelLoadCollector]] = (
    contextvars.ContextVar("model_load_info", default=None)
)


class RequestModelIds:
    """Thread-safe set of model IDs used during a request."""

    def __init__(self):
        self._ids: set = set()
        self._lock = threading.Lock()

    def add(self, model_id: str) -> None:
        with self._lock:
            self._ids.add(model_id)

    def get_ids(self) -> set:
        with self._lock:
            return set(self._ids)


request_model_ids: contextvars.ContextVar[Optional[RequestModelIds]] = (
    contextvars.ContextVar("request_model_ids", default=None)
)

request_workflow_id: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("request_workflow_id", default=None)
)
