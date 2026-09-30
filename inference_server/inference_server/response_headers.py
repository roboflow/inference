"""Per-request model and request-id response headers.

Mirrors the headers the legacy ``inference`` HTTP server (1.7.x) attaches to
every response, so clients reading them keep working against this server:

  - ``X-Model-Id``: comma-joined, sorted ids of the models the request used
    (only when at least one model was used).
  - ``X-Model-Cold-Start``: ``"true"`` when the request triggered a model
    load, ``"false"`` otherwise.
  - ``X-Model-Cold-Start-Count``: number of loads the request triggered.
  - ``X-Model-Load-Time``: total load time in seconds (float as ``str``),
    only on a cold start.
  - ``X-Model-Load-Details``: JSON list ``[{"m": <model_id>, "t": <seconds>}]``,
    only on a cold start and only while it fits in 4096 bytes.
  - ``x-inference-engine``: always ``inference-models``.
  - ``X-Request-ID``: the caller's value when it is a valid UUID4, a fresh
    ``uuid4().hex`` otherwise (same rule as ``asgi_correlation_id``).

The collector travels in a ContextVar holding a mutable, thread-safe object:
worker threads (Workflows steps) and loop tasks spawned for the request share
the same instance through context copies, so everything they record lands in
one place.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple
from uuid import UUID, uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

MODEL_COLD_START_HEADER = "X-Model-Cold-Start"
MODEL_COLD_START_COUNT_HEADER = "X-Model-Cold-Start-Count"
MODEL_LOAD_TIME_HEADER = "X-Model-Load-Time"
MODEL_LOAD_DETAILS_HEADER = "X-Model-Load-Details"
MODEL_ID_HEADER = "X-Model-Id"
INFERENCE_ENGINE_HEADER = "x-inference-engine"
REQUEST_ID_HEADER = "X-Request-ID"

INFERENCE_ENGINE = "inference-models"
MAX_LOAD_DETAILS_BYTES = 4096

# Headers browser clients may read cross-origin (same set the legacy server
# exposed for model metadata).
EXPOSED_MODEL_HEADERS = [
    MODEL_COLD_START_HEADER,
    MODEL_COLD_START_COUNT_HEADER,
    MODEL_LOAD_TIME_HEADER,
    MODEL_LOAD_DETAILS_HEADER,
    MODEL_ID_HEADER,
]


class RequestModelUsage:
    """Thread-safe record of the models one request used and loaded."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_ids: set[str] = set()
        self._loads: List[Tuple[str, float]] = []

    def record_model_used(self, model_id: str) -> None:
        with self._lock:
            self._model_ids.add(model_id)

    def record_model_load(self, model_id: str, load_time_s: float) -> None:
        with self._lock:
            self._loads.append((model_id, load_time_s))

    def snapshot(self) -> Tuple[set[str], List[Tuple[str, float]]]:
        with self._lock:
            return set(self._model_ids), list(self._loads)


_request_model_usage: contextvars.ContextVar[Optional[RequestModelUsage]] = (
    contextvars.ContextVar("request_model_usage", default=None)
)


def current_model_usage() -> Optional[RequestModelUsage]:
    """Collector of the request being served; None outside HTTP requests."""
    return _request_model_usage.get()


@contextmanager
def track_model_usage() -> Iterator[RequestModelUsage]:
    """Install a fresh collector for the current context (one per request)."""
    usage = RequestModelUsage()
    token = _request_model_usage.set(usage)
    try:
        yield usage
    finally:
        _request_model_usage.reset(token)


def record_model_used(model_id: str) -> None:
    usage = _request_model_usage.get()
    if usage is not None:
        usage.record_model_used(model_id)


def build_model_response_headers(usage: RequestModelUsage) -> Dict[str, str]:
    model_ids, loads = usage.snapshot()
    headers = {
        MODEL_COLD_START_HEADER: "false",
        MODEL_COLD_START_COUNT_HEADER: str(len(loads)),
    }
    if model_ids:
        headers[MODEL_ID_HEADER] = ",".join(sorted(model_ids))
    if not loads:
        return headers
    headers[MODEL_COLD_START_HEADER] = "true"
    headers[MODEL_LOAD_TIME_HEADER] = str(sum(load_time for _, load_time in loads))
    details = json.dumps(
        [{"m": model_id, "t": load_time} for model_id, load_time in loads]
    )
    if len(details) <= MAX_LOAD_DETAILS_BYTES:
        headers[MODEL_LOAD_DETAILS_HEADER] = details
    return headers


def _is_valid_uuid4(value: str) -> bool:
    try:
        return UUID(value).version == 4
    except ValueError:
        return False


def resolve_request_id(incoming: Optional[str]) -> str:
    if incoming and _is_valid_uuid4(incoming):
        return incoming
    generated = uuid4().hex
    if incoming:
        logger.warning(
            "Generated new request ID (%s), since request header value failed "
            "validation",
            generated,
        )
    return generated


class ResponseHeadersMiddleware:
    """Raw ASGI middleware adding request-id, engine and model headers.

    Raw ASGI (not ``@app.middleware``) so the request body stream is passed
    through untouched and the ContextVar is set in the task that runs the
    route.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_headers = MutableHeaders(scope=scope)
        incoming_request_id = request_headers.get(REQUEST_ID_HEADER.lower())
        request_id = resolve_request_id(incoming_request_id)
        if request_id != incoming_request_id:
            request_headers[REQUEST_ID_HEADER] = request_id

        with track_model_usage() as usage:

            async def send_with_headers(message: Message) -> None:
                if message["type"] == "http.response.start":
                    headers = MutableHeaders(scope=message)
                    headers[INFERENCE_ENGINE_HEADER] = INFERENCE_ENGINE
                    for name, value in build_model_response_headers(usage).items():
                        headers[name] = value
                    headers.append(REQUEST_ID_HEADER, request_id)
                await send(message)

            await self.app(scope, receive, send_with_headers)
