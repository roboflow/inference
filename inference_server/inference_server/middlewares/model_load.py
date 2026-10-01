"""Model load, model id and workflow id headers of every HTTP response.

``ModelLoadHeadersMiddleware`` publishes a fresh list on ``MODEL_LOAD_EVENTS``
for each request. Code that makes a model ready for the request appends one
``(model_id, cold_start, load_time_s)`` tuple through ``record_model_load``:
the legacy bridge before every gateway ``ensure_loaded`` (the attempted model)
and after a load, the v2 dispatch after every gateway ``ensure_loaded`` and
the in-process gateway after the reload inside ``infer``. Events carry the
model id the operation asked for: the legacy bridge publishes the requested
id of the operation on the ``REQUESTED_MODEL_ID`` contextvar through
``set_requested_model_id``, and ``record_model_load`` records it in place of
the canonical id it belongs to. The contextvar lives in the context of the
operation (the request task, or the workflow step thread and the tasks it
starts), so concurrent operations keep their own ids. A route that runs a
workflow sets ``REQUEST_WORKFLOW_ID``. When the response starts, the
middleware turns all of it into the legacy ``X-Model-*`` and ``X-Workflow-Id``
headers and adds ``x-inference-engine``.

Extension point for model loads reported by remote servers:
``REMOTE_MODEL_LOADS`` holds a fresh empty list for each request. A remote
processing-time collector appends itself to it; merging its model ids and
cold-start snapshots into the remote arguments of
``build_model_response_headers`` is left to that collector's change. The
middleware does not read the list yet.
"""

import contextvars
import json
from typing import Any, Dict, List, Optional, Tuple

from inference_server.middlewares.headers import (
    INFERENCE_ENGINE,
    INFERENCE_ENGINE_HEADER,
    MODEL_COLD_START_COUNT_HEADER,
    MODEL_COLD_START_HEADER,
    MODEL_ID_HEADER,
    MODEL_LOAD_DETAILS_HEADER,
    MODEL_LOAD_TIME_HEADER,
    WORKFLOW_ID_HEADER,
)

MODEL_LOAD_EVENTS: contextvars.ContextVar[Optional[List[Tuple[str, bool, float]]]] = (
    contextvars.ContextVar("model_load_events", default=None)
)
REQUEST_WORKFLOW_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_workflow_id", default=None
)
REQUESTED_MODEL_ID: contextvars.ContextVar[Optional[Tuple[str, str]]] = (
    contextvars.ContextVar("requested_model_id", default=None)
)
REMOTE_MODEL_LOADS: contextvars.ContextVar[Optional[List[Any]]] = (
    contextvars.ContextVar("remote_model_loads", default=None)
)


def set_requested_model_id(model_id: str, *, requested_model_id: str) -> None:
    """Publish the id the current operation asked for a canonical model id.

    Args:
        model_id: Canonical model id the gateway serves.
        requested_model_id: Model id as the operation named it, alias included.
    """
    REQUESTED_MODEL_ID.set((model_id, requested_model_id))


def record_model_load(model_id: str, *, cold_start: bool, load_time_s: float) -> None:
    """Record that the current request made a model ready.

    Args:
        model_id: Model the request uses; when it is the canonical id published
            by ``set_requested_model_id``, the requested id is recorded.
        cold_start: Whether this request loaded the model.
        load_time_s: Seconds the load took.
    """
    events = MODEL_LOAD_EVENTS.get()
    if events is None:
        return

    requested = REQUESTED_MODEL_ID.get()
    if requested is not None and requested[0] == model_id:
        model_id = requested[1]
    events.append((model_id, cold_start, load_time_s))


def summarize_model_load_entries(
    entries: List[Tuple[str, float]], max_detail_bytes: int = 4096
) -> Tuple[float, Optional[str]]:
    """Sum load times and serialize them for ``X-Model-Load-Details``.

    Args:
        entries: ``(model_id, load_time)`` pairs.
        max_detail_bytes: Longest detail string that is still returned.

    Returns:
        Total load time and the JSON detail, or None when the detail is too long.
    """
    total = sum(load_time for _, load_time in entries)
    detail = json.dumps(
        [{"m": model_id, "t": load_time} for model_id, load_time in entries]
    )
    if len(detail) > max_detail_bytes:
        detail = None

    return total, detail


def build_model_response_headers(
    local_model_ids: set,
    local_cold_start_entries: List[Tuple[str, float]],
    remote_model_ids: set,
    remote_cold_start_entries: List[Tuple[str, float]],
    remote_cold_start_count: int,
    remote_cold_start_total_load_time: float,
) -> Dict[str, str]:
    """Build the legacy model headers from local and remote model loads.

    Args:
        local_model_ids: Models this server used for the request.
        local_cold_start_entries: ``(model_id, load_time)`` of local cold starts.
        remote_model_ids: Models remote servers used for the request.
        remote_cold_start_entries: ``(model_id, load_time)`` reported remotely.
        remote_cold_start_count: Cold starts reported remotely.
        remote_cold_start_total_load_time: Load time reported remotely.

    Returns:
        Header names mapped to their values.
    """
    response_headers = {
        MODEL_COLD_START_HEADER: "false",
        MODEL_COLD_START_COUNT_HEADER: "0",
    }
    model_ids = sorted(local_model_ids | remote_model_ids)
    if model_ids:
        response_headers[MODEL_ID_HEADER] = ",".join(model_ids)
    local_cold_start_count = len(local_cold_start_entries)
    cold_start_count = local_cold_start_count + remote_cold_start_count
    response_headers[MODEL_COLD_START_COUNT_HEADER] = str(cold_start_count)
    if cold_start_count == 0:
        return response_headers

    response_headers[MODEL_COLD_START_HEADER] = "true"
    local_load_time = sum(load_time for _, load_time in local_cold_start_entries)
    response_headers[MODEL_LOAD_TIME_HEADER] = str(
        local_load_time + remote_cold_start_total_load_time
    )
    detailed_entries = local_cold_start_entries + remote_cold_start_entries
    if len(detailed_entries) != cold_start_count:
        return response_headers

    _, detail = summarize_model_load_entries(entries=detailed_entries)
    if detail is not None:
        response_headers[MODEL_LOAD_DETAILS_HEADER] = detail

    return response_headers


def _request_headers(events: List[Tuple[str, bool, float]]) -> Dict[str, str]:
    headers = build_model_response_headers(
        local_model_ids={model_id for model_id, _, _ in events},
        local_cold_start_entries=[
            (model_id, load_time_s)
            for model_id, cold_start, load_time_s in events
            if cold_start
        ],
        remote_model_ids=set(),
        remote_cold_start_entries=[],
        remote_cold_start_count=0,
        remote_cold_start_total_load_time=0.0,
    )
    workflow_id = REQUEST_WORKFLOW_ID.get()
    if workflow_id:
        headers[WORKFLOW_ID_HEADER] = workflow_id
    headers[INFERENCE_ENGINE_HEADER] = INFERENCE_ENGINE

    return headers


class ModelLoadHeadersMiddleware:
    """Raw ASGI middleware stamping the model, workflow and engine headers."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        events: List[Tuple[str, bool, float]] = []

        async def _send(message) -> None:
            if message["type"] == "http.response.start":
                added = {
                    name.lower().encode("latin-1"): value.encode("latin-1")
                    for name, value in _request_headers(events).items()
                }
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in added
                ]
                headers.extend(added.items())
                message = {**message, "headers": headers}
            await send(message)

        events_token = MODEL_LOAD_EVENTS.set(events)
        requested_token = REQUESTED_MODEL_ID.set(None)
        workflow_token = REQUEST_WORKFLOW_ID.set(None)
        remote_token = REMOTE_MODEL_LOADS.set([])
        try:
            await self.app(scope, receive, _send)
        finally:
            REMOTE_MODEL_LOADS.reset(remote_token)
            REQUEST_WORKFLOW_ID.reset(workflow_token)
            REQUESTED_MODEL_ID.reset(requested_token)
            MODEL_LOAD_EVENTS.reset(events_token)
