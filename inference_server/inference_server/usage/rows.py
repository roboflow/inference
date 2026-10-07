"""Nested usage rows of a request or pipeline run.

A request handler or a pipeline binds a ``UsageScope`` for its duration; the
model calls, workflow runs and custom Python block runs made inside it record
``model``, ``workflows`` and ``workflow_block`` rows of their own on the
scope's collector, attributed the way the legacy server attributed them.
"""

import contextvars
import json
import logging
import math
import numbers
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from inference_sdk.config import apply_duration_minimum, execution_id
from inference_server import configuration
from inference_server.legacy.errors import redact_text
from inference_server.legacy.telemetry_recording import _recorded_error_type
from inference_server.usage.payload_helpers import sha256_hash

try:
    from streamvision.stream.session import stream_session_id
except ImportError:
    stream_session_id = None

logger = logging.getLogger(__name__)

MODEL_CATEGORY = "model"
WORKFLOWS_CATEGORY = "workflows"
WORKFLOW_BLOCK_CATEGORY = "workflow_block"
UNKNOWN_RESOURCE_ID = "unknown"
WORKFLOW_API_KEY_PARAMETER = "workflows_core.api_key"
MAX_ERROR_MESSAGE_LENGTH = 512
SERVERLESS_MINIMUM_DURATION_S = 0.1
BLOCK_DURATION_SOURCE_WALL_CLOCK = "decorator_wall_clock"
BLOCK_EXECUTION_MODE_BY_DURATION_SOURCE = {
    "local_runtime": "local",
    "remote_runtime": "modal",
    "client_wall_clock": "modal",
    "unavailable": "modal",
}
MEGAPIXEL_BUCKET_UNKNOWN = "unknown"
MEGAPIXEL_BUCKET_OVERFLOW = "8+"
MEGAPIXEL_BUCKET_UPPER_BOUNDS: Tuple[Tuple[str, float], ...] = (
    ("0-0.25", 0.25),
    ("0.25-0.5", 0.5),
    ("0.5-1", 1.0),
    ("1-2", 2.0),
    ("2-4", 4.0),
    ("4-8", 8.0),
)


@dataclass
class UsageScope:
    """Request-level attribution shared by the nested rows of one request.

    Args:
        collector: ``UsageCollector`` the rows are recorded on.
        api_key: Key of the request; a nested row without a key of its own
            is attributed to it.
        billable: Whether the request opted out of billing with a valid
            service secret; nested rows inherit it.
        source: ``source`` tag of the request.
        source_info: ``source_info`` tag of the request, also reported as the
            internal service name.
        service_secret: Secret the request presented.
        exec_session_id: Execution session of the request, written to rows
            recorded from a thread that does not carry it.
        stream_session_id: Session of the pipeline the rows belong to, bound
            when a row is recorded from a thread that does not carry it.
    """

    collector: Any
    api_key: Optional[str]
    billable: bool = True
    source: Optional[str] = None
    source_info: Optional[str] = None
    service_secret: Optional[str] = None
    exec_session_id: Optional[str] = None
    stream_session_id: Optional[str] = None


USAGE_SCOPE: contextvars.ContextVar[Optional[UsageScope]] = contextvars.ContextVar(
    "usage_scope", default=None
)
WORKFLOW_PREVIEW: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "usage_workflow_preview", default=False
)


@contextmanager
def bound_scope(scope: Optional[UsageScope]) -> Iterator[None]:
    """Bind the usage scope of the current context.

    Args:
        scope: Scope the nested rows recorded inside are attributed to.

    Yields:
        Nothing; the previous scope is restored on exit.
    """
    token = USAGE_SCOPE.set(scope)
    try:
        yield
    finally:
        USAGE_SCOPE.reset(token)


@contextmanager
def bound_workflow_preview(is_preview: bool) -> Iterator[None]:
    """Flag the custom Python block rows of a workflow run as preview.

    A run inside a preview run cannot clear the flag.

    Args:
        is_preview: Whether the run is a preview.

    Yields:
        Nothing; the previous flag is restored on exit.
    """
    if not is_preview or WORKFLOW_PREVIEW.get():
        yield
        return

    token = WORKFLOW_PREVIEW.set(True)
    try:
        yield
    finally:
        WORKFLOW_PREVIEW.reset(token)


@contextmanager
def bound_stream_session(session_id: Optional[str]) -> Iterator[None]:
    """Bind the stream session of the current context.

    Args:
        session_id: Session to bind; nothing is bound without the stream
            session variable.

    Yields:
        Nothing; the previous session is restored on exit.
    """
    if stream_session_id is None:
        yield
        return

    token = stream_session_id.set(session_id)
    try:
        yield
    finally:
        stream_session_id.reset(token)


def current_stream_session_id() -> Optional[str]:
    """Stream session of the current context, if any.

    Returns:
        The session identifier, or None.
    """
    if stream_session_id is None:
        return None

    session_id = stream_session_id.get()

    return session_id


def execution_duration(raw: float) -> float:
    """Apply the serverless billing floor to a measured duration.

    Args:
        raw: Measured seconds.

    Returns:
        ``raw``, raised to the floor on GCP serverless unless the request
        disabled the floor.
    """
    if not configuration.GCP_SERVERLESS:
        return raw
    if apply_duration_minimum.get(None) is False:
        return raw

    floored = max(raw, SERVERLESS_MINIMUM_DURATION_S)

    return floored


def error_status_code(error: Any) -> Optional[int]:
    """HTTP status of an error or response, when it carries one in 400-599.

    Args:
        error: Exception or response; its ``inner_error`` is consulted too.

    Returns:
        The status, or None.
    """
    for candidate in (error, getattr(error, "inner_error", None)):
        status_code = getattr(candidate, "status_code", None)
        if (
            isinstance(status_code, numbers.Integral)
            and not isinstance(status_code, bool)
            and 400 <= status_code <= 599
        ):
            return int(status_code)

    return None


def exception_error_details(error: BaseException, secrets: tuple) -> Dict[str, Any]:
    """Error fields of a row recording a failed call.

    Args:
        error: What the call raised.
        secrets: Values redacted from the message.

    Returns:
        ``error``, ``error_type`` and, when known, ``error_status_code``.
    """
    error_type = _recorded_error_type(error)
    inner_error_type = getattr(error, "inner_error_type", None)
    if isinstance(inner_error_type, str) and inner_error_type:
        error_type = inner_error_type
    message = redact_text(str(error), secrets)[:MAX_ERROR_MESSAGE_LENGTH]
    details = {"error": f"{error_type}: {message}", "error_type": error_type}
    status_code = error_status_code(error)
    if status_code is not None:
        details["error_status_code"] = status_code

    return details


def workflow_steps(specification: dict) -> List[str]:
    """Steps of a workflow in the ``type:name`` form the rows carry.

    Args:
        specification: Workflow definition.

    Returns:
        One entry per well-formed step.
    """
    entries = specification.get("steps")
    if not isinstance(entries, list):
        return []
    steps = [
        f"{step.get('type', 'unknown')}:{step.get('name', 'unknown')}"
        for step in entries
        if isinstance(step, dict)
    ]

    return steps


def steps_resource_id(steps: List[str]) -> str:
    """Attribution of a workflow run without an identifier.

    Args:
        steps: Steps of the workflow, as ``workflow_steps`` lists them.

    Returns:
        The short hash of the step list.
    """
    resource_id = sha256_hash(json.dumps({"steps": steps}, sort_keys=True))

    return resource_id


def megapixel_bucket(height: Any, width: Any) -> str:
    """Bucket of an input of the given size.

    Args:
        height: Input height in pixels.
        width: Input width in pixels.

    Returns:
        The bucket name, ``unknown`` when a dimension is not a positive number.
    """
    if not _is_positive_dimension(height) or not _is_positive_dimension(width):
        return MEGAPIXEL_BUCKET_UNKNOWN
    megapixels = (int(height) * int(width)) / 1_000_000.0
    for bucket_name, upper_bound in MEGAPIXEL_BUCKET_UPPER_BOUNDS:
        if megapixels <= upper_bound:
            return bucket_name

    return MEGAPIXEL_BUCKET_OVERFLOW


def add_megapixel_bucket(
    buckets: Dict[str, Dict[str, Any]],
    bucket: str,
    *,
    frames: int,
    duration: float,
) -> None:
    """Add frames and their duration to one bucket.

    Args:
        buckets: Buckets of the row, updated in place.
        bucket: Bucket name.
        frames: Frames to add.
        duration: Seconds to add.
    """
    existing = buckets.get(bucket)
    if existing is None:
        buckets[bucket] = {
            "processed_frames": int(frames),
            "execution_duration": float(duration),
        }
        return

    existing["processed_frames"] += int(frames)
    existing["execution_duration"] += float(duration)


def record_model_usage(
    scope: UsageScope,
    *,
    model_id: Optional[str],
    api_key: Optional[str],
    frames: int,
    duration: float,
    details: Dict[str, Any],
    megapixel_buckets: Optional[Dict[str, Dict[str, Any]]],
    error: Optional[BaseException] = None,
) -> None:
    """Record one ``model`` row increment.

    Args:
        scope: Scope of the request the call was made for.
        model_id: Model as requested; ``unknown`` when empty.
        api_key: Key the model was called with; the scope's when None.
        frames: Images of the call.
        duration: Seconds the call took; floored on serverless.
        details: Labels of the model: architecture, variant, task type and
            fixed input size, when known.
        megapixel_buckets: Per bucket frame and duration counters of the call;
            when None the frames land in the ``unknown`` bucket.
        error: What the call raised, if it failed.
    """
    floored = execution_duration(duration)
    if megapixel_buckets is None:
        megapixel_buckets = {}
        add_megapixel_bucket(
            megapixel_buckets,
            MEGAPIXEL_BUCKET_UNKNOWN,
            frames=frames,
            duration=floored,
        )
    resource_details = {**_common_details(), **details}
    if scope.source is not None:
        resource_details["source"] = scope.source
    _record(
        scope,
        category=MODEL_CATEGORY,
        resource_id=_resource_id(model_id),
        api_key=api_key or scope.api_key,
        details=resource_details,
        frames=frames,
        duration=floored,
        megapixel_buckets=megapixel_buckets,
        error=error,
    )


def record_workflow_usage(
    scope: UsageScope,
    *,
    workflow: Any,
    workflow_id: Optional[str],
    fps: Any,
    is_preview: bool,
    duration: float,
    error: Optional[Exception] = None,
) -> None:
    """Record one ``workflows`` row increment.

    Args:
        scope: Scope of the request or pipeline the run belongs to.
        workflow: Compiled workflow; its definition and API key attribute the
            row.
        workflow_id: Identifier the engine derived; the hash of the step list
            when None, ``unknown`` without a definition.
        fps: Frames per second of the source the run processes.
        is_preview: Whether the run is a preview.
        duration: Seconds the run took; floored on serverless.
        error: What the run raised, if it failed.
    """
    api_key = _workflow_api_key(workflow) or scope.api_key
    workflow_json = _workflow_json(workflow)
    details = _common_details()
    resource_id = workflow_id or UNKNOWN_RESOURCE_ID
    if workflow_json is not None:
        steps = workflow_steps(workflow_json)
        details["steps"] = steps
        if not workflow_id:
            resource_id = steps_resource_id(steps)
    details["is_preview"] = is_preview
    if not _is_positive_fps(fps):
        fps = 0.0
    _record(
        scope,
        category=WORKFLOWS_CATEGORY,
        resource_id=resource_id,
        api_key=api_key,
        details=details,
        frames=1,
        duration=execution_duration(duration),
        fps=fps,
        is_preview=is_preview,
        error=error,
    )


def record_block_usage(
    scope: UsageScope,
    *,
    block: Any,
    frames: int,
    duration: float,
    measured: Any,
    error: Optional[Exception] = None,
) -> None:
    """Record one ``workflow_block`` row increment of a custom Python block.

    Args:
        scope: Scope of the request or pipeline the run belongs to.
        block: The block instance; its resource id, key and step metadata
            attribute the row.
        frames: Batch elements handed to the run.
        duration: Wall time of the run, billed when nothing measured it.
        measured: Duration the engine measured, with its source, or None.
        error: What the run raised, if it failed.
    """
    details = _common_details()
    block_kind = getattr(block, "_usage_block_kind", None)
    if block_kind:
        details["block_kind"] = str(block_kind)
    block_type = getattr(block, "_workflow_step_type", None) or getattr(
        block, "_usage_block_type", None
    )
    if block_type:
        details["block_type"] = str(block_type)
    step_name = getattr(block, "_workflow_step_name", None)
    if step_name:
        details["step_name"] = str(step_name)
    if measured is None:
        details["duration_source"] = BLOCK_DURATION_SOURCE_WALL_CLOCK
    else:
        duration = measured.duration
        details["duration_source"] = measured.source
        execution_mode = BLOCK_EXECUTION_MODE_BY_DURATION_SOURCE.get(measured.source)
        if execution_mode:
            details["execution_mode"] = execution_mode
    is_preview = WORKFLOW_PREVIEW.get()
    details["is_preview"] = is_preview
    if scope.source is not None:
        details["source"] = scope.source
    _record(
        scope,
        category=WORKFLOW_BLOCK_CATEGORY,
        resource_id=_resource_id(getattr(block, "_usage_resource_id", None)),
        api_key=getattr(block, "_api_key", None) or scope.api_key,
        details=details,
        frames=frames,
        duration=execution_duration(duration),
        is_preview=is_preview,
        error=error,
    )


def _record(
    scope: UsageScope,
    *,
    category: str,
    resource_id: str,
    api_key: Optional[str],
    details: Dict[str, Any],
    frames: int,
    duration: float,
    fps: float = 0.0,
    is_preview: bool = False,
    megapixel_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
    error: Optional[BaseException],
) -> None:
    if scope.source_info is not None:
        details["source_info"] = scope.source_info
    error_details: Dict[str, Any] = {}
    if error is not None:
        error_details = exception_error_details(
            error, (api_key, scope.api_key, scope.service_secret)
        )
    details.update(error_details)
    row = {
        "api_key": api_key or "",
        "category": category,
        "resource_id": resource_id,
        "resource_details": details,
        "frames": frames,
        "execution_duration": duration,
        "fps": fps,
        "source_duration": _source_duration(frames, fps),
        "billable": scope.billable,
        "is_preview": is_preview,
        "error_type": error_details.get("error_type"),
        "error_status_code": error_details.get("error_status_code"),
        "roboflow_service_name": scope.source_info,
        "roboflow_internal_secret": scope.service_secret,
        "megapixel_buckets": megapixel_buckets,
    }
    with _request_context(scope):
        scope.collector.record_usage(**row)


@contextmanager
def _request_context(scope: UsageScope) -> Iterator[None]:
    token = execution_id.set(scope.exec_session_id)
    try:
        if scope.stream_session_id and current_stream_session_id() is None:
            with bound_stream_session(scope.stream_session_id):
                yield
            return
        yield
    finally:
        execution_id.reset(token)


def _common_details() -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    if configuration.DEDICATED_DEPLOYMENT_ID:
        details["dedicated_deployment_id"] = configuration.DEDICATED_DEPLOYMENT_ID
    if configuration.DEVICE_ID:
        details["device_id"] = configuration.DEVICE_ID

    return details


def _resource_id(value: Any) -> str:
    if value is None or not str(value).strip():
        return UNKNOWN_RESOURCE_ID

    return str(value).strip()


def _workflow_api_key(workflow: Any) -> Optional[str]:
    init_parameters = getattr(workflow, "init_parameters", None)
    if not isinstance(init_parameters, dict):
        return None

    api_key = init_parameters.get(WORKFLOW_API_KEY_PARAMETER)

    return api_key


def _workflow_json(workflow: Any) -> Optional[dict]:
    workflow_json = getattr(workflow, "workflow_json", None)
    if not isinstance(workflow_json, dict):
        return None

    return workflow_json


def _is_positive_fps(fps: Any) -> bool:
    return isinstance(fps, numbers.Real) and math.isfinite(fps) and fps > 0


def _is_positive_dimension(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _source_duration(frames: int, fps: Any) -> float:
    if not _is_positive_fps(fps):
        return 0.0

    source_duration = frames / fps

    return source_duration
