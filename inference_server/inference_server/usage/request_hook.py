"""One usage row per request of a legacy-compatible model or workflow route."""

import contextvars
import functools
import hashlib
import json
import logging
import numbers
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from fastapi import Request

from inference_sdk.config import apply_duration_minimum
from inference_server import configuration
from inference_server.hosted.common import (
    _coerce_optional_bool,
    service_secret_is_valid,
)
from inference_server.legacy.common import resolve_api_key
from inference_server.legacy.errors import add_redaction_values, redact_text
from inference_server.legacy.telemetry_recording import _recorded_error_type
from inference_server.middlewares.model_load import REQUESTED_MODEL_ID
from inference_server.usage.payload_helpers import (
    _custom_python_identity,
    _merge_entries,
    _model_identity,
)

logger = logging.getLogger(__name__)

REQUEST_CATEGORY = "request"
EXTERNAL_SOURCE = "external"
UNKNOWN_RESOURCE_ID = "unknown"
SPECIFICATION_RESOURCE_ID_PREFIX = "sha:"
MAX_ERROR_MESSAGE_LENGTH = 512
SERVERLESS_MINIMUM_DURATION_S = 0.1
RESPONSE_ERROR_TYPE = "HTTPResponseError"
MODEL_SUMMED_FIELDS = ("frames", "execution_duration")
CUSTOM_PYTHON_SUMMED_FIELDS = ("execution_duration",)
BODY_FIELDS = ("api_key", "model_id", "source", "source_info")
WORKFLOW_REQUEST_ARGUMENT = "workflow_request"

MODEL_INVOCATIONS: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = (
    contextvars.ContextVar("usage_model_invocations", default=None)
)
CUSTOM_PYTHON_RUNS: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = (
    contextvars.ContextVar("usage_custom_python_runs", default=None)
)
_HOLDERS_LOCK = threading.Lock()


def record_model_invocation(entry: Dict[str, Any]) -> None:
    """Add one model invocation to the row of the request being served.

    An invocation of a model the request already invoked is combined with the
    earlier entry the way rows combine their ``models`` lists: ``frames`` and
    ``execution_duration`` are summed, every other field is the latest one.

    Args:
        entry: ``models`` entry of the row: model id and labels, input size,
            latency, duration and frames of the invocation.
    """
    invocations = MODEL_INVOCATIONS.get()
    if invocations is None:
        return

    add_model_invocation(invocations, entry)


def add_model_invocation(
    invocations: List[Dict[str, Any]], entry: Dict[str, Any]
) -> None:
    """Combine one model invocation into the ``models`` list of a request.

    Safe to call from several threads of one request at once.

    Args:
        invocations: ``models`` list captured when the request started.
        entry: ``models`` entry of the invocation.
    """
    with _HOLDERS_LOCK:
        invocations[:] = _merge_entries(
            [*invocations, entry],
            identity=_model_identity,
            summed_fields=MODEL_SUMMED_FIELDS,
        )


def add_custom_python_run(runs: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    """Combine one custom Python block run into the ``custom_python`` list.

    Runs of one step (same ``block_type`` and ``step_name``) sum their
    ``execution_duration``. Safe to call from several threads of one request
    at once.

    Args:
        runs: ``custom_python`` list captured when the request started.
        entry: ``custom_python`` entry of the run.
    """
    with _HOLDERS_LOCK:
        runs[:] = _merge_entries(
            [*runs, entry],
            identity=_custom_python_identity,
            summed_fields=CUSTOM_PYTHON_SUMMED_FIELDS,
        )


def report_request_usage(fn: Callable) -> Callable:
    """Record one ``request`` usage row for every call of a route handler.

    The row is recorded through the collector on ``request.app.state`` when
    the handler returns and when it raises an ``Exception``, which is then
    re-raised. Without a collector nothing is recorded. A failure while
    recording is logged and never changes the response. A handler without an
    ``inference_request`` argument is attributed from the JSON object in the
    request body (``api_key``, ``model_id``, ``source``, ``source_info``),
    except the catch-all route, which reads its image from the body. A
    workflow run handler (``workflow_request`` argument) is attributed to the
    workflow: the path's or the body's ``workflow_id``, else the hash of the
    specification; its row also carries the workflow steps, the preview flag,
    the workspace and the custom Python blocks the run executed.

    Args:
        fn: Route handler called with the request as ``request``.

    Returns:
        The wrapped handler.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        request = _request_of(args, kwargs)
        collector = _collector_of(request)
        if collector is None:
            return await fn(*args, **kwargs)

        invocations: List[Dict[str, Any]] = []
        custom_python: List[Dict[str, Any]] = []
        invocations_token = MODEL_INVOCATIONS.set(invocations)
        custom_python_token = CUSTOM_PYTHON_RUNS.set(custom_python)
        started = time.perf_counter()
        try:
            response = await fn(*args, **kwargs)
        except Exception as error:
            await _record(
                collector,
                request,
                kwargs,
                invocations=invocations,
                custom_python=custom_python,
                duration=time.perf_counter() - started,
                error=error,
            )
            raise
        finally:
            CUSTOM_PYTHON_RUNS.reset(custom_python_token)
            MODEL_INVOCATIONS.reset(invocations_token)
        await _record(
            collector,
            request,
            kwargs,
            invocations=invocations,
            custom_python=custom_python,
            duration=time.perf_counter() - started,
            response=response,
        )

        return response

    return wrapper


def _request_of(args: tuple, kwargs: Dict[str, Any]) -> Optional[Request]:
    for value in (kwargs.get("request"), *args, *kwargs.values()):
        if isinstance(value, Request):
            return value

    return None


def _collector_of(request: Optional[Request]) -> Optional[Any]:
    if request is None:
        return None
    try:
        return getattr(request.app.state, "usage_collector", None)
    except Exception:
        return None


async def _record(
    collector: Any,
    request: Request,
    kwargs: Dict[str, Any],
    *,
    invocations: List[Dict[str, Any]],
    custom_python: List[Dict[str, Any]],
    duration: float,
    error: Optional[Exception] = None,
    response: Any = None,
) -> None:
    try:
        subject = await _subject_of(request, kwargs)
        api_key = resolve_api_key(
            request,
            request.query_params.get("api_key"),
            getattr(subject, "api_key", None),
        )
        service_secret = request.query_params.get("service_secret")
        if error is not None:
            add_redaction_values(api_key, service_secret)
            error_details = _exception_error_details(error, (api_key, service_secret))
        else:
            error_details = _response_error_details(response)
        row = _row(
            request,
            kwargs,
            subject=subject,
            api_key=api_key,
            service_secret=service_secret,
            invocations=invocations,
            custom_python=custom_python,
            duration=duration,
            error_details=error_details,
        )
        collector.record_usage(**row)
    except Exception as failure:
        logger.debug(
            "Usage of the request was not recorded: %s", type(failure).__name__
        )


async def _subject_of(request: Request, kwargs: Dict[str, Any]) -> Any:
    inference_request = kwargs.get("inference_request")
    if inference_request is not None or "dataset_id" in kwargs:
        return inference_request
    if WORKFLOW_REQUEST_ARGUMENT in kwargs:
        return kwargs[WORKFLOW_REQUEST_ARGUMENT]
    try:
        body = await request.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None

    return SimpleNamespace(
        **{
            field: body[field] if isinstance(body.get(field), str) else None
            for field in BODY_FIELDS
        }
    )


def _row(
    request: Request,
    kwargs: Dict[str, Any],
    *,
    subject: Any,
    api_key: Optional[str],
    service_secret: Optional[str],
    invocations: List[Dict[str, Any]],
    custom_python: List[Dict[str, Any]],
    duration: float,
    error_details: Dict[str, Any],
) -> Dict[str, Any]:
    query = request.query_params
    if "countinference" in kwargs:
        countinference = kwargs["countinference"]
    else:
        countinference = query.get("countinference")
    billable = not _non_billable_intent(countinference, service_secret)
    source_info = _source_tag(query, subject, "source_info")
    workflow = WORKFLOW_REQUEST_ARGUMENT in kwargs
    specification = _workflow_specification(request, subject) if workflow else None

    details: Dict[str, Any] = {}
    if configuration.DEDICATED_DEPLOYMENT_ID:
        details["dedicated_deployment_id"] = configuration.DEDICATED_DEPLOYMENT_ID
    if configuration.DEVICE_ID:
        details["device_id"] = configuration.DEVICE_ID
    if specification is not None:
        details["steps"] = _workflow_steps(specification)
    if workflow:
        details["is_preview"] = _is_preview(subject)
        if kwargs.get("workspace_name"):
            details["workspace_id"] = kwargs["workspace_name"]
    source = _source_tag(query, subject, "source")
    if source is not None:
        details["source"] = source
    model_id = getattr(subject, "model_id", None)
    if isinstance(model_id, str) and model_id.startswith("sam3/"):
        details["execution_mode"] = configuration.SAM3_EXEC_MODE
    if source_info is not None:
        details["source_info"] = source_info
    details["models"] = invocations
    if workflow:
        details["custom_python"] = custom_python
    details.update(error_details)

    row = {
        "api_key": api_key or "",
        "category": REQUEST_CATEGORY,
        "resource_id": _resource_id(kwargs, subject, specification),
        "resource_details": details,
        "frames": 1,
        "execution_duration": _execution_duration(duration),
        "billable": billable,
        "error_type": error_details.get("error_type"),
        "error_status_code": error_details.get("error_status_code"),
        "roboflow_service_name": source_info,
        "roboflow_internal_secret": service_secret,
    }
    if workflow:
        row["is_preview"] = details["is_preview"]

    return row


def _workflow_specification(request: Request, subject: Any) -> Optional[dict]:
    specification = getattr(subject, "specification", None)
    if isinstance(specification, dict):
        return specification
    specification = getattr(request.state, "workflow_specification", None)
    if isinstance(specification, dict):
        return specification

    return None


def _workflow_steps(specification: dict) -> List[str]:
    entries = specification.get("steps")
    if not isinstance(entries, list):
        return []
    steps = [
        f"{step.get('type', 'unknown')}:{step.get('name', 'unknown')}"
        for step in entries
        if isinstance(step, dict)
    ]

    return steps


def _is_preview(subject: Any) -> bool:
    return getattr(subject, "is_preview", False) is True


def _specification_resource_id(specification: dict) -> str:
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return f"{SPECIFICATION_RESOURCE_ID_PREFIX}{digest}"


def _non_billable_intent(countinference: Any, service_secret: Any) -> bool:
    if _coerce_optional_bool(countinference) is not False:
        return False

    valid = service_secret_is_valid(service_secret)

    return valid


def _meaningful_source(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or value == EXTERNAL_SOURCE:
        return None

    return value


def _source_tag(query: Any, subject: Any, key: str) -> Optional[str]:
    tag = _meaningful_source(query.get(key))
    if tag is None:
        tag = _meaningful_source(getattr(subject, key, None))

    return tag


def _resource_id(
    kwargs: Dict[str, Any], subject: Any, specification: Optional[dict]
) -> str:
    if WORKFLOW_REQUEST_ARGUMENT in kwargs:
        workflow_id = kwargs.get("workflow_id") or getattr(subject, "workflow_id", None)
        if workflow_id:
            return str(workflow_id)
        if specification is not None:
            return _specification_resource_id(specification)
        return UNKNOWN_RESOURCE_ID
    model_id = getattr(subject, "model_id", None)
    if model_id is not None:
        return str(model_id)
    if "dataset_id" in kwargs and "version_id" in kwargs:
        if kwargs["version_id"]:
            return f"{kwargs['dataset_id']}/{kwargs['version_id']}"
        return str(kwargs["dataset_id"])
    requested = REQUESTED_MODEL_ID.get()
    if requested is not None:
        return requested[1]

    return UNKNOWN_RESOURCE_ID


def _execution_duration(raw: float) -> float:
    if not configuration.GCP_SERVERLESS:
        return raw
    if apply_duration_minimum.get(None) is False:
        return raw

    floored = max(raw, SERVERLESS_MINIMUM_DURATION_S)

    return floored


def _error_status_code(error: BaseException) -> Optional[int]:
    for candidate in (error, getattr(error, "inner_error", None)):
        status_code = getattr(candidate, "status_code", None)
        if (
            isinstance(status_code, numbers.Integral)
            and not isinstance(status_code, bool)
            and 400 <= status_code <= 599
        ):
            return int(status_code)

    return None


def _exception_error_details(error: Exception, secrets: tuple) -> Dict[str, Any]:
    error_type = _recorded_error_type(error)
    message = redact_text(str(error), secrets)[:MAX_ERROR_MESSAGE_LENGTH]
    details = {"error": f"{error_type}: {message}", "error_type": error_type}
    status_code = _error_status_code(error)
    if status_code is not None:
        details["error_status_code"] = status_code

    return details


def _response_error_details(response: Any) -> Dict[str, Any]:
    status_code = _error_status_code(response)
    if status_code is None:
        return {}

    details = {
        "error": (
            f"{RESPONSE_ERROR_TYPE}{status_code}: response returned status "
            f"{status_code}"
        ),
        "error_type": RESPONSE_ERROR_TYPE,
        "error_status_code": status_code,
    }

    return details
