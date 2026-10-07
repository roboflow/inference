"""One ``request`` usage row per request of a legacy-compatible route.

The hook also binds the request's ``UsageScope`` for the duration of the
handler, so the model calls and workflow runs made inside it record their own
``model``, ``workflows`` and ``workflow_block`` rows attributed to the request.
"""

import functools
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

from fastapi import Request

from inference_sdk.config import execution_id, outbound_service_secret
from inference_server import configuration
from inference_server.hosted.common import (
    _coerce_optional_bool,
    service_secret_is_valid,
)
from inference_server.legacy.common import resolve_api_key
from inference_server.legacy.errors import add_redaction_values
from inference_server.middlewares.model_load import REQUESTED_MODEL_ID
from inference_server.usage.rows import (
    UNKNOWN_RESOURCE_ID,
    UsageScope,
    bound_scope,
    error_status_code,
    exception_error_details,
    execution_duration,
    workflow_steps,
)

logger = logging.getLogger(__name__)

REQUEST_CATEGORY = "request"
EXTERNAL_SOURCE = "external"
SPECIFICATION_RESOURCE_ID_PREFIX = "sha:"
RESPONSE_ERROR_TYPE = "HTTPResponseError"
BODY_FIELDS = ("api_key", "model_id", "source", "source_info")
WORKFLOW_REQUEST_ARGUMENT = "workflow_request"


def report_request_usage(fn: Any) -> Any:
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
    specification; its row also carries the workflow steps, the preview flag
    and the workspace. While the handler runs, the request's usage scope is
    bound so nested rows inherit its key, billing intent and source tags; a
    request that opted out of billing with a valid service secret forwards
    the opt-out to the remote steps it runs.

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

        try:
            subject = await _subject_of(request, kwargs)
            scope = _scope_of(collector, request, kwargs, subject)
        except Exception as failure:
            logger.debug(
                "Usage of the request was not recorded: %s", type(failure).__name__
            )
            return await fn(*args, **kwargs)

        started = time.perf_counter()
        with bound_scope(scope), _forwarded_opt_out(scope):
            try:
                response = await fn(*args, **kwargs)
            except Exception as error:
                _record(
                    request,
                    kwargs,
                    subject=subject,
                    scope=scope,
                    duration=time.perf_counter() - started,
                    error=error,
                )
                raise
        _record(
            request,
            kwargs,
            subject=subject,
            scope=scope,
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


@contextmanager
def _forwarded_opt_out(scope: UsageScope) -> Iterator[None]:
    if scope.billable:
        yield
        return

    token = outbound_service_secret.set(configuration.ROBOFLOW_SERVICE_SECRET)
    try:
        yield
    finally:
        outbound_service_secret.reset(token)


def _scope_of(
    collector: Any, request: Request, kwargs: Dict[str, Any], subject: Any
) -> UsageScope:
    query = request.query_params
    api_key = resolve_api_key(
        request, query.get("api_key"), getattr(subject, "api_key", None)
    )
    service_secret = query.get("service_secret")
    if "countinference" in kwargs:
        countinference = kwargs["countinference"]
    else:
        countinference = query.get("countinference")
    scope = UsageScope(
        collector=collector,
        api_key=api_key,
        billable=not _non_billable_intent(countinference, service_secret),
        source=_source_tag(query, subject, "source"),
        source_info=_source_tag(query, subject, "source_info"),
        service_secret=service_secret,
        exec_session_id=execution_id.get(),
    )

    return scope


def _record(
    request: Request,
    kwargs: Dict[str, Any],
    *,
    subject: Any,
    scope: UsageScope,
    duration: float,
    error: Optional[Exception] = None,
    response: Any = None,
) -> None:
    try:
        if error is not None:
            add_redaction_values(scope.api_key, scope.service_secret)
            error_details = exception_error_details(
                error, (scope.api_key, scope.service_secret)
            )
        else:
            error_details = _response_error_details(response)
        row = _row(
            request,
            kwargs,
            subject=subject,
            scope=scope,
            duration=duration,
            error_details=error_details,
        )
        scope.collector.record_usage(**row)
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
    scope: UsageScope,
    duration: float,
    error_details: Dict[str, Any],
) -> Dict[str, Any]:
    workflow = WORKFLOW_REQUEST_ARGUMENT in kwargs
    specification = _workflow_specification(request, subject) if workflow else None

    details: Dict[str, Any] = {}
    if configuration.DEDICATED_DEPLOYMENT_ID:
        details["dedicated_deployment_id"] = configuration.DEDICATED_DEPLOYMENT_ID
    if configuration.DEVICE_ID:
        details["device_id"] = configuration.DEVICE_ID
    if specification is not None:
        details["steps"] = workflow_steps(specification)
    if workflow:
        details["is_preview"] = _is_preview(subject)
        if kwargs.get("workspace_name"):
            details["workspace_id"] = kwargs["workspace_name"]
    if scope.source is not None:
        details["source"] = scope.source
    model_id = getattr(subject, "model_id", None)
    if isinstance(model_id, str) and model_id.startswith("sam3/"):
        details["execution_mode"] = configuration.SAM3_EXEC_MODE
    if scope.source_info is not None:
        details["source_info"] = scope.source_info
    details.update(error_details)

    row = {
        "api_key": scope.api_key or "",
        "category": REQUEST_CATEGORY,
        "resource_id": _resource_id(kwargs, subject, specification),
        "resource_details": details,
        "frames": 1,
        "execution_duration": execution_duration(duration),
        "billable": scope.billable,
        "error_type": error_details.get("error_type"),
        "error_status_code": error_details.get("error_status_code"),
        "roboflow_service_name": scope.source_info,
        "roboflow_internal_secret": scope.service_secret,
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


def _response_error_details(response: Any) -> Dict[str, Any]:
    status_code = error_status_code(response)
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
