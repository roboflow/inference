"""Credit-check authorization middleware of the GCP serverless deployment."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from inference_sdk.config import execution_id

from inference_server import configuration, platform_http
from inference_server.auth import validate_api_key
from inference_server.hosted.assume_identity import (
    assume_identity_authorised_workspace_db_id,
    enforce_credits_verification,
    workspace_db_id_is_valid,
)
from inference_server.hosted.common import (
    AUTHENTICATED_METHODS,
    STATIC_PREFIXES,
    UNAUTHORIZED_MESSAGE,
    HostedRequest,
    error_response,
    is_non_billable_internal_request,
    resolve_api_key,
    send_with_workspace_header,
    workspace_id_is_valid,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.middlewares.correlation_id import (
    correlation_id,
    request_start_time,
)
from inference_server.middlewares.headers import PROCESSING_TIME_HEADER

AUTH_CACHE_TTL_SECONDS = 3600
SHORT_AUTH_CACHE_TTL_SECONDS = 60
SKIP_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/info",
        "/healthz",
        "/readiness",
        "/metrics",
        "/openapi.json",
        "/model/registry",
    }
)
WORKFLOW_DESCRIPTION_PATHS = frozenset(
    {"/workflows/blocks/describe", "/workflows/definition/schema"}
)
SERVERLESS_UNAUTHORIZED_MESSAGE = (
    "Unauthorized api_key. This key is not authorized for serverless inference."
)
CREDITS_DENIED_MESSAGE = (
    "This workspace cannot currently spend credits for serverless inference. "
    "Verify billing or credit cap settings."
)
INCOMPLETE_USAGE_CHECK_MESSAGE = (
    "Serverless authorization failed because the usage check returned "
    "incomplete data."
)
UNEXPECTED_USAGE_CHECK_STATUS_MESSAGE = (
    "Serverless authorization failed because the usage check returned an "
    "unexpected status ({status_code})."
)
INVALID_WORKSPACE_MESSAGE = (
    "Serverless authorization failed because workspace lookup returned an "
    "invalid identity."
)
INVALID_WORKSPACE_DB_MESSAGE = (
    "Serverless authorization failed because workspace lookup returned an "
    "invalid internal identity."
)


@dataclass
class AuthorizationCacheEntry:
    expires_at: float
    workspace_id: Optional[str]
    workspace_db_id: Optional[str] = None
    status_code: int = 200
    message: Optional[str] = None


@dataclass
class UsageCheckResult:
    status_code: int
    workspace_id: Optional[str] = None
    workspace_db_id: Optional[str] = None
    under_cap: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class Denial:
    status_code: int
    message: str
    workspace_id: Optional[str] = None


_cache: Dict[Tuple[str, bool], AuthorizationCacheEntry] = {}
_now = time.monotonic
_sleep = time.sleep


async def _skip_check(request: HostedRequest) -> bool:
    if request.method not in AUTHENTICATED_METHODS:
        return True
    if request.path in SKIP_PATHS or request.path.startswith(STATIC_PREFIXES):
        return True
    if request.path not in WORKFLOW_DESCRIPTION_PATHS:
        return False
    if request.method == "GET":
        return True
    if not request.has_json_body:
        return False

    json_params = await request.json()
    return not json_params.get("dynamic_blocks_definitions")


def _json_payload(response: Any) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _usage_check_response(url: str) -> Any:
    attempts = max(configuration.TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES, 1)
    for attempt in range(1, attempts + 1):
        try:
            response = platform_http._platform_request(
                "get",
                platform_http.wrap_url(url),
                headers=platform_http.build_api_headers(),
                timeout=platform_http.API_REQUEST_TIMEOUT_S,
            )
        except LegacyHTTPError as error:
            if (
                error.status_code != 503
                or not configuration.RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API
                or attempt == attempts
            ):
                raise
            _sleep(configuration.TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL)
            continue
        if (
            response.status_code in (401, 402)
            or response.status_code not in configuration.TRANSIENT_ROBOFLOW_API_ERRORS
            or attempt == attempts
        ):
            return response
        _sleep(configuration.TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL)

    return response


def _usage_check(api_key: str) -> UsageCheckResult:
    url = platform_http._add_params_to_url(
        url=f"{configuration.API_BASE_URL.rstrip('/')}/serverless/usage-check",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    response = _usage_check_response(url)
    if response.status_code == 401:
        return UsageCheckResult(status_code=401)

    if response.status_code == 402:
        payload = _json_payload(response)
        workspace = payload.get("workspace")
        workspace_db_id = payload.get("workspaceId")
        if workspace is None:
            workspace = workspace_db_id
        return UsageCheckResult(
            status_code=402,
            workspace_id=workspace if workspace_id_is_valid(workspace) else None,
            workspace_db_id=(
                workspace_db_id if workspace_db_id_is_valid(workspace_db_id) else None
            ),
            under_cap=payload.get("underCap"),
            error=payload.get("error"),
        )

    if not 200 <= response.status_code < 300:
        return UsageCheckResult(status_code=response.status_code)

    payload = _json_payload(response)
    workspace_id = payload.get("workspace")
    workspace_db_id = payload.get("workspaceId")
    if workspace_id is None:
        workspace_id = workspace_db_id

    return UsageCheckResult(
        status_code=200,
        workspace_id=workspace_id,
        workspace_db_id=workspace_db_id,
        under_cap=payload.get("underCap"),
    )


def _store(cache_key: Tuple[str, bool], entry: AuthorizationCacheEntry) -> None:
    if len(_cache) >= configuration.AUTH_CACHE_MAX_SIZE:
        now = _now()
        for key in [k for k, v in _cache.items() if v.expires_at < now]:
            del _cache[key]
    while len(_cache) >= configuration.AUTH_CACHE_MAX_SIZE:
        oldest = min(_cache, key=lambda k: _cache[k].expires_at)
        del _cache[oldest]
    _cache[cache_key] = entry


def _usage_check_result_is_complete(result: UsageCheckResult) -> bool:
    if not workspace_id_is_valid(result.workspace_id):
        return False
    if result.workspace_db_id is not None and not workspace_db_id_is_valid(
        result.workspace_db_id
    ):
        return False
    if configuration.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN and (
        not workspace_db_id_is_valid(result.workspace_db_id)
    ):
        return False

    return result.under_cap is True


async def _authorize_with_credits(
    api_key: str, cache_key: Tuple[str, bool]
) -> Tuple[Optional[Denial], Optional[AuthorizationCacheEntry]]:
    result = await asyncio.to_thread(_usage_check, api_key)
    if result.status_code == 200:
        if not _usage_check_result_is_complete(result):
            return Denial(500, INCOMPLETE_USAGE_CHECK_MESSAGE), None
        entry = AuthorizationCacheEntry(
            expires_at=_now() + AUTH_CACHE_TTL_SECONDS,
            workspace_id=result.workspace_id,
            workspace_db_id=result.workspace_db_id,
        )
        _store(cache_key, entry)
        return None, entry

    if result.status_code == 401:
        entry = AuthorizationCacheEntry(
            expires_at=_now() + SHORT_AUTH_CACHE_TTL_SECONDS,
            workspace_id=None,
            status_code=401,
            message=SERVERLESS_UNAUTHORIZED_MESSAGE,
        )
        _store(cache_key, entry)
        return Denial(401, entry.message), None

    if result.status_code == 402:
        message = CREDITS_DENIED_MESSAGE
        if result.error:
            message = f"{message} {result.error}"
        entry = AuthorizationCacheEntry(
            expires_at=_now() + SHORT_AUTH_CACHE_TTL_SECONDS,
            workspace_id=result.workspace_id,
            workspace_db_id=result.workspace_db_id,
            status_code=402,
            message=message,
        )
        _store(cache_key, entry)
        return Denial(402, message, workspace_id=result.workspace_id), None

    message = UNEXPECTED_USAGE_CHECK_STATUS_MESSAGE.format(
        status_code=result.status_code
    )
    return Denial(500, message), None


async def _authorize_without_credits(
    api_key: str, cache_key: Tuple[str, bool]
) -> Tuple[Optional[Denial], Optional[AuthorizationCacheEntry]]:
    valid, workspace_id = await validate_api_key(api_key)
    if valid:
        entry = AuthorizationCacheEntry(
            expires_at=_now() + AUTH_CACHE_TTL_SECONDS, workspace_id=workspace_id
        )
        _store(cache_key, entry)
        return None, entry

    entry = AuthorizationCacheEntry(
        expires_at=_now() + SHORT_AUTH_CACHE_TTL_SECONDS,
        workspace_id=None,
        status_code=401,
        message=UNAUTHORIZED_MESSAGE,
    )
    _store(cache_key, entry)
    return Denial(401, entry.message), None


async def _authorize(
    request: HostedRequest, api_key: str
) -> Tuple[Optional[Denial], Optional[AuthorizationCacheEntry], bool]:
    enforce = not is_non_billable_internal_request(request)
    cache_key = (api_key, enforce)
    cache_entry = _cache.get(cache_key)
    cache_entry_is_fresh = cache_entry is not None and cache_entry.expires_at >= _now()
    needs_workspace_db_refresh = (
        bool(configuration.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN)
        and enforce
        and cache_entry_is_fresh
        and cache_entry.status_code == 200
        and cache_entry.workspace_db_id is None
    )
    if cache_entry_is_fresh and not needs_workspace_db_refresh:
        if cache_entry.status_code != 200:
            denial = Denial(
                cache_entry.status_code,
                cache_entry.message or UNAUTHORIZED_MESSAGE,
                workspace_id=cache_entry.workspace_id,
            )
            return denial, None, enforce
        return None, cache_entry, enforce

    if enforce:
        denial, entry = await _authorize_with_credits(api_key, cache_key)
    else:
        denial, entry = await _authorize_without_credits(api_key, cache_key)
    if denial is not None:
        return denial, None, enforce

    if not workspace_id_is_valid(entry.workspace_id):
        _cache.pop(cache_key, None)
        return Denial(500, INVALID_WORKSPACE_MESSAGE), None, enforce
    if entry.workspace_db_id is not None and not workspace_db_id_is_valid(
        entry.workspace_db_id
    ):
        _cache.pop(cache_key, None)
        return Denial(500, INVALID_WORKSPACE_DB_MESSAGE), None, enforce

    return None, entry, enforce


def _attach_observability_headers(response: Any) -> None:
    request_id = correlation_id.get()
    if request_id is not None:
        response.headers[configuration.CORRELATION_ID_HEADER] = request_id
    started_at = request_start_time.get()
    if started_at is not None:
        response.headers[PROCESSING_TIME_HEADER] = str(time.perf_counter() - started_at)
    execution_id_value = execution_id.get()
    if configuration.EXECUTION_ID_HEADER and execution_id_value is not None:
        response.headers[configuration.EXECUTION_ID_HEADER] = execution_id_value


class ServerlessAuthMiddleware:
    """Raw ASGI middleware enforcing the serverless usage check per request.

    Every GET/POST outside the skip list must carry an API key (query,
    Bearer header or JSON body). Billable requests are verified against
    ``/serverless/usage-check``; authenticated non-billable requests only
    resolve their workspace. Outcomes are cached per ``(api_key, billable)``
    for one hour on success and one minute on denial.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = HostedRequest(scope, receive)
        if await _skip_check(request):
            await self.app(scope, request.receive, send)
            return

        api_key = await resolve_api_key(request)
        if api_key is None:
            response = error_response(401, UNAUTHORIZED_MESSAGE)
            _attach_observability_headers(response)
            await response(scope, request.receive, send)
            return

        denial, entry, enforce = await _authorize(request, api_key)
        if denial is not None:
            response = error_response(
                denial.status_code, denial.message, workspace_id=denial.workspace_id
            )
            _attach_observability_headers(response)
            await response(scope, request.receive, send)
            return

        enforce_token = enforce_credits_verification.set(enforce)
        identity_token = None
        if workspace_db_id_is_valid(entry.workspace_db_id):
            identity_token = assume_identity_authorised_workspace_db_id.set(
                entry.workspace_db_id
            )
        try:
            await self.app(
                scope,
                request.receive,
                send_with_workspace_header(send, entry.workspace_id),
            )
        finally:
            if identity_token is not None:
                assume_identity_authorised_workspace_db_id.reset(identity_token)
            enforce_credits_verification.reset(enforce_token)
