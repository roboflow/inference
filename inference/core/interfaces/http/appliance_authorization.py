"""Per-request API key authorization for an on-premise appliance.

An on-premise appliance serves many workspaces from one install. The
appliance's own API (``API_BASE_URL``) is the authority on which API keys
exist and on each workspace's quota. The server asks its usage check about each
key it has not seen recently and keeps the answer for a bounded time. Any key
that API accepts, whose workspace is under its quota, may call the server;
access to a workspace model is still checked per model by the model manager.

The module is used only when ``ON_PREMISE_APPLIANCE`` is on.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from cachetools import TTLCache
from starlette.requests import Request
from starlette.responses import JSONResponse

from inference.core.exceptions import (
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPIUsagePausedError,
)
from inference.core.interfaces.http.api_key_resolution import (
    extract_api_key_from_headers,
)
from inference.core.logger import logger
from inference.core.roboflow_api import (
    ServerlessUsageCheckResponse,
    workspace_db_id_is_valid,
    workspace_id_is_valid,
)

MISSING_API_KEY_MESSAGE = (
    "Missing api_key. Pass it as the api_key query parameter, the api_key JSON "
    "field or an Authorization: Bearer header."
)
CONFLICTING_API_KEYS_MESSAGE = (
    "The request carries different API keys in the api_key query parameter, the "
    "Authorization: Bearer header and the api_key JSON field. Send one API key."
)
UNAUTHORIZED_API_KEY_MESSAGE = "Unauthorized api_key."
FORBIDDEN_API_KEY_MESSAGE = "The api_key is not allowed to use this inference server."
PAUSED_API_KEY_MESSAGE = (
    "Usage of this api_key is paused. Ask your workspace administrator to "
    "re-enable it."
)
GLOBAL_MANAGEMENT_ROUTE_MESSAGE = (
    "This route acts on every workspace that uses this inference server. A "
    "workspace api_key cannot call it on an appliance."
)
QUOTA_EXCEEDED_REASON = "credit_cap_exceeded"
WORKSPACE_RESTRICTED_REASON = "workspace_restricted"
APPLIANCE_API_UNAVAILABLE_RETRY_AFTER_SECONDS = 5

_EXEMPT_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/redoc",
        "/info",
        "/healthz",
        "/readiness",
        "/metrics",
        "/openapi.json",
    }
)
# The async consumer and the start-up probe read the registry to learn which
# models are loaded; it lists model ids only.
_EXEMPT_GET_PATHS = frozenset({"/model/registry"})
# Describing blocks runs no model. It needs a key only when the request
# carries dynamic block definitions, which can hold Python.
_DYNAMIC_BLOCKS_PATHS = frozenset(
    {"/workflows/blocks/describe", "/workflows/definition/schema"}
)
# HEAD is not exempt: Starlette answers HEAD with the GET handler, which runs.
_EXEMPT_METHODS = frozenset({"OPTIONS"})
# These routes act on state that every workspace shares. A workspace key does
# not own it. /model/remove, /model/clear and /clear_cache remove models that
# other workspaces loaded. /notebook/start gives the server notebook token
# (NOTEBOOK_ENABLED). /logs returns the server log lines of every workspace
# (ENABLE_IN_MEMORY_LOGS).
_GLOBAL_MANAGEMENT_PATHS = frozenset(
    {"/model/remove", "/model/clear", "/clear_cache", "/notebook/start", "/logs"}
)


@dataclass(frozen=True)
class ApplianceKeyDecision:
    """The outcome of checking one API key with the appliance API.

    Attributes:
        status_code: 200 when the key is accepted, otherwise the HTTP status the
            server answers with.
        message: The message of the refusal; None when the key is accepted.
        workspace_id: The workspace that owns an accepted key.
        workspace_db_id: The database id of that workspace, when the API sent it.
        retry_after_seconds: Set when the appliance API did not answer, so the
            client may retry.
    """

    status_code: int
    message: Optional[str] = None
    workspace_id: Optional[str] = None
    workspace_db_id: Optional[str] = None
    retry_after_seconds: Optional[int] = None

    @property
    def is_authorized(self) -> bool:
        """Whether the request may continue."""
        return self.status_code == 200

    def to_response(self) -> JSONResponse:
        """Build the refusal answer for this decision.

        Returns:
            The JSON answer with the decision's status code and message.
        """
        headers = {}
        if self.retry_after_seconds is not None:
            headers["Retry-After"] = str(self.retry_after_seconds)

        return JSONResponse(
            status_code=self.status_code,
            content={"status": self.status_code, "message": self.message},
            headers=headers,
        )


UsageCheck = Callable[[str], Awaitable[ServerlessUsageCheckResponse]]


class ApplianceKeyAuthorizer:
    """Decides whether an API key may call the server, with bounded caching.

    The decision comes from the appliance usage check: a valid key whose
    workspace is under its quota is accepted. A workspace with no quota is
    always under it. Accepted keys are kept for ``ttl_seconds`` and refused keys for
    ``negative_ttl_seconds``; each cache holds at most ``max_size`` keys. When
    the appliance API does not answer, the key is refused with 503 and nothing
    is cached. Concurrent requests with the same uncached key share one lookup.

    The authorizer is used from one event loop and is not thread-safe.
    """

    def __init__(
        self,
        usage_check: UsageCheck,
        *,
        ttl_seconds: int,
        negative_ttl_seconds: int,
        max_size: int,
        lookup_timeout_seconds: float,
        max_concurrent_lookups: int,
        timer: Callable[[], float] = time.monotonic,
    ):
        """Create an authorizer.

        Args:
            usage_check: Coroutine function that runs the appliance usage check
                for a key (``get_serverless_usage_check_async``).
            ttl_seconds: How long an accepted key stays accepted.
            negative_ttl_seconds: How long a refused key stays refused.
            max_size: Maximum number of keys in each cache.
            lookup_timeout_seconds: Longest wait for one usage check, including
                the wait for a free lookup slot. Past it the request gets 503.
            max_concurrent_lookups: Most usage checks in flight at once, so a
                flood of distinct keys cannot open unbounded upstream calls.
            timer: Clock for the caches; tests replace it.
        """
        self._usage_check = usage_check
        self._accepted: TTLCache = TTLCache(
            maxsize=max_size, ttl=ttl_seconds, timer=timer
        )
        self._refused: TTLCache = TTLCache(
            maxsize=max_size, ttl=negative_ttl_seconds, timer=timer
        )
        self._pending: Dict[str, "asyncio.Task[ApplianceKeyDecision]"] = {}
        self._lookup_timeout_seconds = lookup_timeout_seconds
        self._lookup_slots = asyncio.Semaphore(max_concurrent_lookups)

    async def authorize(self, api_key: str) -> ApplianceKeyDecision:
        """Decide whether ``api_key`` may call the server.

        Args:
            api_key: The key the request carries.

        Returns:
            The decision. It never raises for an appliance API failure.
        """
        cached = self._accepted.get(api_key) or self._refused.get(api_key)
        if cached is not None:
            return cached

        pending = self._pending.get(api_key)
        if pending is None:
            pending = asyncio.ensure_future(self._look_up(api_key))
            self._pending[api_key] = pending
            pending.add_done_callback(
                lambda task: self._forget_pending(api_key=api_key, task=task)
            )

        # A client that disconnects cancels only its own wait, not the lookup
        # other requests share.
        return await asyncio.shield(pending)

    async def _look_up(self, api_key: str) -> ApplianceKeyDecision:
        try:
            result = await asyncio.wait_for(
                self._bounded_usage_check(api_key),
                timeout=self._lookup_timeout_seconds,
            )
        except RoboflowAPINotAuthorizedError:
            return self._refuse(
                api_key, status_code=401, message=UNAUTHORIZED_API_KEY_MESSAGE
            )
        except RoboflowAPIUsagePausedError:
            return self._refuse(
                api_key, status_code=403, message=PAUSED_API_KEY_MESSAGE
            )
        except RoboflowAPIForbiddenError:
            return self._refuse(
                api_key, status_code=403, message=FORBIDDEN_API_KEY_MESSAGE
            )
        except Exception as error:
            # Fail closed. The error type tells the operator which dependency
            # failed; the key is never part of the message. An incomplete 200
            # (WorkspaceLoadError) lands here too: it is not the key's fault.
            return _appliance_api_unavailable(error_type=type(error).__name__)

        if result.status_code == 401:
            return self._refuse(
                api_key, status_code=401, message=UNAUTHORIZED_API_KEY_MESSAGE
            )
        if result.status_code == 402:
            return self._refuse(
                api_key,
                status_code=403,
                message=_quota_refusal_message(
                    workspace_id=result.workspace_id, error=result.error
                ),
            )
        if (
            result.status_code != 200
            or result.under_cap is not True
            or not workspace_id_is_valid(result.workspace_id)
            or (
                result.workspace_db_id is not None
                and not workspace_db_id_is_valid(result.workspace_db_id)
            )
        ):
            # Fail closed on an incomplete 200 too: the workspace id goes into a
            # response header, so it must be present and safe there.
            return _appliance_api_unavailable(
                error_type=f"UsageCheckStatus{result.status_code}"
            )

        decision = ApplianceKeyDecision(
            status_code=200,
            workspace_id=result.workspace_id,
            workspace_db_id=result.workspace_db_id,
        )
        self._accepted[api_key] = decision

        return decision

    async def _bounded_usage_check(self, api_key: str) -> ServerlessUsageCheckResponse:
        async with self._lookup_slots:
            return await self._usage_check(api_key)

    def _forget_pending(
        self, *, api_key: str, task: "asyncio.Task[ApplianceKeyDecision]"
    ) -> None:
        if self._pending.get(api_key) is task:
            del self._pending[api_key]

    def _refuse(
        self, api_key: str, *, status_code: int, message: str
    ) -> ApplianceKeyDecision:
        decision = ApplianceKeyDecision(status_code=status_code, message=message)
        self._refused[api_key] = decision

        return decision


def _appliance_api_unavailable(*, error_type: str) -> ApplianceKeyDecision:
    logger.warning(
        "Appliance usage check failed, request refused. Error type: %s", error_type
    )

    return ApplianceKeyDecision(
        status_code=503,
        message=(
            "The api_key could not be verified because the appliance API did not "
            f"answer ({error_type}). The request was refused."
        ),
        retry_after_seconds=APPLIANCE_API_UNAVAILABLE_RETRY_AFTER_SECONDS,
    )


def _quota_refusal_message(*, workspace_id: Optional[str], error: Any) -> str:
    """Build the 403 message for a usage check that answered 402.

    Only numbers and the workspace slug go into the message, never text from
    the API, so a key or URL in the API's message cannot leak.
    """
    details = error if isinstance(error, dict) else {}
    workspace = f"Workspace {workspace_id}" if workspace_id else "The workspace"
    if details.get("reason") == WORKSPACE_RESTRICTED_REASON:
        return (
            f"{workspace} is restricted. Ask your organization administrator to "
            "restore access."
        )

    used = _credits(details.get("planCycleUsed"))
    cap = _credits(details.get("cycleCreditCap"))
    amount = f" ({used} of {cap} credits)" if used and cap else ""

    return (
        f"{workspace} has used its inference quota{amount}. Ask your organization "
        "administrator to raise the workspace quota."
    )


def _credits(value: Any) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number or number < 0:
        return None

    return f"{number:g}"


def request_is_global_management(request: Request) -> bool:
    """Tell whether a request calls a route that acts on every workspace.

    Args:
        request: The incoming request.

    Returns:
        True for the routes that remove one loaded model by id or all loaded
        models, start the server notebook, or read the server logs. In
        appliance mode no workspace key may call them.
    """
    # Raw ASGI path, as in request_skips_api_key_check.
    return request.scope["path"] in _GLOBAL_MANAGEMENT_PATHS


async def request_skips_api_key_check(request: Request) -> bool:
    """Tell whether a request may run without an API key in appliance mode.

    Args:
        request: The incoming request.

    Returns:
        True for probes, documentation, metrics, the model registry listing,
        CORS preflight (OPTIONS), and block descriptions that carry no dynamic
        blocks.
    """
    # CVE-2026-48710: use the raw ASGI path, which a Host header cannot change.
    scope_path = request.scope["path"]
    method = request.method
    if method in _EXEMPT_METHODS:
        return True
    if scope_path in _EXEMPT_PATHS:
        return True
    if scope_path.startswith("/static/") or scope_path.startswith("/_next/"):
        return True
    if method == "GET" and scope_path in _EXEMPT_GET_PATHS:
        return True
    if scope_path in _DYNAMIC_BLOCKS_PATHS:
        if method == "GET":
            return True
        json_body = await _read_json_body(request)
        return json_body is not None and not json_body.get("dynamic_blocks_definitions")

    return False


class ConflictingApiKeysError(Exception):
    """Two channels of one request carry different API keys."""


async def read_request_api_key(request: Request) -> Optional[str]:
    """Read the one API key of a request.

    A request can carry a key in the query parameter ``api_key``, in the
    ``Authorization: Bearer`` header and in the ``api_key`` field of a JSON
    body. The routes do not all read the same channels: some read the query
    parameter, and others read only the header and the body. So every channel
    that carries a key must carry the same key; otherwise the server could
    check one key and run the route with another. A present but empty query
    parameter counts as no key, as it does in the routes that read it.

    Args:
        request: The incoming request.

    Returns:
        The key, or None when the request carries none.

    Raises:
        ConflictingApiKeysError: Two channels carry different keys.
    """
    keys = []
    if "api_key" in request.query_params:
        query_api_key = request.query_params["api_key"]
        if not query_api_key:
            return None
        keys.append(query_api_key)

    header_api_key = extract_api_key_from_headers(request.headers)
    if header_api_key:
        keys.append(header_api_key)

    json_body = await _read_json_body(request)
    if json_body is not None:
        body_api_key = json_body.get("api_key")
        if isinstance(body_api_key, str) and body_api_key:
            keys.append(body_api_key)

    if len(set(keys)) > 1:
        raise ConflictingApiKeysError()

    return keys[0] if keys else None


async def _read_json_body(request: Request) -> Optional[dict]:
    content_type = request.headers.get("content-type", "")
    if content_type.split(";")[0].strip().lower() != "application/json":
        return None
    # Read the body even when Content-Length is absent: a chunked request has
    # no length but can still carry an api_key or dynamic blocks. An empty
    # body does not parse and gives None.
    try:
        body = await request.json()
    except Exception:
        # Some legacy routes send another payload under a JSON content type.
        return None
    if not isinstance(body, dict):
        return None

    return body
