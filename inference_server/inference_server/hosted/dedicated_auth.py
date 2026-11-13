"""Workspace allow-list middleware of dedicated, whitelisted and key-checked deployments."""

import logging
from typing import Optional, Set

from inference_server import configuration
from inference_server.auth import validate_api_key
from inference_server.errors import AuthBackendUnavailable
from inference_server.hosted.common import (
    STATIC_PREFIXES,
    UNAUTHORIZED_MESSAGE,
    HostedRequest,
    RequestBodyTooLarge,
    _is_v2_request,
    error_response,
    payload_too_large_response,
    resolve_api_key,
    send_with_workspace_header,
)

logger = logging.getLogger(__name__)

NO_WORKSPACE_MESSAGE = (
    "This server checks API keys but has no workspace to check them against. "
    "Set ROBOFLOW_API_KEY or WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT on the server."
)
SERVER_KEY_REJECTED_MESSAGE = (
    "The server's own ROBOFLOW_API_KEY was rejected by the Roboflow API, "
    "so the server cannot tell which workspace it belongs to."
)
AUTH_BACKEND_UNAVAILABLE_MESSAGE = "auth backend unavailable, try again"

SKIP_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/redoc",
        "/info",
        "/healthz",
        "/readiness",
        "/secure-gateway/health",
        "/metrics",
        "/openapi.json",
    }
)


# Workspace of the server's own ROBOFLOW_API_KEY once resolved; a failed lookup
# is never stored, so the next request tries again.
_server_workspace: Optional[str] = None


def configured_workspaces() -> Set[str]:
    """Return the workspaces this deployment serves by configuration.

    Returns:
        Union of ``DEDICATED_DEPLOYMENT_WORKSPACE_URL`` and
        ``WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT``.
    """
    workspaces = set()
    if configuration.DEDICATED_DEPLOYMENT_WORKSPACE_URL:
        workspaces.add(configuration.DEDICATED_DEPLOYMENT_WORKSPACE_URL)
    if configuration.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT:
        workspaces.update(configuration.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT)

    return workspaces


def uses_server_workspace() -> bool:
    """Return whether the server serves the workspace of its own API key.

    True under ``REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT`` when no workspace is
    configured and the server has a ``ROBOFLOW_API_KEY``.
    """
    return bool(
        configuration.REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT
        and not configured_workspaces()
        and configuration.DEFAULT_API_KEY
    )


async def allowed_workspaces() -> Set[str]:
    """Return the workspaces this deployment serves.

    Returns:
        :func:`configured_workspaces`, or the workspace of the server's own API
        key when :func:`uses_server_workspace`; empty when that key is rejected.

    Raises:
        AuthBackendUnavailable: The server's key could not be looked up.
    """
    global _server_workspace
    if not uses_server_workspace():
        return configured_workspaces()
    if _server_workspace is None:
        valid, workspace_id = await validate_api_key(
            configuration.DEFAULT_API_KEY, through_secure_gateway=True
        )
        if not valid:
            logger.error(
                "SECURITY: the Roboflow API rejected the server's ROBOFLOW_API_KEY; "
                "every authenticated route answers 401 until it is fixed."
            )
            return set()
        _server_workspace = workspace_id

    return {_server_workspace}


def _skip_check(request: HostedRequest) -> bool:
    # Only CORS preflight requests skip the check, whatever the method of the
    # route behind them.
    if request.method == "OPTIONS":
        return True
    if _is_v2_request(request):
        return False

    return request.path in SKIP_PATHS or request.path.startswith(STATIC_PREFIXES)


def _backend_unavailable_response():
    response = error_response(503, AUTH_BACKEND_UNAVAILABLE_MESSAGE)
    response.headers["Retry-After"] = "5"

    return response


class DedicatedAuthMiddleware:
    """Raw ASGI middleware admitting only keys of allow-listed workspaces.

    Every request except CORS preflight outside the skip list must carry an
    API key (query, Bearer header or JSON body) whose workspace is in
    :func:`allowed_workspaces`; anything else is answered with 401, and with
    503 while the Roboflow API cannot be reached to check it.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = HostedRequest(scope, receive)
        if _skip_check(request):
            await self.app(scope, receive, send)
            return
        if not configured_workspaces() and not uses_server_workspace():
            response = error_response(401, NO_WORKSPACE_MESSAGE)
            await response(scope, request.receive, send)
            return

        try:
            if request.declares_oversized_body:
                raise RequestBodyTooLarge()
            api_key = await resolve_api_key(request)
        except RequestBodyTooLarge:
            response = payload_too_large_response(request)
            await response(scope, request.receive, send)
            return
        if api_key is None:
            response = error_response(401, UNAUTHORIZED_MESSAGE)
            await response(scope, request.receive, send)
            return

        try:
            valid, workspace_id = await validate_api_key(
                api_key, through_secure_gateway=not _is_v2_request(request)
            )
            workspaces = await allowed_workspaces() if valid else set()
        except AuthBackendUnavailable:
            response = _backend_unavailable_response()
            await response(scope, request.receive, send)
            return
        if valid and not workspaces:
            response = error_response(401, SERVER_KEY_REJECTED_MESSAGE)
            await response(scope, request.receive, send)
            return
        if not valid or workspace_id not in workspaces:
            response = error_response(401, UNAUTHORIZED_MESSAGE)
            await response(scope, request.receive, send)
            return

        await self.app(
            scope, request.receive, send_with_workspace_header(send, workspace_id)
        )
