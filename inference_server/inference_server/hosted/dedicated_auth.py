"""Workspace allow-list middleware of dedicated and whitelisted deployments."""

from typing import Set

from inference_server import configuration
from inference_server.auth import validate_api_key
from inference_server.hosted.common import (
    AUTHENTICATED_METHODS,
    STATIC_PREFIXES,
    UNAUTHORIZED_MESSAGE,
    HostedRequest,
    _is_v2_request,
    error_response,
    resolve_api_key,
    send_with_workspace_header,
)

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


def allowed_workspaces() -> Set[str]:
    """Return the workspaces this deployment serves.

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


def _skip_check(request: HostedRequest) -> bool:
    if request.method not in AUTHENTICATED_METHODS:
        return True

    return request.path in SKIP_PATHS or request.path.startswith(STATIC_PREFIXES)


class DedicatedAuthMiddleware:
    """Raw ASGI middleware admitting only keys of allow-listed workspaces.

    Every GET/POST outside the skip list must carry an API key (query,
    Bearer header or JSON body) whose workspace is in
    :func:`allowed_workspaces`; anything else is answered with 401.
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

        api_key = await resolve_api_key(request)
        if api_key is None:
            response = error_response(401, UNAUTHORIZED_MESSAGE)
            await response(scope, request.receive, send)
            return

        valid, workspace_id = await validate_api_key(
            api_key, through_secure_gateway=not _is_v2_request(request)
        )
        if not valid or workspace_id not in allowed_workspaces():
            response = error_response(401, UNAUTHORIZED_MESSAGE)
            await response(scope, request.receive, send)
            return

        await self.app(
            scope, request.receive, send_with_workspace_header(send, workspace_id)
        )
