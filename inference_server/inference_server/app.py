"""Inference server — FastAPI application.

Routes are split into routers:
  - routers/v2_models.py  — /v2/models/* (load, unload, list, infer, interface)
  - routers/v2_server.py  — /v2/server/* (health, ready, info, metrics)

Per-process gateway state lives in whatever gateway_resolver.resolve_gateway()
returns.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from inference_server import configuration as _cfg
from inference_server.auth import extract_bearer, validate_api_key
from inference_server.errors import AuthBackendUnavailable
from inference_server.routers import v2_models, v2_server

# ---------------------------------------------------------------------------
# Lifespan — initialize the per-process gateway
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Keep multipart uploads in memory — Starlette default is 1MB, which causes
    # disk rollover (write + read) for typical image uploads (2-10MB).
    from starlette.formparsers import MultiPartParser

    MultiPartParser.spool_max_size = _cfg.MULTIPART_SPOOL_MB * 1024 * 1024
    from inference_server.gateway_resolver import resolve_gateway

    proxy = resolve_gateway()
    try:
        await proxy.start()
        app.state.model_manager = proxy
        yield
    finally:
        await proxy.shutdown()


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=_lifespan)

_AUTH_SKIP_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/v2/server/health",
        "/v2/server/ready",
    }
)

_CONTROL_PLANE_ROUTES = frozenset(
    {
        ("GET", "/v2/models"),
        ("DELETE", "/v2/models"),
        ("POST", "/v2/models/load"),
        ("POST", "/v2/models/unload"),
        ("GET", "/v2/server/info"),
        ("GET", "/v2/server/metrics"),
    }
)

class _AuthMiddleware:
    """ASGI middleware for auth — does NOT buffer the request body.

    Starlette's @app.middleware("http") with call_next consumes the body
    stream before passing to the route, breaking request.stream() in endpoints.
    This raw ASGI middleware avoids that by passing receive through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # No websocket routes exist; a future one must not ship
            # unauthenticated by default.
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # rstrip alone turns "/" into "" which is not in the skip set —
        # the root path was always 401 despite being skip-listed.
        path = scope.get("path", "").rstrip("/") or "/"
        if path in _AUTH_SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        if (
            not _cfg.ENABLE_CONTROL_PLANE_ROUTES
            and (scope.get("method", ""), path) in _CONTROL_PLANE_ROUTES
        ):
            response = Response(
                status_code=403,
                content=b"control-plane routes disabled; "
                b"set ENABLE_CONTROL_PLANE_ROUTES=true to enable",
            )
            await response(scope, receive, send)
            return

        # Extract Bearer token from headers
        headers = dict(
            (k.decode("latin-1").lower(), v.decode("latin-1"))
            for k, v in scope.get("headers", [])
        )
        token = extract_bearer(headers.get("authorization", ""))

        if not token:
            response = Response(
                status_code=401,
                content=b"Authorization: Bearer <api_key> header required",
            )
            await response(scope, receive, send)
            return

        try:
            valid, _ = await validate_api_key(token)
        except AuthBackendUnavailable:
            response = Response(
                status_code=503,
                headers={"Retry-After": "5"},
                content=b"auth backend unavailable, try again",
            )
            await response(scope, receive, send)
            return
        if not valid:
            response = Response(status_code=403, content=b"Invalid API key")
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


app.add_middleware(_AuthMiddleware)


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(v2_models.router)
app.include_router(v2_server.router)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get(_cfg.PORT_ENV, str(_cfg.APP_PORT_DEFAULT)))
    workers = _cfg.NUM_WORKERS
    uvicorn.run(
        "inference_server.app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
    )
