"""Inference server — FastAPI application.

Routes are split into routers:
  - routers/v2_models.py  — /v2/models/* (load, unload, list, infer, interface)
  - routers/v2_server.py  — /v2/server/* (health, ready, info, metrics)

Per-process gateway state lives in whatever gateway_resolver.resolve_gateway()
returns.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from inference_server import configuration as _cfg
from inference_server.auth import extract_bearer, validate_api_key
from inference_server.errors import AuthBackendUnavailable
from inference_server.legacy.bridge import LegacyModelBridge, LoopBridge
from inference_server.routers import v2_models, v2_server

logger = logging.getLogger(__name__)

_WORKFLOWS_INSTALLED = importlib.util.find_spec("roboflow_workflows") is not None
if _WORKFLOWS_INSTALLED:
    from inference_server.workflows import host as _workflows_host
else:
    _workflows_host = None
    logger.info(
        "Workflows routes disabled: roboflow-workflows is not installed "
        "(pip install 'inference-server[workflows]')"
    )

_WORKFLOWS_ROUTES_ENABLED = (
    _workflows_host is not None and not _cfg.DISABLE_WORKFLOW_ENDPOINTS
)
_LEGACY_ERROR_HANDLING_ENABLED = _cfg.LEGACY_ROUTES_ENABLED or _WORKFLOWS_ROUTES_ENABLED

# ---------------------------------------------------------------------------
# Lifespan — initialize the per-process gateway
# ---------------------------------------------------------------------------


async def _preload_models(
    proxy, model_ids: list[str], api_key: Optional[str]
) -> None:
    async def _one(mid):
        try:
            result = await proxy.load(mid, api_key=api_key, timeout_s=300.0)
            logger.info("Preload of '%s': %s", mid, result)
        except Exception:
            logger.warning("Preload of '%s' failed", mid, exc_info=True)

    await asyncio.gather(*(_one(m) for m in model_ids))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Keep multipart uploads in memory — Starlette default is 1MB, which causes
    # disk rollover (write + read) for typical image uploads (2-10MB).
    from starlette.formparsers import MultiPartParser

    MultiPartParser.spool_max_size = _cfg.MULTIPART_SPOOL_MB * 1024 * 1024
    from inference_server.gateway_resolver import resolve_gateway

    proxy = resolve_gateway()
    preload_task = None
    watchdog_daemons = []
    workflows_executor = ThreadPoolExecutor(
        max_workers=_cfg.WORKFLOWS_THREAD_POOL_WORKERS
    )
    app.state.workflows_executor = workflows_executor
    try:
        await proxy.start()

        from inference_model_manager.watchdogs import start_enabled_watchdogs

        watchdog_daemons = start_enabled_watchdogs()

        app.state.model_manager = proxy
        app.state.loop = asyncio.get_running_loop()
        app.state.loop_bridge = LoopBridge(app.state.loop)
        app.state.legacy_bridge = LegacyModelBridge(proxy)
        if _workflows_host is not None:
            _workflows_host.GUARDED_IMAGE_CODEC.bind_loop(app.state.loop_bridge)
        preload_ids = _cfg.preload_model_ids()
        preload_task = (
            asyncio.create_task(
                _preload_models(proxy, preload_ids, _cfg.PRELOAD_API_KEY or None)
            )
            if preload_ids
            else None
        )
        yield
    finally:
        if preload_task is not None:
            if not preload_task.done():
                preload_task.cancel()
            await asyncio.gather(preload_task, return_exceptions=True)
        try:
            await proxy.shutdown()
        finally:
            for daemon in watchdog_daemons:
                daemon.stop(timeout=5)
            workflows_executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Roboflow Inference Server",
    description="Roboflow inference server",
    version=_cfg.SERVER_VERSION,
    terms_of_service="https://roboflow.com/terms",
    contact={
        "name": "Roboflow Inc.",
        "url": "https://roboflow.com/contact",
        "email": "help@roboflow.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=_lifespan,
)

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

_V2_PREFIX = "/v2/"


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

        if not path.startswith(_V2_PREFIX):
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


if _LEGACY_ERROR_HANDLING_ENABLED:
    from inference_server.legacy.errors import (
        _BodyLimitMiddleware,
        install_legacy_exception_handlers,
    )

    app.add_middleware(_BodyLimitMiddleware)
    install_legacy_exception_handlers(app)

app.add_middleware(_AuthMiddleware)

if _cfg.ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cfg.ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(v2_models.router)
app.include_router(v2_server.router)

if _cfg.LEGACY_ROUTES_ENABLED:
    from inference_server.legacy.router import (
        include_legacy_catch_all,
        include_legacy_routers,
    )

    include_legacy_routers(app)

if _WORKFLOWS_ROUTES_ENABLED:
    from inference_server.workflows import router as workflows_router

    app.include_router(workflows_router.router)

if _cfg.LEGACY_ROUTES_ENABLED:
    include_legacy_catch_all(app)


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
