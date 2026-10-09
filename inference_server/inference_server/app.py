"""Inference server — FastAPI application.

Routes are split into routers:
  - routers/v2_models.py  — /v2/models/* (load, unload, list, infer, interface)
  - routers/v2_server.py  — /v2/server/* (health, ready, info, metrics)
  - prometheus.py         — /metrics (Prometheus text format)

Per-process gateway state lives in whatever gateway_resolver.resolve_gateway()
returns.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from inference_server.legacy_env import apply_legacy_env

apply_legacy_env()

from inference_server.logging_config import (  # noqa: E402
    configure_logging,
    install_structured_access_log,
)

configure_logging()

import anyio.to_thread  # noqa: E402
from fastapi import FastAPI, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from inference_server import configuration as _cfg  # noqa: E402
from inference_server import hf_preload  # noqa: E402
from inference_server import pingback  # noqa: E402
from inference_server.auth import (  # noqa: E402
    close_session,
    extract_bearer,
    validate_api_key,
)
from inference_server.cors import PathAwareCORSMiddleware  # noqa: E402
from inference_server.errors import AuthBackendUnavailable  # noqa: E402
from inference_server.hosted.common import BillingIntentMiddleware  # noqa: E402
from inference_server.legacy import active_learning_registration  # noqa: E402
from inference_server.legacy.bridge import (  # noqa: E402
    LegacyModelBridge,
    LoopBridge,
    registry_id_for,
)
from inference_server.middlewares.correlation_id import (  # noqa: E402
    CorrelationIdMiddleware,
    correlation_id,
)
from inference_server.middlewares.headers import CORS_EXPOSE_HEADERS  # noqa: E402
from inference_server.middlewares.model_load import (  # noqa: E402
    ModelLoadHeadersMiddleware,
)
from inference_server.ops.memory_logs import setup_memory_logging  # noqa: E402
from inference_server.routers import v2_models, v2_server  # noqa: E402
from inference_server.telemetry import (  # noqa: E402
    setup_telemetry,
    shutdown_telemetry,
)
from inference_server.usage.collector import UsageCollector  # noqa: E402

setup_memory_logging()

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
_LEGACY_ERROR_HANDLING_ENABLED = (
    _cfg.LEGACY_ROUTES_ENABLED
    or _WORKFLOWS_ROUTES_ENABLED
    or _cfg.ENABLE_BUILDER
    or _cfg.ENABLE_STREAM_API
)

# ---------------------------------------------------------------------------
# Lifespan — initialize the per-process gateway
# ---------------------------------------------------------------------------


_PRELOAD_CONCURRENCY = 2


async def _preload_models(
    state,
    proxy,
    preload: list[tuple[str, str]],
    pinned: list[tuple[str, str]],
) -> None:
    loads = {mid: (api_key, False) for mid, api_key in preload}
    loads.update({mid: (api_key, True) for mid, api_key in pinned})
    semaphore = asyncio.Semaphore(_PRELOAD_CONCURRENCY)

    async def _one(mid, api_key, is_pinned):
        async with semaphore:
            try:
                result = await proxy.load(
                    registry_id_for(mid),
                    api_key=api_key or None,
                    timeout_s=300.0,
                    pinned=is_pinned,
                )
            except Exception:
                logger.error("Preload of '%s' failed", mid, exc_info=True)
                return

        if result and result[0] == "ok":
            logger.info("Preload of '%s': %s", mid, result)
            bridge = getattr(state, "legacy_bridge", None)
            if bridge is not None:
                bridge.register_preloaded(mid)
        else:
            logger.error(
                "Preload of '%s' failed: %s", mid, result[:2] if result else result
            )

    try:
        await asyncio.gather(
            *(_one(mid, key, is_pinned) for mid, (key, is_pinned) in loads.items())
        )
    finally:
        state.preload_finished = True


def _start_usage_collector() -> UsageCollector:
    usage_collector = UsageCollector()
    usage_collector.start()

    return usage_collector


STREAM_MANAGER_STOP_TIMEOUT_S = 30.0
STREAM_MANAGER_STARTUP_GRACE_S = 5.0


def _start_stream_manager(
    streams_configuration: Any,
    *,
    host_descriptor: Any,
    expected_warmed_up_pipelines: int,
) -> multiprocessing.Process:
    from streamvision.stream_manager.manager_app.bootstrap import run_stream_manager

    process = multiprocessing.get_context("spawn").Process(
        target=partial(
            run_stream_manager,
            configuration=streams_configuration,
            host_descriptor=host_descriptor,
            expected_warmed_up_pipelines=expected_warmed_up_pipelines,
        ),
        name="stream-manager",
    )
    process.start()

    return process


def _stop_stream_manager(process: Any) -> None:
    process.terminate()
    process.join(STREAM_MANAGER_STOP_TIMEOUT_S)
    if not process.is_alive():
        return None

    logger.warning(
        "The stream manager process did not stop within %s s; killing it",
        STREAM_MANAGER_STOP_TIMEOUT_S,
    )
    process.kill()
    process.join(STREAM_MANAGER_STOP_TIMEOUT_S)

    return None


def _warn_if_stream_manager_died(process: Any) -> None:
    if process.is_alive():
        return None

    logger.warning(
        "The stream manager process exited with code %s right after start; the "
        "/inference_pipelines routes will fail until the server restarts. With "
        "NUM_WORKERS > 1 only one worker's manager can bind STREAM_MANAGER_PORT.",
        process.exitcode,
    )

    return None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Keep multipart uploads in memory — Starlette default is 1MB, which causes
    # disk rollover (write + read) for typical image uploads (2-10MB).
    from starlette.formparsers import MultiPartParser

    MultiPartParser.spool_max_size = _cfg.MULTIPART_SPOOL_MB * 1024 * 1024
    from inference_server.gateway_resolver import resolve_gateway

    if not _cfg.ROBOFLOW_API_VERIFY_SSL:
        logger.warning(
            "TLS certificate verification is disabled for Roboflow platform requests"
        )
    proxy = resolve_gateway()
    from inference_models.models.vllm_proxy.vllm_client import (
        set_request_id_provider,
    )

    set_request_id_provider(correlation_id.get)
    preload_task = None
    hf_preload_task = None
    watchdog_daemons = []
    pingback_sender = None
    stream_manager_process = None
    stream_manager_watch = None
    if _cfg.HTTP_API_THREADPOOL_WORKERS is not None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = (
            _cfg.HTTP_API_THREADPOOL_WORKERS
        )
        logger.info(
            "HTTP API thread pool resized to %s threads",
            _cfg.HTTP_API_THREADPOOL_WORKERS,
        )
    workflows_executor = (
        ThreadPoolExecutor(max_workers=_cfg.WORKFLOWS_THREAD_POOL_WORKERS)
        if _cfg.WORKFLOWS_THREAD_POOL_ENABLED
        else None
    )
    app.state.workflows_executor = workflows_executor
    try:
        await proxy.start()

        from inference_models import configuration as models_cfg

        if models_cfg.OFFLINE_MODE:
            from inference_model_manager import configuration as manager_cfg
            from inference_model_manager.watchdogs import (
                CudaMemoryReclamationWatchdog,
            )

            if manager_cfg.ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG:
                cuda_daemon = CudaMemoryReclamationWatchdog(
                    interval_seconds=(
                        manager_cfg.CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS
                    )
                )
                cuda_daemon.start()
                watchdog_daemons = [cuda_daemon]
        else:
            from inference_model_manager.watchdogs import start_enabled_watchdogs

            watchdog_daemons = start_enabled_watchdogs()

        app.state.model_manager = proxy
        app.state.loop = asyncio.get_running_loop()
        app.state.loop_bridge = LoopBridge(app.state.loop)
        app.state.legacy_bridge = LegacyModelBridge(proxy)
        pingback_sender = pingback.start_sender()
        active_learning_registration.start()
        app.state.usage_collector = _start_usage_collector()
        if _workflows_host is not None:
            _workflows_host.GUARDED_IMAGE_CODEC.bind_loop(app.state.loop_bridge)
        if _cfg.ENABLE_STREAM_API:
            from inference_server.streams.configuration import (
                install_streams_configuration,
            )
            from inference_server.streams.host import SERVER_PIPELINE_HOST_DESCRIPTOR

            streams_configuration = install_streams_configuration()
            from streamvision.stream_manager.api.stream_manager_client import (
                StreamManagerClient,
            )

            stream_manager_process = _start_stream_manager(
                streams_configuration,
                host_descriptor=SERVER_PIPELINE_HOST_DESCRIPTOR,
                expected_warmed_up_pipelines=_cfg.STREAM_API_PRELOADED_PROCESSES,
            )
            stream_manager_watch = app.state.loop.call_later(
                STREAM_MANAGER_STARTUP_GRACE_S,
                _warn_if_stream_manager_died,
                stream_manager_process,
            )
            app.state.stream_manager_process = stream_manager_process
            app.state.stream_manager_client = StreamManagerClient.init(
                host=_cfg.STREAM_MANAGER_HOST,
                port=_cfg.STREAM_MANAGER_PORT,
                operations_timeout=_cfg.STREAM_MANAGER_OPERATIONS_TIMEOUT,
            )
        preload_ids = _cfg.preload_model_ids()
        pinned_ids = _cfg.pinned_model_ids()
        hf_ids = _cfg.preload_hf_ids()
        app.state.preload_finished = not (preload_ids or pinned_ids)
        preload_task = (
            asyncio.create_task(
                _preload_models(app.state, proxy, preload_ids, pinned_ids)
            )
            if preload_ids or pinned_ids
            else None
        )
        hf_preload_task = (
            asyncio.create_task(
                hf_preload.preload_hf_models(
                    proxy, hf_ids, api_key=_cfg.PRELOAD_API_KEY or None
                )
            )
            if hf_ids
            else None
        )
        yield
    finally:
        for task in (preload_task, hf_preload_task):
            if task is None:
                continue
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if pingback_sender is not None:
            pingback_sender.stop()
        active_learning_registration.stop()
        usage_collector = getattr(app.state, "usage_collector", None)
        app.state.usage_collector = None
        if usage_collector is not None:
            await asyncio.to_thread(usage_collector.flush)
            await asyncio.to_thread(usage_collector.stop)
        if stream_manager_watch is not None:
            stream_manager_watch.cancel()
        if stream_manager_process is not None:
            app.state.stream_manager_client = None
            app.state.stream_manager_process = None
            await asyncio.to_thread(_stop_stream_manager, stream_manager_process)
        shutdown_telemetry()
        try:
            await proxy.shutdown()
        finally:
            for daemon in watchdog_daemons:
                daemon.stop(timeout=5)
            if workflows_executor is not None:
                workflows_executor.shutdown(wait=False, cancel_futures=True)
            await close_session()


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------


def _ensure_offline_mode_is_supported() -> None:
    if _cfg.OFFLINE_MODE and (_cfg.LAMBDA or _cfg.GCP_SERVERLESS):
        raise RuntimeError(
            "OFFLINE_MODE is not supported together with LAMBDA / "
            "GCP_SERVERLESS deployments because authentication and usage "
            "accounting require API connectivity."
        )
    if _cfg.OFFLINE_MODE and (
        _cfg.DEDICATED_DEPLOYMENT_WORKSPACE_URL
        or _cfg.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT
    ):
        raise RuntimeError(
            "OFFLINE_MODE is not supported together with dedicated or "
            "workspace-whitelist authentication because API keys cannot be "
            "mapped to workspaces without API connectivity."
        )


_ensure_offline_mode_is_supported()

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


def _metrics_model_id(registry_id: str) -> str:
    bridge = getattr(app.state, "legacy_bridge", None)
    route = bridge._routes.get(registry_id) if bridge is not None else None
    if route is None:
        return registry_id

    return route.model_id


# Before the legacy catch-all and the root static mount, which would shadow it.
if _cfg.ENABLE_PROMETHEUS:
    from inference_server.prometheus import install_prometheus_metrics

    install_prometheus_metrics(app, resolve_model_id=_metrics_model_id)

if _LEGACY_ERROR_HANDLING_ENABLED:
    from inference_server.legacy.errors import (
        _BodyLimitMiddleware,
        install_legacy_exception_handlers,
    )

    app.add_middleware(_BodyLimitMiddleware)
    install_legacy_exception_handlers(app)

app.add_middleware(_AuthMiddleware)
app.add_middleware(BillingIntentMiddleware)

if _cfg.GCP_SERVERLESS:
    from inference_server.hosted.serverless_context import ServerlessContextMiddleware

    app.add_middleware(ServerlessContextMiddleware)

if _cfg.ALLOW_ORIGINS:
    app.add_middleware(
        PathAwareCORSMiddleware,
        match_paths=r"^(?!/build).*",
        allow_origins=_cfg.ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=CORS_EXPOSE_HEADERS,
    )

if _cfg.ENABLE_BUILDER:
    app.add_middleware(
        PathAwareCORSMiddleware,
        match_paths=r"^/(build/api|workflows/).*",
        allow_origins=[_cfg.BUILDER_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
        allow_private_network=True,
    )

if _cfg.GCP_SERVERLESS:
    from inference_server.hosted.serverless_auth import ServerlessAuthMiddleware

    app.add_middleware(ServerlessAuthMiddleware)

if (
    _cfg.DEDICATED_DEPLOYMENT_WORKSPACE_URL
    or _cfg.WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT
):
    from inference_server.hosted.dedicated_auth import DedicatedAuthMiddleware

    app.add_middleware(DedicatedAuthMiddleware)

app.add_middleware(ModelLoadHeadersMiddleware)
app.add_middleware(CorrelationIdMiddleware)
setup_telemetry(app)
install_structured_access_log(app)


def mount_landing_assets(app: FastAPI) -> bool:
    if not os.path.isdir(_cfg.LANDING_DIR):
        logger.warning(
            "Landing page directory %s not found; / is not served", _cfg.LANDING_DIR
        )
        return False

    if not _cfg.ENABLE_DASHBOARD:

        @app.get("/dashboard.html")
        @app.head("/dashboard.html")
        async def _dashboard_guard() -> Response:
            return Response(status_code=404)

    static_dir = os.path.join(_cfg.LANDING_DIR, "static")
    if os.path.isdir(static_dir):
        app.mount(
            "/static", StaticFiles(directory=static_dir, html=True), name="static"
        )

    next_static_dir = os.path.join(_cfg.LANDING_DIR, "_next", "static")
    if os.path.isdir(next_static_dir):
        app.mount(
            "/_next/static",
            StaticFiles(directory=next_static_dir, html=True),
            name="_next_static",
        )

    return True


def mount_landing_root(app: FastAPI) -> bool:
    if not os.path.isdir(_cfg.LANDING_DIR):
        return False

    app.mount("/", StaticFiles(directory=_cfg.LANDING_DIR, html=True), name="root")
    return True


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

_LANDING_ASSETS_MOUNTED = mount_landing_assets(app)

app.include_router(v2_models.router)
app.include_router(v2_server.router)

if _cfg.LEGACY_ROUTES_ENABLED:
    from inference_server.legacy.router import (
        include_legacy_catch_all,
        include_legacy_routers,
    )
    from inference_server.ops.router import include_ops_routers

    include_legacy_routers(app)
    include_ops_routers(app)

if _WORKFLOWS_ROUTES_ENABLED:
    from inference_server.workflows import router as workflows_router

    app.include_router(workflows_router.router)

if _cfg.ENABLE_BUILDER:
    from inference_server.builder.routes import router as builder_router

    app.include_router(builder_router, prefix="/build", tags=["builder"])

if _cfg.ENABLE_STREAM_API:
    from inference_server.streams.configuration import install_streams_configuration

    install_streams_configuration()
    from inference_server.streams.router import include_streams_router

    include_streams_router(app)

if _cfg.LEGACY_ROUTES_ENABLED:
    include_legacy_catch_all(app)

_LANDING_ROOT_MOUNTED = mount_landing_root(app)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Serve the application with uvicorn on ``HOST`` and ``PORT`` with ``NUM_WORKERS``.

    Passes ``log_config=None`` so uvicorn keeps the handlers installed by
    ``configure_logging()`` instead of applying its default logging config.
    """
    import uvicorn

    port = int(os.environ.get(_cfg.PORT_ENV, str(_cfg.APP_PORT_DEFAULT)))
    workers = _cfg.NUM_WORKERS
    uvicorn.run(
        "inference_server.app:app",
        host=_cfg.HOST,
        port=port,
        workers=workers,
        log_config=None,
    )


if __name__ == "__main__":
    main()
