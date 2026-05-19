"""Inference server — FastAPI application.

Routes are split into routers:
  - routers/infer.py      — POST /infer (fast binary endpoint)
  - routers/v2_models.py  — /v2/models/* (load, unload, list, infer, interface)
  - routers/v2_server.py  — /v2/server/* (health, ready, info, metrics)

Per-process state (ZMQ, SHM, protocol helpers) lives in state.py.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from multiprocessing.shared_memory import SharedMemory

import zmq
import zmq.asyncio
from fastapi import FastAPI, Response

from inference_server import configuration as _cfg
from inference_server import state
from inference_server.auth import validate_api_key
from inference_server.proxies.base import ModelManagerProxy
from inference_server.proxies.mm_wrapper import MMWrapper
from inference_server.proxies.mmp_client import MMPClient
from inference_server.routers import infer, v2_models, v2_server


# ---------------------------------------------------------------------------
# Lifespan — initialize per-process ZMQ + SHM
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Keep multipart uploads in memory — Starlette default is 1MB, which causes
    # disk rollover (write + read) for typical image uploads (2-10MB).
    from starlette.formparsers import MultiPartParser

    MultiPartParser.spool_max_size = _cfg.MULTIPART_SPOOL_MB * 1024 * 1024

    mode = _cfg.INFERENCE_DEPLOYMENT_MODE
    proxy: ModelManagerProxy

    if mode == _cfg.MODE_BUNDLED:
        from inference_model_manager.model_manager import ModelManager

        proxy = MMWrapper(ModelManager())
        await proxy.start()
    elif mode == _cfg.MODE_MMP:
        # Transitional: state.py setup is still required because routers
        # call state.X helpers. MMPClient runs alongside; routers will
        # migrate to it in a follow-up step, then state.py goes away.
        state.init_from_env()

        identity = f"uv_{os.getpid()}_{uuid.uuid4().hex[:8]}".encode()
        state.ctx = zmq.asyncio.Context()
        state.sock = state.ctx.socket(zmq.DEALER)
        state.sock.setsockopt(zmq.IDENTITY, identity)
        state.sock.setsockopt(zmq.SNDHWM, 0)
        state.sock.setsockopt(zmq.RCVHWM, 0)
        state.sock.setsockopt(zmq.LINGER, 0)
        state.sock.connect(state.MMP_ADDR)
        state.shm = SharedMemory(name=state.SHM_NAME, create=False)
        state.pending = {}
        state.recv_task = asyncio.create_task(state.recv_loop(), name="zmq-recv")
        state.start_pipeline_csv_writer()

        proxy = MMPClient()
        await proxy.start()
    else:
        raise RuntimeError(
            f"unknown {_cfg.INFERENCE_DEPLOYMENT_MODE_ENV}={mode!r} "
            f"(expected {_cfg.MODE_BUNDLED!r} or {_cfg.MODE_MMP!r})"
        )

    app.state.model_manager = proxy

    yield

    await proxy.shutdown()

    if mode == _cfg.MODE_MMP:
        state.recv_task.cancel()
        try:
            await state.recv_task
        except asyncio.CancelledError:
            pass
        state.sock.close()
        state.ctx.term()
        state.shm.close()


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

_DEBUG_BENCHMARK_MODE = _cfg.DEBUG_BENCHMARK_MODE


class _AuthMiddleware:
    """ASGI middleware for auth — does NOT buffer the request body.

    Starlette's @app.middleware("http") with call_next consumes the body
    stream before passing to the route, breaking request.stream() in endpoints.
    This raw ASGI middleware avoids that by passing receive through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if _DEBUG_BENCHMARK_MODE:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "").rstrip("/")
        if path in _AUTH_SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        # Extract Bearer token from headers
        headers = dict(
            (k.decode("latin-1").lower(), v.decode("latin-1"))
            for k, v in scope.get("headers", [])
        )
        auth = headers.get("authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""

        if not token:
            response = Response(
                status_code=401,
                content=b"Authorization: Bearer <api_key> header required",
            )
            await response(scope, receive, send)
            return

        valid, _ = await validate_api_key(token)
        if not valid:
            response = Response(status_code=403, content=b"Invalid API key")
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


app.add_middleware(_AuthMiddleware)


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(infer.router)
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
