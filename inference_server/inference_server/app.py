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

from inference_server import state
from inference_server.auth import validate_api_key
from inference_server.routers import infer, v2_models, v2_server

# ---------------------------------------------------------------------------
# Lifespan — initialize per-process ZMQ + SHM
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Re-read env vars set by server.py (before uvicorn forked this worker)
    state.init_from_env()

    # Keep multipart uploads in memory — Starlette default is 1MB, which causes
    # disk rollover (write + read) for typical image uploads (2-10MB).
    from starlette.formparsers import MultiPartParser

    MultiPartParser.spool_max_size = (
        int(os.environ.get("INFERENCE_MULTIPART_SPOOL_MB", "32")) * 1024 * 1024
    )

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

    yield

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

_DEBUG_BENCHMARK_MODE = os.environ.get("DEBUG_BENCHMARK_MODE", "").strip() == "1"


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

    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("NUM_WORKERS", "4"))
    uvicorn.run(
        "inference_server.app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
    )
