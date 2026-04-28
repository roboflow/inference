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
from fastapi import FastAPI, Request, Response

from inference_server import state
from inference_server.auth import validate_api_key
from inference_server.routers import infer, v2_models, v2_server

# ---------------------------------------------------------------------------
# Lifespan — initialize per-process ZMQ + SHM
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_: FastAPI):
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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if _DEBUG_BENCHMARK_MODE:
        return await call_next(request)
    if request.url.path.rstrip("/") in _AUTH_SKIP_PATHS:
        return await call_next(request)
    token = _bearer_token(request)
    if not token:
        return Response(
            status_code=401, content=b"Authorization: Bearer <api_key> header required"
        )
    valid, _ = await validate_api_key(token)
    if not valid:
        return Response(status_code=403, content=b"Invalid API key")
    return await call_next(request)


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


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
