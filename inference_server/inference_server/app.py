"""FastAPI benchmark app — orchestrated ModelManagerProcess mode.

Accepts raw image bytes in the request body.
Infer parameters (including model_id) are passed as query params.
model_id may contain '/' (URL-encode as %2F; Starlette decodes automatically).

Usage::

    POST /infer?model_id=yolov8n%2F640&confidence=0.5
    Authorization: Bearer <api_key>
    Content-Type: image/jpeg
    <raw jpeg bytes>

    200 OK  → result bytes (application/octet-stream)
    503     → model loading (Retry-After header set) or no SHM slots
    504     → inference timeout
    413     → payload larger than SHM slot

Per-request flow:
  1. T_ENSURE_LOADED → MMP  (no-op on warm path; MMP replies instantly)
  2. T_ALLOC        → MMP  → slot_id
  3. body chunks streamed directly into SHM input area (zero intermediate copy)
  4. T_SUBMIT       → MMP  → backend worker picks up slot
  5. await T_RESULT_READY from MMP
  6. read result from SHM result area → T_FREE slot

Environment variables::

    INFERENCE_MMP_ADDR          ZMQ address of MMP (default: platform auto via transport.py)
    INFERENCE_ZMQ_TRANSPORT     "ipc" or "tcp" (default: platform default)
    INFERENCE_SHM_NAME          Shared memory block name (default: inference_pool)
    INFERENCE_SHM_DATA_SIZE     Bytes per slot data area   (default: 25 MB)
    INFERENCE_LOAD_WAIT_S       Max seconds to wait for model load (default: 10)
    INFERENCE_INFER_TIMEOUT_S   Max seconds to wait for inference result (default: 30)
    INFERENCE_ALLOC_TIMEOUT_S   Max seconds to wait for slot alloc (default: 2)
    PORT                        HTTP port (default: 8000)
    NUM_WORKERS                 uvicorn worker processes (default: 4)
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import struct
import time
import uuid
from contextlib import asynccontextmanager
from multiprocessing.shared_memory import SharedMemory
from typing import Annotated, Optional

import filetype
import uvicorn
import zmq
import zmq.asyncio
from fastapi import Depends, FastAPI, Request, Response

from inference_model_manager.backends.utils.transport import zmq_addr
from inference_server.auth import validate_api_key
from inference_server.serializers import serialize_json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MMP_ADDR = os.environ.get("INFERENCE_MMP_ADDR") or zmq_addr("mmprocess")
_SHM_NAME = os.environ.get("INFERENCE_SHM_NAME", "inference_pool")
_SHM_DATA_SIZE = int(os.environ.get("INFERENCE_SHM_DATA_SIZE", str(25 * 1024 * 1024)))
_LOAD_WAIT_S = float(os.environ.get("INFERENCE_LOAD_WAIT_S", "10.0"))
_INFER_TIMEOUT_S = float(os.environ.get("INFERENCE_INFER_TIMEOUT_S", "30.0"))
_ALLOC_TIMEOUT_S = float(os.environ.get("INFERENCE_ALLOC_TIMEOUT_S", "2.0"))

# SHM slot layout (must match shm_pool.py):
#   [HEADER 64B | DATA _SHM_DATA_SIZE]
_HEADER_SIZE = 64
_SLOT_TOTAL = _HEADER_SIZE + _SHM_DATA_SIZE

# ---------------------------------------------------------------------------
# Image format detection (magic bytes via ``filetype`` lib + numpy .npy)
# ---------------------------------------------------------------------------

_NPY_MAGIC = b"\x93NUMPY"


def _looks_like_image(data: bytes | memoryview) -> bool:
    head = bytes(data[:262])  # filetype needs up to 262 bytes
    if head[:6] == _NPY_MAGIC:
        return True
    return filetype.is_image(head)


# ---------------------------------------------------------------------------
# ZMQ message types (must match model_manager_process.py)
# ---------------------------------------------------------------------------

# uvicorn → MMP
_T_ALLOC = b"\x01"
_T_SUBMIT = b"\x02"
_T_FREE = b"\x03"
_T_ENSURE_LOADED = b"\x09"

# MMP → uvicorn
_T_ALLOC_OK = b"\x11"
_T_RESULT_READY = b"\x14"
_T_ERROR = b"\xFF"
_T_MODEL_READY = b"\x0A"
_T_LOAD_TIMEOUT = b"\x0B"
_T_OK = b"\x40"

# uvicorn → MMP (lifecycle)
_T_LOAD = b"\x20"
_T_UNLOAD = b"\x21"

# Wire formats (big-endian):
#   _T_ALLOC:         Q H N    req_id(8) flavor_len(2) flavor(N)
#   _T_ALLOC_OK:      Q I      req_id(8) slot_id(4)
#   _T_SUBMIT:        Q I I H N  req_id(8) slot_id(4) input_sz(4) flavor_len(2) flavor(N)
#                     + frame2: json params bytes
#   _T_RESULT_READY:  Q I I    req_id(8) slot_id(4) result_sz(4)
#   _T_FREE:          I        slot_id(4)
#   _T_ERROR:         Q B      req_id(8) error_code(1)
#   _T_ENSURE_LOADED: Q I H N H M H D  req_id(8) wait_ms(4) model_id_len(2) model_id(N) key_len(2) key(M) device_len(2) device(D)
#   _T_MODEL_READY:   Q        req_id(8)
#   _T_LOAD_TIMEOUT:  Q I      req_id(8) retry_after_s(4)

# ---------------------------------------------------------------------------
# Per-process state
# ---------------------------------------------------------------------------
#
# uvicorn --workers N spawns N OS processes. Each process needs its own:
#   - ZMQ DEALER socket with a unique identity so MMP can route replies back
#     to exactly this process (ROUTER identity routing)
#   - _pending dict: asyncio.Future objects are per-event-loop, not sharable
#   - SHM attachment: each process attaches the shared block independently
#
# These are set once per process in the FastAPI lifespan handler.

_shm: Optional[SharedMemory] = None
_sock: Optional[zmq.asyncio.Socket] = None
_ctx: Optional[zmq.asyncio.Context] = None
_pending: dict[int, asyncio.Future] = {}
_recv_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _shm, _sock, _ctx, _pending, _recv_task

    identity = f"uv_{os.getpid()}_{uuid.uuid4().hex[:8]}".encode()

    _ctx = zmq.asyncio.Context()
    _sock = _ctx.socket(zmq.DEALER)
    _sock.setsockopt(zmq.IDENTITY, identity)
    _sock.setsockopt(zmq.SNDHWM, 0)  # unlimited send buffer
    _sock.setsockopt(zmq.RCVHWM, 0)  # unlimited recv buffer
    _sock.setsockopt(zmq.LINGER, 0)  # don't block on close
    _sock.connect(_MMP_ADDR)

    _shm = SharedMemory(name=_SHM_NAME, create=False)
    _pending = {}
    _recv_task = asyncio.create_task(_recv_loop(), name="zmq-recv")

    yield

    _recv_task.cancel()
    try:
        await _recv_task
    except asyncio.CancelledError:
        pass
    _sock.close()
    _ctx.term()
    _shm.close()


# ---------------------------------------------------------------------------
# ZMQ recv loop — asyncio task, resolves futures on the same event loop
# ---------------------------------------------------------------------------


async def _recv_loop() -> None:
    while True:
        try:
            parts = await _sock.recv_multipart()
        except (asyncio.CancelledError, zmq.ZMQError):
            break
        if not parts:
            continue
        _dispatch(parts[0], parts[1:])


def _dispatch(msg_type: bytes, frames: list[bytes]) -> None:
    try:
        if msg_type == _T_ALLOC_OK:
            req_id, slot_id = struct.unpack(">QI", frames[0])
            _resolve(req_id, ("alloc_ok", slot_id))

        elif msg_type == _T_RESULT_READY:
            req_id, slot_id, result_sz = struct.unpack(">QII", frames[0])
            _resolve(req_id, ("result", slot_id, result_sz))

        elif msg_type == _T_MODEL_READY:
            req_id = struct.unpack(">Q", frames[0])[0]
            _resolve(req_id, ("model_ready",))

        elif msg_type == _T_LOAD_TIMEOUT:
            req_id, retry_after = struct.unpack(">QI", frames[0])
            _resolve(req_id, ("load_timeout", retry_after))

        elif msg_type == _T_OK:
            req_id = struct.unpack(">Q", frames[0])[0]
            _resolve(req_id, ("ok",))

        elif msg_type == _T_ERROR:
            req_id = struct.unpack(">Q", frames[0][:8])[0]
            err = frames[0][8] if len(frames[0]) > 8 else 0
            _resolve(req_id, ("error", err))

    except (struct.error, IndexError):
        pass  # malformed — drop


def _resolve(req_id: int, value: tuple) -> None:
    fut = _pending.pop(req_id, None)
    if fut is not None and not fut.done():
        fut.set_result(value)


def _new_req_id() -> int:
    return uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF


def _make_future(req_id: int) -> asyncio.Future:
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    _pending[req_id] = fut
    return fut


def _drop_future(req_id: int) -> None:
    """Remove timed-out future so a late MMP reply doesn't set_result on it."""
    _pending.pop(req_id, None)


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


def _routing_key(model_id: str, instance: str) -> str:
    """Compose MMP routing key from model_id and optional instance suffix."""
    return f"{model_id}:{instance}" if instance else model_id


async def _ensure_loaded(
    model_id: str,
    instance: str = "",
    api_key: str = "",
    device: str = "",
) -> tuple:
    """Ask MMP to ensure model is loaded.

    model_id + instance together identify the backend (routing key = model_id:instance).
    device is a hint for cold-path loading (which GPU to use).

    Warm path (model already loaded): MMP replies T_MODEL_READY instantly.
    Cold path: MMP loads the model; replies when ready or wait_ms exceeded.

    Returns one of:
        ("model_ready",)
        ("load_timeout", retry_after_s)
        ("error", code)
    """
    req_id = _new_req_id()
    mid_bytes = _routing_key(model_id, instance).encode()
    key_bytes = api_key.encode()
    dev_bytes = device.encode()
    wait_ms = int(_LOAD_WAIT_S * 1000)
    # wire: Q I H N H M H D
    #   req_id(8) wait_ms(4) model_id_len(2) model_id(N)
    #   key_len(2) key(M) device_len(2) device(D)
    payload = (
        struct.pack(">QIH", req_id, wait_ms, len(mid_bytes))
        + mid_bytes
        + struct.pack(">H", len(key_bytes))
        + key_bytes
        + struct.pack(">H", len(dev_bytes))
        + dev_bytes
    )
    fut = _make_future(req_id)
    await _sock.send_multipart([_T_ENSURE_LOADED, payload])
    try:
        # +1s headroom beyond MMP's own budget
        return await asyncio.wait_for(fut, timeout=_LOAD_WAIT_S + 1.0)
    except asyncio.TimeoutError:
        _drop_future(req_id)
        return ("load_timeout", int(_LOAD_WAIT_S))


async def _alloc_slot(model_id: str, instance: str = "") -> int:
    """Claim a SHM slot from MMP. Returns slot_id.

    Raises asyncio.TimeoutError if no slot granted within ALLOC_TIMEOUT_S.
    Raises RuntimeError if MMP replies with an error.
    """
    req_id = _new_req_id()
    mid = _routing_key(model_id, instance).encode()
    payload = struct.pack(">QH", req_id, len(mid)) + mid
    fut = _make_future(req_id)
    await _sock.send_multipart([_T_ALLOC, payload])
    try:
        result = await asyncio.wait_for(fut, timeout=_ALLOC_TIMEOUT_S)
    except asyncio.TimeoutError:
        _drop_future(req_id)
        raise
    if result[0] == "error":
        raise RuntimeError(f"alloc error code={result[1]}")
    return result[1]  # slot_id


async def _submit_and_wait(
    slot_id: int,
    model_id: str,
    instance: str,
    input_sz: int,
    params: dict,
) -> tuple:
    """Signal MMP that slot data is ready; await the inference result.

    Returns one of:
        ("result", slot_id, result_sz)
        ("error",  code)

    Raises asyncio.TimeoutError if no result within INFER_TIMEOUT_S.
    """
    req_id = _new_req_id()
    mid = _routing_key(model_id, instance).encode()
    header = struct.pack(">QIIH", req_id, slot_id, input_sz, len(mid)) + mid
    params_json = json.dumps(params).encode() if params else b"{}"
    fut = _make_future(req_id)
    await _sock.send_multipart([_T_SUBMIT, header, params_json])
    try:
        return await asyncio.wait_for(fut, timeout=_INFER_TIMEOUT_S)
    except asyncio.TimeoutError:
        _drop_future(req_id)
        raise


def _free_slot(slot_id: int) -> None:
    async def _send():
        await _sock.send_multipart([_T_FREE, struct.pack(">I", slot_id)])

    asyncio.create_task(_send())


def _write_input(slot_id: int, chunk: bytes | memoryview, offset: int) -> None:
    base = slot_id * _SLOT_TOTAL + _HEADER_SIZE
    _shm.buf[base + offset : base + offset + len(chunk)] = chunk


def _read_result(slot_id: int, result_sz: int) -> bytes:
    base = slot_id * _SLOT_TOTAL + _HEADER_SIZE
    return bytes(_shm.buf[base : base + result_sz])


# ---------------------------------------------------------------------------
# FastAPI app
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


def _bearer_token(request: Request) -> str:
    """Extract API key from ``Authorization: Bearer <key>`` header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in _AUTH_SKIP_PATHS:
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


BearerToken = Annotated[str, Depends(_bearer_token)]


@app.post("/infer")
async def infer(request: Request, api_key: BearerToken) -> Response:
    """Infer endpoint.

    Headers:
        Authorization: Bearer <api_key>    Required.

    Query params:
        model_id    Required. May contain '/' — URL-encode as %2F.
        task        Optional. Task name (e.g. "infer", "embed_text", "caption").
                    None → default task for this model.
        instance    Optional. Differentiates multiple workers of the same model
                    (e.g. "0", "1", "gpu0"). Routed as model_id:instance internally.
        device      Optional. Device hint for cold-path load (e.g. "cuda:0", "cuda:1").
                    No effect if model is already loaded.
        format      Optional. "json" or "pickle" (default: pickle).
        *           Any additional params forwarded opaquely to the backend.

    Body:
        Raw image bytes (e.g. Content-Type: image/jpeg).
        May be empty for text-only tasks (e.g. embed_text).
    """
    params = dict(request.query_params)
    model_id = params.pop("model_id", "")
    instance = params.pop("instance", "")
    device = params.pop("device", "")
    fmt = params.pop("format", "pickle")  # "json" or "pickle"
    if not model_id:
        return Response(status_code=400, content=b"model_id query param required")

    _t0 = time.monotonic()

    # 1. Ensure model is loaded (instant on warm path)
    status = await _ensure_loaded(model_id, instance, api_key, device)
    _t1 = time.monotonic()
    if status[0] == "load_timeout":
        return Response(
            status_code=503,
            headers={"Retry-After": str(status[1])},
            content=b"model loading, try again shortly",
        )
    if status[0] == "error":
        return Response(status_code=500, content=b"model load failed")

    # 2. Claim a SHM slot
    try:
        slot_id = await _alloc_slot(model_id, instance)
    except asyncio.TimeoutError:
        return Response(
            status_code=503,
            headers={"Retry-After": "1"},
            content=b"server busy, try again",
        )
    except RuntimeError:
        return Response(status_code=503, content=b"server busy, try again")
    _t2 = time.monotonic()

    # 3. Stream body into SHM  4. Submit  5. Read result
    # Slot is always freed in finally — even on early return paths above
    # that return before alloc succeed don't reach this block.
    try:
        pos = 0
        async for chunk in request.stream():
            if pos + len(chunk) > _SHM_DATA_SIZE:
                return Response(status_code=413, content=b"payload too large")
            if pos == 0 and not _looks_like_image(chunk):
                return Response(
                    status_code=415, content=b"body is not a recognized image format"
                )
            _write_input(slot_id, chunk, pos)
            pos += len(chunk)

        _t3 = time.monotonic()

        result = await _submit_and_wait(slot_id, model_id, instance, pos, params)
        _t4 = time.monotonic()

        print(
            f"[TIMING] ensure={(_t1-_t0)*1000:.1f}ms "
            f"alloc={(_t2-_t1)*1000:.1f}ms "
            f"stream={(_t3-_t2)*1000:.1f}ms "
            f"infer={(_t4-_t3)*1000:.1f}ms "
            f"total={(_t4-_t0)*1000:.1f}ms "
            f"body={pos}B",
            flush=True,
        )

        if result[0] == "error":
            return Response(status_code=500, content=b"inference failed")
        if result[0] != "result":
            return Response(
                status_code=500,
                content=b"internal error",
            )

        _, result_slot_id, result_sz = result
        raw = _read_result(result_slot_id, result_sz)

        if fmt == "json":
            obj = pickle.loads(raw)
            return Response(
                content=serialize_json(obj),
                media_type="application/json",
            )

        return Response(
            content=raw,
            media_type="application/octet-stream",
        )

    except asyncio.TimeoutError:
        return Response(status_code=504, content=b"inference timeout")

    finally:
        _free_slot(slot_id)


# ---------------------------------------------------------------------------
# Lifecycle endpoints (T_LOAD / T_UNLOAD via MMP)
# ---------------------------------------------------------------------------


async def _lifecycle_req(msg_type: bytes, payload: bytes, timeout_s: float = 30.0):
    """Send a lifecycle message to MMP and wait for T_OK or T_ERROR."""
    req_id = _new_req_id()
    fut = asyncio.get_running_loop().create_future()
    _pending[req_id] = fut

    full_payload = struct.pack(">Q", req_id) + payload
    await _sock.send_multipart([msg_type, full_payload])

    try:
        return await asyncio.wait_for(fut, timeout=timeout_s)
    except asyncio.TimeoutError:
        _pending.pop(req_id, None)
        raise


@app.post("/models/load")
async def load_model(request: Request, api_key: BearerToken) -> Response:
    """Trigger model load on MMP.

    Headers:
        Authorization: Bearer <api_key>    Required.

    Query params:
        model_id    Required.
    """
    params = dict(request.query_params)
    model_id = params.get("model_id", "")
    if not model_id:
        return Response(status_code=400, content=b"model_id query param required")

    mid_bytes = model_id.encode()
    key_bytes = api_key.encode()
    payload = (
        struct.pack(">H", len(mid_bytes))
        + mid_bytes
        + struct.pack(">H", len(key_bytes))
        + key_bytes
    )

    try:
        result = await _lifecycle_req(_T_LOAD, payload)
    except asyncio.TimeoutError:
        return Response(status_code=504, content=b"load request timeout")

    if result[0] == "error":
        return Response(status_code=500, content=b"load failed")
    return Response(status_code=200, content=f"load triggered: {model_id}".encode())


@app.post("/models/unload")
async def unload_model(request: Request, _token: BearerToken) -> Response:
    """Trigger model unload on MMP.

    Headers:
        Authorization: Bearer <api_key>    Required.

    Query params:
        model_id    Required.
    """
    params = dict(request.query_params)
    model_id = params.get("model_id", "")
    if not model_id:
        return Response(status_code=400, content=b"model_id query param required")

    mid_bytes = model_id.encode()
    payload = struct.pack(">H", len(mid_bytes)) + mid_bytes

    try:
        result = await _lifecycle_req(_T_UNLOAD, payload)
    except asyncio.TimeoutError:
        return Response(status_code=504, content=b"unload request timeout")

    if result[0] == "error":
        return Response(status_code=500, content=b"unload failed")
    return Response(status_code=200, content=f"unloaded: {model_id}".encode())


# ---------------------------------------------------------------------------
# Discovery endpoint
# ---------------------------------------------------------------------------


@app.get("/models/tasks")
async def get_model_tasks(request: Request, api_key: BearerToken) -> Response:
    """Discover supported tasks for a model WITHOUT loading it.

    Resolves model_id to model class via Roboflow API (cached 24h),
    reads class-level ``get_supported_tasks()``. No download, no GPU.

    Headers:
        Authorization: Bearer <api_key>    Required.

    Query params:
        model_id    Required.
    """
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return Response(status_code=400, content=b"model_id query param required")

    try:
        from inference_model_manager.dispatch import discover_tasks

        tasks = discover_tasks(model_id, api_key=api_key)
    except Exception:
        return Response(status_code=404, content=b"model not found or not compatible")

    import orjson

    return Response(
        content=orjson.dumps(tasks),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "inference_server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("NUM_WORKERS", "4")),
        loop="uvloop",
        http="httptools",
        log_level="error",
    )
