"""FastAPI benchmark app — orchestrated ModelManagerProcess mode.

Accepts raw image bytes in the request body.
Infer parameters (including model_id) are passed as query params.
model_id may contain '/' (URL-encode as %2F; Starlette decodes automatically).

Usage::

    POST /infer?model_id=yolov8n%2F640&confidence=0.5
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
    INFERENCE_SHM_INPUT_SIZE    Bytes per slot input area  (default: 25 MB)
    INFERENCE_SHM_RESULT_SIZE   Bytes per slot result area (default: 4 MB)
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
import struct
import uuid
from contextlib import asynccontextmanager
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request, Response

from inference_models.backends.utils.transport import zmq_addr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MMP_ADDR        = os.environ.get("INFERENCE_MMP_ADDR") or zmq_addr("mmprocess")
_SHM_NAME        = os.environ.get("INFERENCE_SHM_NAME", "inference_pool")
_SHM_INPUT_SIZE  = int(os.environ.get("INFERENCE_SHM_INPUT_SIZE",  str(25 * 1024 * 1024)))
_SHM_RESULT_SIZE = int(os.environ.get("INFERENCE_SHM_RESULT_SIZE", str(4  * 1024 * 1024)))
_LOAD_WAIT_S     = float(os.environ.get("INFERENCE_LOAD_WAIT_S",     "10.0"))
_INFER_TIMEOUT_S = float(os.environ.get("INFERENCE_INFER_TIMEOUT_S", "30.0"))
_ALLOC_TIMEOUT_S = float(os.environ.get("INFERENCE_ALLOC_TIMEOUT_S",  "2.0"))

# SHM slot layout (must match shm_pool.py):
#   [HEADER 64B | INPUT _SHM_INPUT_SIZE | RESULT _SHM_RESULT_SIZE]
_HEADER_SIZE = 64
_SLOT_TOTAL  = _HEADER_SIZE + _SHM_INPUT_SIZE + _SHM_RESULT_SIZE

# ---------------------------------------------------------------------------
# Image magic-byte prefixes (first bytes of each supported format)
# ---------------------------------------------------------------------------

_IMAGE_MAGICS: tuple[bytes, ...] = (
    b"\xff\xd8",        # JPEG
    b"\x89PNG",         # PNG
    b"RIFF",            # WebP  (RIFF????WEBP — prefix sufficient)
    b"GIF8",            # GIF
    b"BM",              # BMP
    b"II*\x00",         # TIFF little-endian
    b"MM\x00*",         # TIFF big-endian
    b"\x93NUMPY",       # numpy .npy (standalone/direct mode)
)
_IMAGE_MAGIC_MAX = max(len(m) for m in _IMAGE_MAGICS)


def _looks_like_image(data: bytes | memoryview) -> bool:
    head = bytes(data[:_IMAGE_MAGIC_MAX])
    return any(head[:len(m)] == m for m in _IMAGE_MAGICS)


# ---------------------------------------------------------------------------
# ZMQ message types (must match model_manager_process.py)
# ---------------------------------------------------------------------------

# uvicorn → MMP
_T_ALLOC         = b'\x01'
_T_SUBMIT        = b'\x02'
_T_FREE          = b'\x03'
_T_ENSURE_LOADED = b'\x09'

# MMP → uvicorn
_T_ALLOC_OK      = b'\x11'
_T_RESULT_READY  = b'\x14'
_T_ERROR         = b'\xFF'
_T_MODEL_READY   = b'\x0A'
_T_LOAD_TIMEOUT  = b'\x0B'

# Wire formats (big-endian):
#   _T_ALLOC:         Q H N    req_id(8) flavor_len(2) flavor(N)
#   _T_ALLOC_OK:      Q I      req_id(8) slot_id(4)
#   _T_SUBMIT:        Q I I H N  req_id(8) slot_id(4) input_sz(4) flavor_len(2) flavor(N)
#                     + frame2: json params bytes
#   _T_RESULT_READY:  Q I I    req_id(8) slot_id(4) result_sz(4)
#   _T_FREE:          I        slot_id(4)
#   _T_ERROR:         Q B      req_id(8) error_code(1)
#   _T_ENSURE_LOADED: Q I H N H M  req_id(8) wait_ms(4) flavor_len(2) flavor(N) key_len(2) key(M)
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

_shm:     Optional[SharedMemory]       = None
_sock:    Optional[zmq.asyncio.Socket] = None
_ctx:     Optional[zmq.asyncio.Context]= None
_pending: dict[int, asyncio.Future]    = {}
_recv_task: Optional[asyncio.Task]     = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _shm, _sock, _ctx, _pending, _recv_task

    identity = f"uv_{os.getpid()}_{uuid.uuid4().hex[:8]}".encode()

    _ctx  = zmq.asyncio.Context()
    _sock = _ctx.socket(zmq.DEALER)
    _sock.setsockopt(zmq.IDENTITY, identity)
    _sock.setsockopt(zmq.SNDHWM, 0)   # unlimited send buffer
    _sock.setsockopt(zmq.RCVHWM, 0)   # unlimited recv buffer
    _sock.setsockopt(zmq.LINGER, 0)   # don't block on close
    _sock.connect(_MMP_ADDR)

    _shm     = SharedMemory(name=_SHM_NAME, create=False)
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

async def _ensure_loaded(model_id: str, api_key: str = "") -> tuple:
    """Ask MMP to ensure model is loaded.

    Warm path (model already loaded): MMP replies T_MODEL_READY instantly.
    Cold path: MMP loads the model using api_key (falls back to server default);
    replies when ready or wait_ms exceeded.

    Returns one of:
        ("model_ready",)
        ("load_timeout", retry_after_s)
        ("error", code)
    """
    req_id  = _new_req_id()
    flavor  = model_id.encode()
    key     = api_key.encode()
    wait_ms = int(_LOAD_WAIT_S * 1000)
    # wire: Q I H N H M  — req_id(8) wait_ms(4) flavor_len(2) flavor(N) key_len(2) key(M)
    payload = (
        struct.pack(">QIH", req_id, wait_ms, len(flavor))
        + flavor
        + struct.pack(">H", len(key))
        + key
    )
    fut = _make_future(req_id)
    await _sock.send_multipart([_T_ENSURE_LOADED, payload])
    try:
        # +1s headroom beyond MMP's own budget
        return await asyncio.wait_for(fut, timeout=_LOAD_WAIT_S + 1.0)
    except asyncio.TimeoutError:
        _drop_future(req_id)
        return ("load_timeout", int(_LOAD_WAIT_S))


async def _alloc_slot(model_id: str) -> int:
    """Claim a SHM slot from MMP. Returns slot_id.

    Raises asyncio.TimeoutError if no slot granted within ALLOC_TIMEOUT_S.
    Raises RuntimeError if MMP replies with an error.
    """
    req_id  = _new_req_id()
    flavor  = model_id.encode()
    payload = struct.pack(">QH", req_id, len(flavor)) + flavor
    fut     = _make_future(req_id)
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
    slot_id:  int,
    model_id: str,
    input_sz: int,
    params:   dict,
) -> tuple:
    """Signal MMP that slot data is ready; await the inference result.

    Returns one of:
        ("result", slot_id, result_sz)
        ("error",  code)

    Raises asyncio.TimeoutError if no result within INFER_TIMEOUT_S.
    """
    req_id      = _new_req_id()
    flavor      = model_id.encode()
    header      = struct.pack(">QIIH", req_id, slot_id, input_sz, len(flavor)) + flavor
    params_json = json.dumps(params).encode() if params else b"{}"
    fut         = _make_future(req_id)
    await _sock.send_multipart([_T_SUBMIT, header, params_json])
    try:
        return await asyncio.wait_for(fut, timeout=_INFER_TIMEOUT_S)
    except asyncio.TimeoutError:
        _drop_future(req_id)
        raise


def _free_slot(slot_id: int) -> None:
    asyncio.create_task(
        _sock.send_multipart([_T_FREE, struct.pack(">I", slot_id)])
    )


def _write_input(slot_id: int, chunk: bytes | memoryview, offset: int) -> None:
    base = slot_id * _SLOT_TOTAL + _HEADER_SIZE
    _shm.buf[base + offset: base + offset + len(chunk)] = chunk


def _read_result(slot_id: int, result_sz: int) -> bytes:
    base = slot_id * _SLOT_TOTAL + _HEADER_SIZE + _SHM_INPUT_SIZE
    return bytes(_shm.buf[base: base + result_sz])


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=_lifespan)


@app.post("/infer")
async def infer(request: Request) -> Response:
    """Infer endpoint.

    Query params:
        model_id    Required. May contain '/' — URL-encode as %2F.
        *           Any additional params forwarded opaquely to the backend.

    Body:
        Raw image bytes (e.g. Content-Type: image/jpeg).
    """
    params   = dict(request.query_params)
    model_id = params.pop("model_id", "")
    api_key  = params.pop("api_key", "")
    if not model_id:
        return Response(status_code=400, content=b"model_id query param required")

    # 1. Ensure model is loaded (instant on warm path)
    status = await _ensure_loaded(model_id, api_key)
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
        slot_id = await _alloc_slot(model_id)
    except asyncio.TimeoutError:
        return Response(
            status_code=503,
            headers={"Retry-After": "1"},
            content=b"no SHM slots available",
        )
    except RuntimeError as exc:
        return Response(status_code=503, content=str(exc).encode())

    # 3. Stream body into SHM  4. Submit  5. Read result
    # Slot is always freed in finally — even on early return paths above
    # that return before alloc succeed don't reach this block.
    try:
        pos = 0
        async for chunk in request.stream():
            if pos + len(chunk) > _SHM_INPUT_SIZE:
                return Response(status_code=413, content=b"payload exceeds slot capacity")
            if pos == 0 and not _looks_like_image(chunk):
                return Response(status_code=415, content=b"body is not a recognized image format")
            _write_input(slot_id, chunk, pos)
            pos += len(chunk)

        if pos == 0:
            return Response(status_code=400, content=b"empty body")

        result = await _submit_and_wait(slot_id, model_id, pos, params)

        if result[0] == "error":
            return Response(status_code=500, content=b"inference failed")
        if result[0] != "result":
            return Response(
                status_code=500,
                content=f"unexpected MMP reply: {result[0]}".encode(),
            )

        _, result_slot_id, result_sz = result
        return Response(
            content=_read_result(result_slot_id, result_sz),
            media_type="application/octet-stream",
        )

    except asyncio.TimeoutError:
        return Response(status_code=504, content=b"inference timeout")

    finally:
        _free_slot(slot_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "inference_models.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("NUM_WORKERS", "4")),
        loop="uvloop",
        http="httptools",
        log_level="error",
    )
