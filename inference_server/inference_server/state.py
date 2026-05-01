"""Per-process shared state: ZMQ socket, SHM, pending futures, protocol helpers.

Initialized once per uvicorn worker in the FastAPI lifespan handler (app.py).
Routers import from here to access MMP communication primitives.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import struct
import threading
import time
import uuid
from collections import deque
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import zmq.asyncio

from inference_model_manager.backends.utils.transport import zmq_addr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOAD_WAIT_S = float(os.environ.get("INFERENCE_LOAD_WAIT_S", "10.0"))
INFER_TIMEOUT_S = float(os.environ.get("INFERENCE_INFER_TIMEOUT_S", "30.0"))
ALLOC_TIMEOUT_S = float(os.environ.get("INFERENCE_ALLOC_TIMEOUT_S", "2.0"))

# SHM slot layout constants. MMP_ADDR, SHM_NAME, SHM_DATA_SIZE are set
# by server.py AFTER import but BEFORE uvicorn forks workers. With fork,
# module globals are copied at fork time — so these must be re-read from
# env in init_from_env() which runs in the lifespan (per-worker).
HEADER_SIZE = 64
SLOT_STATUS_ERROR = 5  # SlotStatus.ERROR

# Mutable — set by init_from_env() at lifespan start
MMP_ADDR: str = ""
SHM_NAME: str = ""
SHM_DATA_SIZE: int = 0
SLOT_TOTAL: int = 0


def init_from_env() -> None:
    """Re-read config from env. Called once per worker in lifespan."""
    global MMP_ADDR, SHM_NAME, SHM_DATA_SIZE, SLOT_TOTAL
    MMP_ADDR = os.environ.get("INFERENCE_MMP_ADDR") or zmq_addr("mmprocess")
    SHM_NAME = os.environ.get("INFERENCE_SHM_NAME", "inference_pool")
    SHM_DATA_SIZE = int(
        os.environ.get("INFERENCE_SHM_DATA_SIZE", str(25 * 1024 * 1024))
    )
    SLOT_TOTAL = HEADER_SIZE + SHM_DATA_SIZE


_OFF_STATUS = 0
_OFF_RESULT_SZ = 8

# ---------------------------------------------------------------------------
# ZMQ message types (must match model_manager_process.py)
# ---------------------------------------------------------------------------

# uvicorn -> MMP
T_ALLOC = b"\x01"
T_SUBMIT = b"\x02"
T_FREE = b"\x03"
T_ENSURE_LOADED = b"\x09"

# MMP -> uvicorn
T_ALLOC_OK = b"\x11"
T_RESULT_READY = b"\x14"
T_ERROR = b"\xFF"
T_MODEL_READY = b"\x0A"
T_LOAD_TIMEOUT = b"\x0B"
T_OK = b"\x40"

# uvicorn -> MMP (lifecycle)
T_LOAD = b"\x20"
T_UNLOAD = b"\x21"
T_STATS = b"\x30"

# MMP -> uvicorn (stats)
T_STATS_RESP = b"\x41"

# ---------------------------------------------------------------------------
# Per-process state (set in lifespan)
# ---------------------------------------------------------------------------

shm: Optional[SharedMemory] = None
sock: Optional[zmq.asyncio.Socket] = None
ctx: Optional[zmq.asyncio.Context] = None
pending: dict[int, asyncio.Future] = {}
recv_task: Optional[asyncio.Task] = None

# ---------------------------------------------------------------------------
# Image format detection
# ---------------------------------------------------------------------------

import filetype

_NPY_MAGIC = b"\x93NUMPY"


def looks_like_image(data: bytes | memoryview) -> bool:
    head = bytes(data[:262])
    if head[:6] == _NPY_MAGIC:
        return True
    return filetype.is_image(head)


# ---------------------------------------------------------------------------
# Slot header view
# ---------------------------------------------------------------------------


class SlotHeaderView:
    __slots__ = ("status", "result_size")

    def __init__(self, status: int, result_size: int) -> None:
        self.status = status
        self.result_size = result_size


def read_slot_header(slot_id: int) -> SlotHeaderView | None:
    off = slot_id * SLOT_TOTAL
    if off + HEADER_SIZE > len(shm.buf):
        return None
    status = shm.buf[off + _OFF_STATUS]
    result_size = struct.unpack_from("<I", shm.buf, off + _OFF_RESULT_SZ)[0]
    return SlotHeaderView(status, result_size)


# ---------------------------------------------------------------------------
# ZMQ recv loop + dispatch
# ---------------------------------------------------------------------------


async def recv_loop() -> None:
    while True:
        try:
            parts = await sock.recv_multipart()
        except (asyncio.CancelledError, zmq.ZMQError):
            break
        if not parts:
            continue
        _dispatch(parts[0], parts[1:])


def _dispatch(msg_type: bytes, frames: list[bytes]) -> None:
    try:
        if msg_type == T_ALLOC_OK:
            req_id, slot_id = struct.unpack(">QI", frames[0])
            _resolve(req_id, ("alloc_ok", slot_id))
        elif msg_type == T_RESULT_READY:
            req_id, slot_id, result_sz = struct.unpack(">QII", frames[0])
            _resolve(req_id, ("result", slot_id, result_sz))
        elif msg_type == T_MODEL_READY:
            req_id = struct.unpack(">Q", frames[0])[0]
            _resolve(req_id, ("model_ready",))
        elif msg_type == T_LOAD_TIMEOUT:
            req_id, retry_after = struct.unpack(">QI", frames[0])
            _resolve(req_id, ("load_timeout", retry_after))
        elif msg_type == T_OK:
            req_id = struct.unpack(">Q", frames[0])[0]
            _resolve(req_id, ("ok",))
        elif msg_type == T_ERROR:
            req_id = struct.unpack(">Q", frames[0][:8])[0]
            err = frames[0][8] if len(frames[0]) > 8 else 0
            _resolve(req_id, ("error", err))
        elif msg_type == T_STATS_RESP:
            req_id, json_len = struct.unpack_from(">QI", frames[0])
            json_bytes = frames[0][12 : 12 + json_len]
            _resolve(req_id, ("stats", json_bytes))
        else:
            logger.warning("_dispatch: unknown msg type %r", msg_type)
    except (struct.error, IndexError):
        logger.warning("_dispatch: malformed msg type=%r", msg_type, exc_info=True)


def _resolve(req_id: int, value: tuple) -> None:
    fut = pending.pop(req_id, None)
    if fut is not None and not fut.done():
        fut.set_result(value)


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


def new_req_id() -> int:
    return uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF


def make_future(req_id: int) -> asyncio.Future:
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    pending[req_id] = fut
    return fut


def drop_future(req_id: int) -> None:
    pending.pop(req_id, None)


def routing_key(model_id: str, instance: str) -> str:
    return f"{model_id}:{instance}" if instance else model_id


async def ensure_loaded(
    model_id: str,
    instance: str = "",
    api_key: str = "",
    device: str = "",
) -> tuple:
    req_id = new_req_id()
    mid_bytes = routing_key(model_id, instance).encode()
    key_bytes = api_key.encode()
    dev_bytes = device.encode()
    wait_ms = int(LOAD_WAIT_S * 1000)
    payload = (
        struct.pack(">QIH", req_id, wait_ms, len(mid_bytes))
        + mid_bytes
        + struct.pack(">H", len(key_bytes))
        + key_bytes
        + struct.pack(">H", len(dev_bytes))
        + dev_bytes
    )
    fut = make_future(req_id)
    await sock.send_multipart([T_ENSURE_LOADED, payload])
    try:
        return await asyncio.wait_for(fut, timeout=LOAD_WAIT_S + 1.0)
    except asyncio.TimeoutError:
        drop_future(req_id)
        return ("load_timeout", int(LOAD_WAIT_S))


async def alloc_slot(model_id: str, instance: str = "") -> int:
    req_id = new_req_id()
    mid = routing_key(model_id, instance).encode()
    payload = struct.pack(">QH", req_id, len(mid)) + mid
    fut = make_future(req_id)
    await sock.send_multipart([T_ALLOC, payload])
    try:
        result = await asyncio.wait_for(fut, timeout=ALLOC_TIMEOUT_S)
    except asyncio.TimeoutError:
        drop_future(req_id)
        raise
    if result[0] == "error":
        raise RuntimeError(f"alloc error code={result[1]}")
    return result[1]


async def submit_and_wait(
    slot_id: int,
    model_id: str,
    instance: str,
    input_sz: int,
    params: dict,
    request=None,
) -> tuple:
    req_id = new_req_id()
    mid = routing_key(model_id, instance).encode()
    header = struct.pack(">QIIH", req_id, slot_id, input_sz, len(mid)) + mid
    params_json = json.dumps(params).encode() if params else b"{}"
    fut = make_future(req_id)
    await sock.send_multipart([T_SUBMIT, header, params_json])
    try:
        if request is not None:
            # Race inference future against client disconnect
            return await _wait_with_disconnect(fut, request)
        return await asyncio.wait_for(fut, timeout=INFER_TIMEOUT_S)
    except asyncio.TimeoutError:
        drop_future(req_id)
        raise
    except _ClientDisconnected:
        drop_future(req_id)
        raise


class _ClientDisconnected(Exception):
    """Raised when client disconnects during submit_and_wait."""


async def _wait_with_disconnect(fut: asyncio.Future, request) -> tuple:
    """Wait for inference result, but bail early if client disconnects."""

    async def _poll_disconnect():
        while True:
            if await request.is_disconnected():
                return True
            await asyncio.sleep(0.5)

    disconnect_task = asyncio.ensure_future(_poll_disconnect())
    infer_task = asyncio.ensure_future(asyncio.wait_for(fut, timeout=INFER_TIMEOUT_S))

    done, pending_tasks = await asyncio.wait(
        [disconnect_task, infer_task], return_when=asyncio.FIRST_COMPLETED
    )

    for t in pending_tasks:
        t.cancel()

    if disconnect_task in done:
        # Client gone — inference may still complete (slot freed by caller)
        raise _ClientDisconnected()

    # Inference completed (or timed out) — propagate result or exception
    return infer_task.result()


def free_slot(slot_id: int) -> None:
    async def _send():
        try:
            await sock.send_multipart([T_FREE, struct.pack(">I", slot_id)])
        except Exception:
            logger.warning("free_slot: failed for slot %d", slot_id, exc_info=True)

    task = asyncio.create_task(_send())
    task.add_done_callback(
        lambda t: t.result() if not t.cancelled() and t.exception() is None else None
    )


def write_input(slot_id: int, chunk: bytes | memoryview, offset: int) -> None:
    base = slot_id * SLOT_TOTAL + HEADER_SIZE
    end = base + offset + len(chunk)
    if base < 0 or end > len(shm.buf):
        raise ValueError(
            f"SHM write out of bounds: slot={slot_id} offset={offset} len={len(chunk)}"
        )
    shm.buf[base + offset : end] = chunk


def read_result(slot_id: int, result_sz: int) -> bytes:
    if result_sz > SHM_DATA_SIZE or result_sz < 0:
        raise ValueError(f"result_sz {result_sz} out of bounds (max {SHM_DATA_SIZE})")
    if slot_id < 0 or slot_id * SLOT_TOTAL + HEADER_SIZE + result_sz > len(shm.buf):
        raise ValueError(f"slot_id {slot_id} out of bounds")
    base = slot_id * SLOT_TOTAL + HEADER_SIZE
    return bytes(shm.buf[base : base + result_sz])


async def lifecycle_req(msg_type: bytes, payload: bytes, timeout_s: float = 30.0):
    req_id = new_req_id()
    fut = make_future(req_id)
    full_payload = struct.pack(">Q", req_id) + payload
    await sock.send_multipart([msg_type, full_payload])
    try:
        return await asyncio.wait_for(fut, timeout=timeout_s)
    except asyncio.TimeoutError:
        drop_future(req_id)
        raise


async def fetch_stats(timeout_s: float = 5.0) -> dict:
    result = await lifecycle_req(T_STATS, b"", timeout_s=timeout_s)
    if result[0] == "stats":
        return json.loads(result[1])
    return {}


# ---------------------------------------------------------------------------
# Pipeline timing CSV
# ---------------------------------------------------------------------------

_PIPELINE_CSV = os.environ.get("INFERENCE_PIPELINE_CSV", "")
_PIPELINE_FLUSH_INTERVAL_S = float(os.environ.get("INFERENCE_PIPELINE_FLUSH_S", "5.0"))
TIMING = bool(_PIPELINE_CSV)

_CSV_FIELDS = [
    "timestamp",
    "worker_pid",
    "endpoint",
    "payload_bytes",
    "status",
    "body_extract_ms",
    "ensure_loaded_ms",
    "zmq_alloc_ms",
    "shm_write_ms",
    "zmq_submit_ms",
    "zmq_result_wait_ms",
    "result_read_ms",
    "serialize_ms",
    "total_ms",
]

_pipeline_buffer: deque[dict] = deque()
_pipeline_lock = threading.Lock()
_pipeline_thread: Optional[threading.Thread] = None


def pipeline_record(rec: dict) -> None:
    """Append a timing record. No-op if INFERENCE_PIPELINE_CSV not set."""
    if not _PIPELINE_CSV:
        return
    with _pipeline_lock:
        _pipeline_buffer.append(rec)


def start_pipeline_csv_writer() -> None:
    """Start background thread that flushes pipeline records to CSV."""
    global _pipeline_thread
    if not _PIPELINE_CSV:
        return
    if _pipeline_thread is not None:
        return

    csv_path = _PIPELINE_CSV

    # Write header only if file doesn't exist or is empty
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    def _writer():
        while True:
            time.sleep(_PIPELINE_FLUSH_INTERVAL_S)
            with _pipeline_lock:
                batch = list(_pipeline_buffer)
                _pipeline_buffer.clear()
            if not batch:
                continue
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
                for rec in batch:
                    w.writerow(rec)

    _pipeline_thread = threading.Thread(target=_writer, daemon=True, name="pipeline-csv")
    _pipeline_thread.start()
    logger.info("Pipeline CSV writer started: %s (flush every %.1fs)",
                csv_path, _PIPELINE_FLUSH_INTERVAL_S)
