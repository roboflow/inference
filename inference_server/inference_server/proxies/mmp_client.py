"""MMPClient — ZMQ DEALER + SHMPool client to ModelManagerProcess.

Owns the FastAPI-worker-side connection state:
  - DEALER socket connected to MMP ROUTER
  - SHMPool attach by name
  - `_pending: dict[req_id, asyncio.Future]` + recv-loop task

One instance per FastAPI worker process. Constructed and started in app
lifespan. Method names match `ModelManagerProxy` (see proxies/base.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import struct
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from typing import Any, AsyncIterable, Optional

import zmq.asyncio
from fastapi import Request

from inference_model_manager.backends.utils.shm_pool import read_free_count
from inference_model_manager.backends.utils.transport import zmq_addr
from inference_server import configuration
from inference_server.errors import (
    PayloadTooLargeError,
    ServerBusyError,
    UploadTooSlowError,
)
from inference_server.image_headers import image_pixels
from inference_server.proxies.base import ClientDisconnected

logger = logging.getLogger(__name__)

# Optional resolution reject gate: max decoded pixels per image (0 = off).
_MAX_DECODED_PIXELS = (
    int(configuration.MAX_DECODED_MEGAPIXELS * 1_000_000)
    if configuration.MAX_DECODED_MEGAPIXELS > 0
    else 0
)


# ---------------------------------------------------------------------------
# Wire protocol constants (match model_manager_process.py)
# ---------------------------------------------------------------------------

# uvicorn -> MMP
T_ALLOC = b"\x01"
T_SUBMIT = b"\x02"
T_FREE = b"\x03"
T_CANCEL = b"\x04"
T_ENSURE_LOADED = b"\x09"
T_LOAD = b"\x20"
T_UNLOAD = b"\x21"
T_STATS = b"\x30"
T_SHM_INFO = b"\x31"

# MMP -> uvicorn
T_ALLOC_OK = b"\x11"
T_RESULT_READY = b"\x14"
T_ERROR = b"\xff"
T_MODEL_READY = b"\x0a"
T_LOAD_TIMEOUT = b"\x0b"
T_OK = b"\x40"
T_STATS_RESP = b"\x41"
T_SHM_INFO_RESP = b"\x42"

# Slot layout
HEADER_SIZE = 64
SLOT_STATUS_ERROR = 5
_OFF_STATUS = 0
_OFF_RESULT_SZ = 8


class _SlotHeaderView:
    __slots__ = ("status", "result_size")

    def __init__(self, status: int, result_size: int) -> None:
        self.status = status
        self.result_size = result_size


# ---------------------------------------------------------------------------
# MMPClient — ModelManagerProxy impl over ZMQ + SHM
# ---------------------------------------------------------------------------


class MMPClient:
    """ZMQ DEALER + SHMPool client for ModelManagerProcess.

    Lifecycle:
        client = MMPClient()
        await client.start()       # lifespan startup
        ...                        # use as ModelManagerProxy
        await client.shutdown()    # lifespan teardown
    """

    def __init__(
        self,
        *,
        mmp_addr: Optional[str] = None,
        shm_name: Optional[str] = None,
        shm_data_size: Optional[int] = None,
        n_slots: Optional[int] = None,
        load_wait_s: Optional[float] = None,
        infer_timeout_s: Optional[float] = None,
        alloc_timeout_s: Optional[float] = None,
    ) -> None:
        cfg = configuration
        self.mmp_addr = (
            mmp_addr
            or os.environ.get(cfg.INFERENCE_MMP_ADDR_ENV)
            or zmq_addr("mmprocess")
        )
        self.shm_name = shm_name or os.environ.get(
            cfg.INFERENCE_SHM_NAME_ENV, cfg.INFERENCE_SHM_NAME_DEFAULT
        )
        self.shm_data_size = shm_data_size or int(
            os.environ.get(
                cfg.INFERENCE_SHM_DATA_SIZE_ENV,
                str(cfg.INFERENCE_SHM_DATA_SIZE_DEFAULT),
            )
        )
        self.slot_total = HEADER_SIZE + self.shm_data_size
        self.n_slots = n_slots if n_slots is not None else cfg.SERVER_N_SLOTS
        self.load_wait_s = load_wait_s if load_wait_s is not None else cfg.LOAD_WAIT_S
        self.infer_timeout_s = (
            infer_timeout_s if infer_timeout_s is not None else cfg.INFER_TIMEOUT_S
        )
        self.alloc_timeout_s = (
            alloc_timeout_s if alloc_timeout_s is not None else cfg.ALLOC_TIMEOUT_S
        )

        self.ensure_cache_ttl_s = cfg.ENSURE_CACHE_TTL_S
        self._loaded_cache: dict[str, float] = {}
        self.shm_admission = cfg.SHM_ADMISSION

        self._ctx: Optional[zmq.asyncio.Context] = None
        self._sock: Optional[zmq.asyncio.Socket] = None
        self._shm: Optional[SharedMemory] = None
        self._pending: dict[int, asyncio.Future] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._bg_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle (lifespan)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        identity = f"uv_{os.getpid()}_{uuid.uuid4().hex[:8]}".encode()
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.IDENTITY, identity)
        self._sock.setsockopt(zmq.SNDHWM, 0)
        self._sock.setsockopt(zmq.RCVHWM, 0)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.mmp_addr)

        self._recv_task = asyncio.create_task(self._recv_loop(), name="zmq-recv")

        # MMP owns the pool — ask it for the shm name and geometry rather than
        # relying on env agreement (the name is generated at pool creation).
        info = await self._lifecycle_req(T_SHM_INFO, b"", timeout_s=5.0)
        if info[0] != "shm_info":
            raise RuntimeError(f"shm-info handshake failed: {info!r}")
        _, self.n_slots, self.shm_data_size, self.shm_name = info
        self.slot_total = HEADER_SIZE + self.shm_data_size

        self._shm = SharedMemory(name=self.shm_name, create=False)
        # OS rounds the mmap up to a page boundary, so the buffer holds n_slots
        # slots plus < 1 slot of trailing padding. Anything outside that range
        # means slot_total disagrees with the pool — writes would corrupt slots.
        expected = self.n_slots * self.slot_total
        buf_size = len(self._shm.buf)
        if not (expected <= buf_size < expected + self.slot_total):
            raise RuntimeError(
                f"SHM geometry mismatch: pool size {buf_size} does not hold "
                f"{self.n_slots} slots of slot_total {self.slot_total} "
                f"(expected {expected}..{expected + self.slot_total}) — "
                f"segment {self.shm_name!r} does not match the geometry MMP "
                "reported over T_SHM_INFO"
            )

    async def shutdown(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._sock is not None:
            self._sock.close()
        if self._ctx is not None:
            self._ctx.term()
        if self._shm is not None:
            self._shm.close()

    # ------------------------------------------------------------------
    # ModelManagerProxy interface
    # ------------------------------------------------------------------

    async def ensure_loaded(
        self,
        model_id: str,
        instance: str = "",
        api_key: str = "",
        device: str = "",
    ) -> tuple:
        routing_key = _routing_key(model_id, instance)
        exp = self._loaded_cache.get(routing_key)
        if exp is not None and exp > time.monotonic():
            return ("model_ready",)

        req_id = _new_req_id()
        mid_bytes = routing_key.encode()
        key_bytes = api_key.encode()
        dev_bytes = device.encode()
        wait_ms = int(self.load_wait_s * 1000)
        payload = (
            struct.pack(">QIH", req_id, wait_ms, len(mid_bytes))
            + mid_bytes
            + struct.pack(">H", len(key_bytes))
            + key_bytes
            + struct.pack(">H", len(dev_bytes))
            + dev_bytes
        )
        fut = self._make_future(req_id)
        await self._sock.send_multipart([T_ENSURE_LOADED, payload])
        try:
            result = await asyncio.wait_for(fut, timeout=self.load_wait_s + 1.0)
            if result[0] == "model_ready" and self.ensure_cache_ttl_s > 0:
                self._loaded_cache[routing_key] = (
                    time.monotonic() + self.ensure_cache_ttl_s
                )
            return result
        except asyncio.TimeoutError:
            self._drop_future(req_id)
            return ("load_timeout", int(self.load_wait_s))
        except asyncio.CancelledError:
            self._drop_future(req_id)
            raise

    async def load(self, model_id: str, api_key: str = "") -> tuple:
        mid_bytes = model_id.encode()
        key_bytes = api_key.encode()
        payload = (
            struct.pack(">H", len(mid_bytes))
            + mid_bytes
            + struct.pack(">H", len(key_bytes))
            + key_bytes
        )
        return await self._lifecycle_req(T_LOAD, payload)

    async def unload(self, model_id: str) -> tuple:
        mid_bytes = model_id.encode()
        payload = struct.pack(">H", len(mid_bytes)) + mid_bytes
        return await self._lifecycle_req(T_UNLOAD, payload)

    async def infer(
        self,
        *,
        model_id: str,
        image: bytes,
        task: Optional[str] = None,
        instance: str = "",
        params: Optional[dict] = None,
        request: Optional[Request] = None,
        raw_pickle: bool = False,
    ) -> Any:
        if len(image) > self.shm_data_size:
            raise PayloadTooLargeError(
                f"image exceeds slot size ({len(image)} > {self.shm_data_size})"
            )
        if _MAX_DECODED_PIXELS:
            px = image_pixels(image)
            if px and px > _MAX_DECODED_PIXELS:
                raise PayloadTooLargeError(
                    f"image {px / 1e6:.1f}MP exceeds max "
                    f"{_MAX_DECODED_PIXELS / 1e6:.0f}MP"
                )
        effective_params = dict(params) if params else {}
        if task:
            effective_params.setdefault("task", task)

        req_id = _new_req_id()
        slot_id = await self._alloc_slot(req_id, model_id, instance)
        submitted = False
        done = False
        try:
            self._write_input(slot_id, image, 0)
            # Mark BEFORE awaiting: if _submit_and_wait raises (timeout /
            # disconnect) after T_SUBMIT went out, the finally block must
            # cancel — not free a slot the worker still holds.
            submitted = True
            result = await self._submit_and_wait(
                req_id,
                slot_id,
                model_id,
                instance,
                len(image),
                effective_params,
                request=request,
            )

            if result[0] == "error":
                raise RuntimeError("inference failed")
            if result[0] != "result":
                raise RuntimeError(f"unexpected result type: {result[0]!r}")

            # Got a real result: the worker has finished writing this slot and
            # will not touch it again, so the client now owns the free.
            done = True
            _, result_slot_id, result_sz = result
            hdr = self._read_slot_header(result_slot_id)
            if hdr is not None and hdr.status == SLOT_STATUS_ERROR:
                err_msg = "inference failed"
                if hdr.result_size > 0:
                    err_msg = self._read_result(result_slot_id, hdr.result_size).decode(
                        "utf-8", errors="replace"
                    )
                raise RuntimeError(err_msg)

            raw = self._read_result(result_slot_id, result_sz)
            return self._decode_result(raw, raw_pickle)
        finally:
            if done:
                # Worker finished, we read the result → safe to free.
                self._free_slot(slot_id, req_id)
            elif submitted:
                # Gave up (timeout / disconnect / error) while the worker may
                # still hold a ticket for this slot. Do NOT free — that would let
                # the slot be reused under a live ticket. Cancel instead; MMP
                # frees the slot once the worker drains the ticket (or the reaper
                # reclaims it).
                self._cancel_req(req_id)
            else:
                # Never submitted (alloc ok but write/submit failed) → no worker
                # ticket exists → safe to free directly.
                self._free_slot(slot_id, req_id)

    async def infer_stream(
        self,
        *,
        model_id: str,
        image_chunks: AsyncIterable[bytes],
        content_length: Optional[int] = None,
        task: Optional[str] = None,
        instance: str = "",
        params: Optional[dict] = None,
        request: Optional[Request] = None,
        raw_pickle: bool = False,
        upload_timeout_s: Optional[float] = None,
    ) -> Any:
        if content_length is not None and content_length > self.shm_data_size:
            raise PayloadTooLargeError(
                f"image exceeds slot size ({content_length} > {self.shm_data_size})"
            )

        effective_params = dict(params) if params else {}
        if task:
            effective_params.setdefault("task", task)

        req_id = _new_req_id()
        slot_id = await self._alloc_slot(req_id, model_id, instance)
        submitted = False
        done = False
        input_sz = 0
        deadline = (
            time.monotonic() + upload_timeout_s
            if upload_timeout_s is not None
            else None
        )
        try:
            aiter = image_chunks.__aiter__()
            while True:
                try:
                    if deadline is None:
                        chunk = await aiter.__anext__()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise UploadTooSlowError("upload too slow")
                        chunk = await asyncio.wait_for(aiter.__anext__(), remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise UploadTooSlowError("upload too slow") from None
                if not chunk:
                    continue
                if input_sz + len(chunk) > self.shm_data_size:
                    raise PayloadTooLargeError(
                        f"image exceeds slot size ({input_sz + len(chunk)} > "
                        f"{self.shm_data_size})"
                    )
                self._write_input(slot_id, chunk, input_sz)
                input_sz += len(chunk)

            if input_sz == 0:
                raise ValueError("empty body")
            if content_length is not None and input_sz != content_length:
                raise ValueError(
                    f"content-length mismatch ({content_length} != {input_sz})"
                )

            submitted = True
            result = await self._submit_and_wait(
                req_id,
                slot_id,
                model_id,
                instance,
                input_sz,
                effective_params,
                request=request,
            )

            if result[0] == "error":
                raise RuntimeError("inference failed")
            if result[0] != "result":
                raise RuntimeError(f"unexpected result type: {result[0]!r}")

            done = True
            _, result_slot_id, result_sz = result
            hdr = self._read_slot_header(result_slot_id)
            if hdr is not None and hdr.status == SLOT_STATUS_ERROR:
                err_msg = "inference failed"
                if hdr.result_size > 0:
                    err_msg = self._read_result(result_slot_id, hdr.result_size).decode(
                        "utf-8", errors="replace"
                    )
                raise RuntimeError(err_msg)

            return self._decode_result(
                self._read_result(result_slot_id, result_sz), raw_pickle
            )
        finally:
            if done:
                self._free_slot(slot_id, req_id)
            elif submitted:
                self._cancel_req(req_id)
            else:
                self._free_slot(slot_id, req_id)

    async def stats(self) -> dict:
        result = await self._lifecycle_req(T_STATS, b"", timeout_s=5.0)
        if result[0] == "stats":
            return json.loads(result[1])
        # Swallowing this as {} made /v2/server/ready report 200 with a dead
        # MMP — raise so callers hit their 503 paths.
        raise RuntimeError(f"stats request failed: {result!r}")

    async def interface(self, model_id: str) -> dict:
        stats = await self.stats()
        models = stats.get("mmp_models", {})
        info = models.get(model_id)
        if info is None:
            raise RuntimeError(f"model '{model_id}' is not loaded")
        return {"model_id": model_id, "tasks": info.get("tasks", {})}

    # ------------------------------------------------------------------
    # Private helpers (ZMQ + SHM internals)
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        while True:
            try:
                parts = await self._sock.recv_multipart()
            except (asyncio.CancelledError, zmq.ZMQError):
                break
            if not parts:
                continue
            self._dispatch(parts[0], parts[1:])

    def _dispatch(self, msg_type: bytes, frames: list[bytes]) -> None:
        try:
            if msg_type == T_ALLOC_OK:
                req_id, slot_id = struct.unpack(">QI", frames[0])
                self._resolve(req_id, ("alloc_ok", slot_id))
            elif msg_type == T_RESULT_READY:
                req_id, slot_id, sz = struct.unpack(">QII", frames[0])
                self._resolve(req_id, ("result", slot_id, sz))
            elif msg_type == T_MODEL_READY:
                req_id = struct.unpack(">Q", frames[0])[0]
                self._resolve(req_id, ("model_ready",))
            elif msg_type == T_LOAD_TIMEOUT:
                req_id, retry_after = struct.unpack(">QI", frames[0])
                self._resolve(req_id, ("load_timeout", retry_after))
            elif msg_type == T_OK:
                req_id = struct.unpack(">Q", frames[0])[0]
                self._resolve(req_id, ("ok",))
            elif msg_type == T_ERROR:
                req_id = struct.unpack(">Q", frames[0][:8])[0]
                err = frames[0][8] if len(frames[0]) > 8 else 0
                self._resolve(req_id, ("error", err))
            elif msg_type == T_STATS_RESP:
                req_id, json_len = struct.unpack_from(">QI", frames[0])
                json_bytes = frames[0][12 : 12 + json_len]
                self._resolve(req_id, ("stats", json_bytes))
            elif msg_type == T_SHM_INFO_RESP:
                req_id, n_slots, data_size, name_len = struct.unpack_from(
                    ">QIQH", frames[0]
                )
                off = struct.calcsize(">QIQH")
                name = frames[0][off : off + name_len].decode()
                self._resolve(req_id, ("shm_info", n_slots, data_size, name))
            else:
                logger.warning("_dispatch: unknown msg type %r", msg_type)
        except (struct.error, IndexError):
            logger.warning("_dispatch: malformed msg type=%r", msg_type, exc_info=True)

    def _resolve(self, req_id: int, value: tuple) -> None:
        fut = self._pending.pop(req_id, None)
        if fut is not None and not fut.done():
            fut.set_result(value)

    def _make_future(self, req_id: int) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        return fut

    def _drop_future(self, req_id: int) -> None:
        self._pending.pop(req_id, None)

    def _decode_result(self, raw: bytes, raw_pickle: bool) -> Any:
        if raw_pickle:
            return raw
        try:
            return pickle.loads(raw)
        except Exception as exc:
            raise RuntimeError("result deserialization failed") from exc

    async def _alloc_slot(self, req_id: int, model_id: str, instance: str = "") -> int:
        if self.shm_admission and self._shm is not None:
            free = read_free_count(self._shm.buf, self.n_slots, self.slot_total)
            if free == 0:
                raise ServerBusyError("no capacity (admission: pool full)")

        mid = _routing_key(model_id, instance).encode()
        payload = struct.pack(">QH", req_id, len(mid)) + mid
        fut = self._make_future(req_id)
        await self._sock.send_multipart([T_ALLOC, payload])
        try:
            result = await asyncio.wait_for(fut, timeout=self.alloc_timeout_s)
        except asyncio.TimeoutError:
            # MMP unresponsive / pool stalled — a capacity condition, not an
            # inference timeout (504).
            self._drop_future(req_id)
            raise ServerBusyError("slot allocation timed out") from None
        except asyncio.CancelledError:
            # Handler task cancelled (client disconnect) while T_ALLOC may be in
            # flight. Drop the future; if MMP did allocate, the slot has no
            # submit ticket and the reaper reclaims it (ownership-checked).
            self._drop_future(req_id)
            raise
        if result[0] == "error":
            # Alloc-time T_ERROR is pool-full / per-model cap — retryable.
            raise ServerBusyError(f"no capacity (code={result[1]})")
        return result[1]

    async def _submit_and_wait(
        self,
        req_id: int,
        slot_id: int,
        model_id: str,
        instance: str,
        input_sz: int,
        params: dict,
        request: Optional[Request] = None,
    ) -> tuple:
        mid = _routing_key(model_id, instance).encode()
        header = struct.pack(">QIIH", req_id, slot_id, input_sz, len(mid)) + mid
        params_json = json.dumps(params).encode() if params else b"{}"
        fut = self._make_future(req_id)
        await self._sock.send_multipart([T_SUBMIT, header, params_json])
        try:
            if request is not None:
                return await self._wait_with_disconnect(fut, request)
            return await asyncio.wait_for(fut, timeout=self.infer_timeout_s)
        except asyncio.TimeoutError:
            self._drop_future(req_id)
            raise
        except ClientDisconnected:
            self._drop_future(req_id)
            raise
        except asyncio.CancelledError:
            self._drop_future(req_id)
            raise

    async def _wait_with_disconnect(
        self, fut: asyncio.Future, request: Request
    ) -> tuple:
        async def _poll() -> bool:
            while True:
                if await request.is_disconnected():
                    return True
                await asyncio.sleep(0.5)

        disc_task = asyncio.ensure_future(_poll())
        infer_task = asyncio.ensure_future(
            asyncio.wait_for(fut, timeout=self.infer_timeout_s)
        )
        done, pending_tasks = await asyncio.wait(
            [disc_task, infer_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending_tasks:
            t.cancel()
        if infer_task in done:
            # Result (or its timeout) won — or tied with disconnect. A resolved
            # result means the worker is finished and the slot is readable; do
            # not discard it just because the client also went away.
            return infer_task.result()
        raise ClientDisconnected()

    def _spawn_bg(self, coro) -> None:
        """create_task with a strong reference — fire-and-forget tasks can be
        GC'd mid-flight otherwise, silently dropping the T_FREE/T_CANCEL."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _free_slot(self, slot_id: int, req_id: int) -> None:
        async def _send() -> None:
            try:
                await self._sock.send_multipart(
                    [T_FREE, struct.pack(">QI", req_id, slot_id)]
                )
            except Exception:
                logger.warning("free_slot: failed for slot %d", slot_id, exc_info=True)

        self._spawn_bg(_send())

    def _cancel_req(self, req_id: int) -> None:
        """Tell MMP we gave up on req_id. MMP frees the slot once the worker
        drains the ticket (or the reaper reclaims it) — never reuses it early."""
        self._drop_future(req_id)

        async def _send() -> None:
            try:
                await self._sock.send_multipart([T_CANCEL, struct.pack(">Q", req_id)])
            except Exception:
                logger.warning("cancel_req: failed for req %d", req_id, exc_info=True)

        self._spawn_bg(_send())

    def _write_input(
        self, slot_id: int, chunk: bytes | memoryview, offset: int
    ) -> None:
        base = slot_id * self.slot_total + HEADER_SIZE
        end = base + offset + len(chunk)
        if base < 0 or end > len(self._shm.buf):
            raise ValueError(
                f"SHM write out of bounds: slot={slot_id} "
                f"offset={offset} len={len(chunk)}"
            )
        self._shm.buf[base + offset : end] = chunk

    def _read_result(self, slot_id: int, result_sz: int) -> bytes:
        if result_sz > self.shm_data_size or result_sz < 0:
            raise ValueError(
                f"result_sz {result_sz} out of bounds " f"(max {self.shm_data_size})"
            )
        if slot_id < 0 or slot_id * self.slot_total + HEADER_SIZE + result_sz > len(
            self._shm.buf
        ):
            raise ValueError(f"slot_id {slot_id} out of bounds")
        base = slot_id * self.slot_total + HEADER_SIZE
        return bytes(self._shm.buf[base : base + result_sz])

    def _read_slot_header(self, slot_id: int) -> Optional[_SlotHeaderView]:
        off = slot_id * self.slot_total
        if off + HEADER_SIZE > len(self._shm.buf):
            return None
        status = self._shm.buf[off + _OFF_STATUS]
        result_size = struct.unpack_from("<I", self._shm.buf, off + _OFF_RESULT_SZ)[0]
        return _SlotHeaderView(status, result_size)

    async def _lifecycle_req(
        self, msg_type: bytes, payload: bytes, timeout_s: float = 30.0
    ) -> tuple:
        req_id = _new_req_id()
        fut = self._make_future(req_id)
        full_payload = struct.pack(">Q", req_id) + payload
        await self._sock.send_multipart([msg_type, full_payload])
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._drop_future(req_id)
            raise
        except asyncio.CancelledError:
            self._drop_future(req_id)
            raise


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _new_req_id() -> int:
    return uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF


def _routing_key(model_id: str, instance: str) -> str:
    return f"{model_id}:{instance}" if instance else model_id
