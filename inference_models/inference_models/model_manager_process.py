"""ModelManagerProcess (MMP) — orchestrated mode hub.

Phase 3 (hot path): ZMQ ROUTER slot routing between uvicorn workers and backends.
Phase 4 (cold path): Optional ModelManager embedding, auto-load via executor,
LRU eviction, GPU/CPU monitoring, lifecycle control messages.

Hot-path protocol (uvicorn ↔ MMP, ZMQ DEALER/ROUTER, no empty delimiter):
  uvicorn  →  T_ENSURE_LOADED  →  MMP  →  T_MODEL_READY
  uvicorn  →  T_ALLOC          →  MMP  →  T_ALLOC_OK
  uvicorn writes SHM slot directly
  uvicorn  →  T_SUBMIT         →  MMP  →  backend.signal_slot(slot_id, req_id)
  backend worker finishes, calls mmp.on_result(req_id, slot_id, result_sz)
  MMP      →  T_RESULT_READY   →  uvicorn (by stored identity)
  uvicorn reads SHM slot, sends T_FREE  →  MMP frees slot

Lifecycle API (admin ↔ MMP, same ROUTER):
  T_LOAD / T_UNLOAD / T_SLEEP / T_WAKE / T_STATS

Run standalone::

    python -m inference_models.model_manager_process \\
        --n-slots 256 --input-mb 20 --result-mb 4

or embed (Phase 4 with real loading)::

    from inference_models.model_manager import ModelManager
    mmp = ModelManagerProcess(n_slots=256, input_mb=20, result_mb=4,
                              manager=ModelManager())
    ready = threading.Event()
    t = threading.Thread(target=lambda: asyncio.run(mmp.run(ready_event=ready)))
    t.start()
    ready.wait()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import zmq
import zmq.asyncio

from inference_models.backends.utils.shm_pool import SHMPool
from inference_models.backends.utils.transport import zmq_addr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZMQ message type bytes — must match app.py exactly
# ---------------------------------------------------------------------------

# uvicorn → MMP (hot path)
T_ENSURE_LOADED = b"\x09"
T_ALLOC         = b"\x01"
T_SUBMIT        = b"\x02"
T_FREE          = b"\x03"

# MMP → uvicorn (hot path)
T_MODEL_READY   = b"\x0A"
T_LOAD_TIMEOUT  = b"\x0B"
T_ALLOC_OK      = b"\x11"
T_RESULT_READY  = b"\x14"
T_ERROR         = b"\xFF"

# Lifecycle API (admin → MMP)
T_LOAD   = b"\x20"
T_UNLOAD = b"\x21"
T_SLEEP  = b"\x22"
T_WAKE   = b"\x23"
T_STATS  = b"\x30"

# Lifecycle replies (MMP → admin)
T_OK         = b"\x40"
T_STATS_RESP = b"\x41"

# Error codes embedded in T_ERROR payload byte
_ERR_POOL_FULL   = 1
_ERR_NO_BACKEND  = 2
_ERR_STALE       = 3
_ERR_BACKEND     = 4
_ERR_LOAD_FAILED = 5
_ERR_NOT_LOADED  = 6


# ---------------------------------------------------------------------------
# GPU memory helper (module level — no dependency on MMP instance)
# ---------------------------------------------------------------------------

def _gpu_used_fraction() -> float:
    """Fraction of GPU 0 memory in use (0.0–1.0). Returns 0.0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            if total > 0:
                used = (torch.cuda.memory_allocated(0)
                        + torch.cuda.memory_reserved(0))
                return used / total
    except Exception:
        pass
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.total > 0:
            return info.used / info.total
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# FlavorState — per model flavor lifecycle
# ---------------------------------------------------------------------------

@dataclass
class FlavorState:
    """Tracks load lifecycle for one model flavor."""
    loaded:   bool = False
    loading:  bool = False
    sleeping: bool = False   # in ModelManager but VRAM evicted
    # Each waiter: (uvicorn_identity, req_id, deadline_monotonic_s)
    load_waiters: list[tuple[bytes, int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BackendLike protocol
# ---------------------------------------------------------------------------

class BackendLike(Protocol):
    def signal_slot(self, slot_id: int, req_id: int) -> None: ...


# ---------------------------------------------------------------------------
# ModelManagerProcess
# ---------------------------------------------------------------------------

class ModelManagerProcess:
    """ZMQ ROUTER hub — Phase 3 hot path + Phase 4 cold path.

    Thread-safety:
        ``on_result()`` and ``register_backend()`` are the only methods
        called from external threads.  All other methods run on the asyncio
        event loop owned by ``run()``.
    """

    def __init__(
        self,
        n_slots:                int   = 256,
        input_mb:               float = 20.0,
        result_mb:              float = 4.0,
        stale_reap_interval_s:  float = 10.0,
        stale_slot_max_age_s:   float = 30.0,
        evict_threshold:        float = 0.9,
        evict_check_interval_s: float = 5.0,
        monitor_interval_s:     float = 5.0,
        default_api_key:        str   = "",
        manager: Optional[Any]        = None,
    ) -> None:
        """
        Args:
            manager: Optional ModelManager (or duck-type compatible) for real
                model loading (Phase 4).  ``None`` → stub load mode: flavors
                are marked loaded immediately without a real model, so
                T_ENSURE_LOADED waiters get T_MODEL_READY but T_SUBMIT returns
                T_ERROR (no backend).  Useful for tests and hot-path benchmarks.
        """
        self._n_slots               = n_slots
        self._input_mb              = input_mb
        self._result_mb             = result_mb
        self._stale_reap_interval_s = stale_reap_interval_s
        self._stale_slot_max_age_s  = stale_slot_max_age_s
        self._evict_threshold       = evict_threshold
        self._evict_check_interval_s = evict_check_interval_s
        self._monitor_interval_s    = monitor_interval_s
        self._default_api_key       = (
            default_api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        )
        self._manager               = manager
        self._own_manager           = False  # set True only if we created it

        self._pool:   Optional[SHMPool]                   = None
        self._router: Optional[zmq.asyncio.Socket]        = None
        self._loop:   Optional[asyncio.AbstractEventLoop] = None

        # req_id → (uvicorn_identity, slot_id, flavor)
        self._pending: dict[int, tuple[bytes, int, str]] = {}

        # flavor → FlavorState
        self._flavors: dict[str, FlavorState] = {}

        # flavor → BackendLike (registered by register_backend or _load_flavor)
        self._backends: dict[str, BackendLike] = {}

        # flavor → monotonic timestamp of last T_SUBMIT (LRU eviction)
        self._flavor_access: dict[str, float] = {}

        # Latest monitoring snapshot (updated by _monitoring_loop)
        self._stats_snapshot: dict[str, Any] = {}

        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def shm_name(self) -> Optional[str]:
        """SHM pool name — valid after run() has started."""
        return self._pool.name if self._pool else None

    def register_backend(self, flavor: str, backend: BackendLike) -> None:
        """Mark flavor loaded and register its backend.

        Called by ModelManager (or tests) after a model is ready.
        Thread-safe: may be called from any thread before or during run().
        """
        self._backends[flavor] = backend
        fs = self._flavors.setdefault(flavor, FlavorState())
        fs.loading  = False
        fs.loaded   = True
        fs.sleeping = False

        if self._loop is not None and fs.load_waiters:
            self._loop.call_soon_threadsafe(self._flush_load_waiters, flavor)

    def on_result(self, req_id: int, slot_id: int, result_sz: int) -> None:
        """Called by SubprocessBackend recv thread when a slot completes.

        Thread-safe. ``result_sz == 0`` signals inference error.
        """
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._on_result_on_loop, req_id, slot_id, result_sz
            )

    # ------------------------------------------------------------------
    # Main async entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        addr:        Optional[str]             = None,
        ready_event: Optional[threading.Event] = None,
    ) -> None:
        """Bind ROUTER, create SHM pool, dispatch messages until stopped."""
        self._loop    = asyncio.get_running_loop()
        self._running = True

        # Create SHM pool
        self._pool = SHMPool.create(self._n_slots, self._input_mb, self._result_mb)
        logger.info(
            "MMP: SHM pool ready  name=%s  slots=%d  input=%.0fMB  result=%.0fMB",
            self._pool.name, self._n_slots, self._input_mb, self._result_mb,
        )

        # Bind ROUTER
        ctx = zmq.asyncio.Context()
        self._router = ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._router.setsockopt(zmq.SNDHWM, 0)
        self._router.setsockopt(zmq.RCVHWM, 0)
        self._router.setsockopt(zmq.LINGER, 0)
        bind_addr = addr or zmq_addr("mmprocess")
        self._router.bind(bind_addr)
        logger.info("MMP: ROUTER bound on %s", bind_addr)

        try:
            self._loop.add_signal_handler(signal.SIGTERM, self._on_sigterm)
        except (NotImplementedError, RuntimeError):
            pass

        self._stop_event: asyncio.Event = asyncio.Event()

        if ready_event is not None:
            ready_event.set()

        tasks = [
            asyncio.create_task(self._recv_loop(),         name="mmp-recv"),
            asyncio.create_task(self._stale_reaper_loop(), name="mmp-stale-reaper"),
            asyncio.create_task(self._eviction_loop(),     name="mmp-evictor"),
            asyncio.create_task(self._monitoring_loop(),   name="mmp-monitor"),
        ]

        try:
            await self._stop_event.wait()
        finally:
            for t in tasks:
                t.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass
            await self._drain_pending()
            self._router.close()
            ctx.term()
            self._pool.close()
            self._pool   = None
            self._router = None
            if self._own_manager and self._manager is not None:
                self._manager.shutdown()
            logger.info("MMP: shut down")

    def stop(self) -> None:
        """Request graceful shutdown. Thread-safe."""
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._set_stop_event)

    def _set_stop_event(self) -> None:
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    # ------------------------------------------------------------------
    # Recv loop
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """Receive ZMQ frames and dispatch. Runs as asyncio task."""
        while True:
            try:
                frames = await self._router.recv_multipart()
                await self._dispatch(frames)
            except asyncio.CancelledError:
                break
            except zmq.ZMQError as exc:
                logger.warning("MMP: recv error: %s", exc)
                break

    # ------------------------------------------------------------------
    # SIGTERM + drain
    # ------------------------------------------------------------------

    def _on_sigterm(self) -> None:
        logger.info("MMP: SIGTERM — shutting down")
        self._running = False
        self._set_stop_event()

    async def _drain_pending(self) -> None:
        """Send T_ERROR to all in-flight requests on shutdown."""
        for req_id, (identity, slot_id, _) in list(self._pending.items()):
            try:
                await self._send(identity, T_ERROR, struct.pack(">QB", req_id, 0))
            except Exception:
                pass
            try:
                self._pool.free_slot(slot_id)
            except Exception:
                pass
        self._pending.clear()

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, frames: list[bytes]) -> None:
        # DEALER/ROUTER — no empty delimiter: [identity, msg_type, *data]
        if len(frames) < 2:
            return
        identity = frames[0]
        msg_type = frames[1]
        data     = frames[2:]

        try:
            if msg_type == T_ENSURE_LOADED:
                await self._handle_ensure_loaded(identity, data)
            elif msg_type == T_ALLOC:
                await self._handle_alloc(identity, data)
            elif msg_type == T_SUBMIT:
                await self._handle_submit(data)
            elif msg_type == T_FREE:
                self._handle_free(data)
            elif msg_type == T_LOAD:
                await self._handle_load(identity, data)
            elif msg_type == T_UNLOAD:
                await self._handle_unload(identity, data)
            elif msg_type == T_SLEEP:
                await self._handle_sleep(identity, data)
            elif msg_type == T_WAKE:
                await self._handle_wake(identity, data)
            elif msg_type == T_STATS:
                await self._handle_stats(identity, data)
            else:
                logger.debug(
                    "MMP: unknown msg_type=%r from %r", msg_type, identity
                )
        except Exception:
            logger.exception("MMP: unhandled error dispatching %r", msg_type)

    # ------------------------------------------------------------------
    # T_ENSURE_LOADED  →  T_MODEL_READY | T_LOAD_TIMEOUT
    # wire: Q I H N   req_id(8) wait_ms(4) flavor_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_ensure_loaded(
        self, identity: bytes, data: list[bytes]
    ) -> None:
        if not data or len(data[0]) < 14:
            return
        frame = data[0]
        req_id, wait_ms, flavor_len = struct.unpack_from(">QIH", frame)
        flavor   = frame[14: 14 + flavor_len].decode(errors="replace")
        deadline = time.monotonic() + wait_ms / 1000.0

        fs = self._flavors.get(flavor)
        if fs is not None and fs.loaded:
            await self._send(identity, T_MODEL_READY, struct.pack(">Q", req_id))
            return

        if fs is None:
            fs = FlavorState()
            self._flavors[flavor] = fs

        fs.load_waiters.append((identity, req_id, deadline))

        if not fs.loading:
            fs.loading = True
            asyncio.create_task(
                self._load_flavor(flavor), name=f"mmp-load-{flavor}"
            )

    # ------------------------------------------------------------------
    # T_ALLOC  →  T_ALLOC_OK | T_ERROR
    # wire: Q H N   req_id(8) flavor_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_alloc(
        self, identity: bytes, data: list[bytes]
    ) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, flavor_len = struct.unpack_from(">QH", frame)
        flavor = frame[10: 10 + flavor_len].decode(errors="replace")

        try:
            slot_id = self._pool.alloc_slot(timeout=0)
        except TimeoutError:
            await self._send(identity, T_ERROR,
                             struct.pack(">QB", req_id, _ERR_POOL_FULL))
            return

        self._pool.mark_allocated(slot_id, req_id)
        self._pending[req_id] = (identity, slot_id, flavor)
        await self._send(identity, T_ALLOC_OK, struct.pack(">QI", req_id, slot_id))

    # ------------------------------------------------------------------
    # T_SUBMIT  (no reply — result delivered via on_result callback)
    # wire: Q I I H N   req_id(8) slot_id(4) input_sz(4) flavor_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_submit(self, data: list[bytes]) -> None:
        if not data or len(data[0]) < 18:
            return
        frame = data[0]
        req_id, slot_id, input_sz, flavor_len = struct.unpack_from(">QIIH", frame)
        flavor = frame[18: 18 + flavor_len].decode(errors="replace")

        self._pool.mark_written(slot_id, input_sz)
        self._forward_to_backend(flavor, slot_id, req_id)

    # ------------------------------------------------------------------
    # T_FREE
    # wire: I   slot_id(4)
    # ------------------------------------------------------------------

    def _handle_free(self, data: list[bytes]) -> None:
        if not data or len(data[0]) < 4:
            return
        slot_id = struct.unpack_from(">I", data[0])[0]
        try:
            self._pool.free_slot(slot_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # T_LOAD (admin) — trigger model load + reply T_OK immediately
    # wire: Q H N H N   req_id(8) flavor_len(2) flavor(N) api_key_len(2) api_key(M)
    # ------------------------------------------------------------------

    async def _handle_load(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, flavor_len = struct.unpack_from(">QH", frame)
        off    = 10
        flavor = frame[off: off + flavor_len].decode(errors="replace")
        off   += flavor_len
        api_key = ""
        if len(frame) >= off + 2:
            klen    = struct.unpack_from(">H", frame, off)[0]
            api_key = frame[off + 2: off + 2 + klen].decode(errors="replace")

        fs = self._flavors.setdefault(flavor, FlavorState())
        if not fs.loaded and not fs.loading:
            fs.loading = True
            asyncio.create_task(
                self._load_flavor(flavor, api_key=api_key),
                name=f"mmp-load-{flavor}",
            )
        await self._send(identity, T_OK, struct.pack(">Q", req_id))

    # ------------------------------------------------------------------
    # T_UNLOAD (admin)
    # wire: Q H N   req_id(8) flavor_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_unload(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, flavor_len = struct.unpack_from(">QH", frame)
        flavor = frame[10: 10 + flavor_len].decode(errors="replace")

        loop = asyncio.get_running_loop()
        try:
            if self._manager is not None:
                await loop.run_in_executor(
                    None, lambda: self._manager.unload(flavor)
                )
            fs = self._flavors.get(flavor)
            if fs:
                fs.loaded   = False
                fs.sleeping = False
                fs.loading  = False
            self._backends.pop(flavor, None)
            await self._send(identity, T_OK, struct.pack(">Q", req_id))
        except Exception:
            logger.exception("MMP: T_UNLOAD '%s' failed", flavor)
            await self._send(identity, T_ERROR,
                             struct.pack(">QB", req_id, _ERR_NOT_LOADED))

    # ------------------------------------------------------------------
    # T_SLEEP (admin) — offload VRAM, keep weights on CPU
    # wire: Q H N   req_id(8) flavor_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_sleep(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, flavor_len = struct.unpack_from(">QH", frame)
        flavor = frame[10: 10 + flavor_len].decode(errors="replace")

        loop = asyncio.get_running_loop()
        try:
            if self._manager is not None:
                await loop.run_in_executor(
                    None, lambda: self._manager.sleep(flavor)
                )
            fs = self._flavors.get(flavor)
            if fs:
                fs.loaded   = False
                fs.sleeping = True
            await self._send(identity, T_OK, struct.pack(">Q", req_id))
        except Exception:
            logger.exception("MMP: T_SLEEP '%s' failed", flavor)
            await self._send(identity, T_ERROR,
                             struct.pack(">QB", req_id, _ERR_NOT_LOADED))

    # ------------------------------------------------------------------
    # T_WAKE (admin) — restore weights to VRAM
    # wire: Q H N   req_id(8) flavor_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_wake(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, flavor_len = struct.unpack_from(">QH", frame)
        flavor = frame[10: 10 + flavor_len].decode(errors="replace")

        loop = asyncio.get_running_loop()
        try:
            if self._manager is not None:
                await loop.run_in_executor(
                    None, lambda: self._manager.wake(flavor)
                )
            fs = self._flavors.get(flavor)
            if fs:
                fs.loaded   = True
                fs.sleeping = False
            await self._send(identity, T_OK, struct.pack(">Q", req_id))
        except Exception:
            logger.exception("MMP: T_WAKE '%s' failed", flavor)
            await self._send(identity, T_ERROR,
                             struct.pack(">QB", req_id, _ERR_NOT_LOADED))

    # ------------------------------------------------------------------
    # T_STATS — snapshot reply
    # wire: Q   req_id(8)
    # reply: T_STATS_RESP  Q I N   req_id(8) json_len(4) json(N)
    # ------------------------------------------------------------------

    async def _handle_stats(self, identity: bytes, data: list[bytes]) -> None:
        req_id = (
            struct.unpack_from(">Q", data[0])[0]
            if (data and len(data[0]) >= 8) else 0
        )
        # Start from cached monitoring snapshot (includes psutil/GPU from last poll)
        snapshot: dict[str, Any] = dict(self._stats_snapshot)
        # Overlay with fresh manager stats and live MMP state
        if self._manager is not None:
            try:
                snapshot.update(self._manager.stats())
            except Exception:
                pass
        snapshot.update({
            "mmp_free_slots":  self._pool.free_count if self._pool else 0,
            "mmp_total_slots": self._n_slots,
            "mmp_pending":     len(self._pending),
            "mmp_flavors": {
                f: {
                    "loaded":   fs.loaded,
                    "sleeping": fs.sleeping,
                    "loading":  fs.loading,
                }
                for f, fs in self._flavors.items()
            },
        })
        payload = json.dumps(snapshot, default=str).encode()
        await self._send(
            identity, T_STATS_RESP,
            struct.pack(">QI", req_id, len(payload)) + payload,
        )

    # ------------------------------------------------------------------
    # Backend routing
    # ------------------------------------------------------------------

    def _forward_to_backend(
        self, flavor: str, slot_id: int, req_id: int
    ) -> None:
        self._flavor_access[flavor] = time.monotonic()   # LRU update
        backend = self._backends.get(flavor)
        if backend is None:
            logger.warning("MMP: no backend for '%s', req_id=%d", flavor, req_id)
            self._on_result_on_loop(req_id, slot_id, 0)
            return
        try:
            backend.signal_slot(slot_id, req_id)
        except Exception:
            logger.exception("MMP: signal_slot raised for '%s'", flavor)
            self._on_result_on_loop(req_id, slot_id, 0)

    # ------------------------------------------------------------------
    # on_result — event-loop side
    # ------------------------------------------------------------------

    def _on_result_on_loop(
        self, req_id: int, slot_id: int, result_sz: int
    ) -> None:
        """Must be called on the event loop thread."""
        pending = self._pending.pop(req_id, None)
        if pending is None:
            return
        identity, _, _ = pending
        asyncio.create_task(
            self._reply_result(identity, req_id, slot_id, result_sz),
            name=f"mmp-reply-{req_id}",
        )

    async def _reply_result(
        self,
        identity:  bytes,
        req_id:    int,
        slot_id:   int,
        result_sz: int,
    ) -> None:
        if result_sz == 0:
            await self._send(identity, T_ERROR,
                             struct.pack(">QB", req_id, _ERR_BACKEND))
            self._pool.free_slot(slot_id)
            return
        self._pool.mark_done(slot_id, result_sz)
        sent = await self._send(
            identity, T_RESULT_READY,
            struct.pack(">QII", req_id, slot_id, result_sz),
        )
        if not sent:
            # Peer disconnected — result will never be read; free slot now
            # (backend already finished; we just drop the result silently)
            logger.debug(
                "MMP: peer gone for req_id=%d slot=%d — freeing slot",
                req_id, slot_id,
            )
            self._pool.free_slot(slot_id)

    # ------------------------------------------------------------------
    # Cold path — load / wake
    # ------------------------------------------------------------------

    async def _load_flavor(self, flavor: str, api_key: str = "") -> None:
        """Load flavor via ModelManager, falling back to stub on failure/no manager.

        Stub mode: marks flavor loaded immediately (no real model).
        T_ENSURE_LOADED waiters get T_MODEL_READY, but T_SUBMIT will return
        T_ERROR because no backend is registered.
        """
        loop = asyncio.get_running_loop()
        fs   = self._flavors.get(flavor)

        # Sleeping → wake instead of reload
        if fs is not None and fs.sleeping:
            await self._wake_flavor(flavor)
            return

        if self._manager is not None:
            effective_key = api_key or self._default_api_key
            logger.info("MMP: loading '%s' via ModelManager", flavor)
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self._manager.load(
                        flavor, effective_key, backend="subprocess"
                    ),
                )
                backend = self._manager.get_backend(flavor)
                if backend is not None and hasattr(backend, "signal_slot"):
                    # register_backend marks loaded + flushes waiters
                    self.register_backend(flavor, backend)
                    return
                # Loaded but no signal_slot yet (Phase 6 will add it)
                # Fall through to stub-mark-loaded below so waiters unblock.
            except Exception:
                logger.info(
                    "MMP: manager.load('%s') raised — stub-loading without backend",
                    flavor,
                )

        # Stub: mark loaded, flush waiters; no backend registered
        logger.info("MMP: '%s' stub-loaded (no real model)", flavor)
        if fs is None:
            fs = self._flavors.setdefault(flavor, FlavorState())
        fs.loading = False
        fs.loaded  = True
        self._flush_load_waiters(flavor)

    async def _wake_flavor(self, flavor: str) -> None:
        """Wake a sleeping flavor via ModelManager."""
        loop = asyncio.get_running_loop()
        logger.info("MMP: waking '%s'", flavor)
        try:
            if self._manager is not None:
                await loop.run_in_executor(
                    None, lambda: self._manager.wake(flavor)
                )
            fs = self._flavors.get(flavor)
            if fs:
                fs.sleeping = False
                fs.loading  = False
                fs.loaded   = True
                # Re-register backend if signal_slot is now available
                if self._manager is not None:
                    backend = self._manager.get_backend(flavor)
                    if backend is not None and hasattr(backend, "signal_slot"):
                        self._backends[flavor] = backend
                self._flush_load_waiters(flavor)
        except Exception:
            logger.exception("MMP: failed to wake '%s'", flavor)
            fs = self._flavors.get(flavor)
            if fs:
                waiters, fs.load_waiters = fs.load_waiters, []
                fs.loading = False
                for identity, req_id, _ in waiters:
                    asyncio.create_task(
                        self._send(identity, T_ERROR,
                                   struct.pack(">QB", req_id, _ERR_LOAD_FAILED))
                    )

    def _flush_load_waiters(self, flavor: str) -> None:
        """Notify all T_ENSURE_LOADED waiters for this flavor.

        Must be called on the event loop thread.
        """
        fs = self._flavors.get(flavor)
        if not fs:
            return
        waiters, fs.load_waiters = fs.load_waiters, []
        now = time.monotonic()
        for identity, req_id, deadline in waiters:
            if now <= deadline:
                asyncio.create_task(
                    self._send(identity, T_MODEL_READY,
                               struct.pack(">Q", req_id))
                )
            else:
                asyncio.create_task(
                    self._send(identity, T_LOAD_TIMEOUT,
                               struct.pack(">QI", req_id, 1))
                )

    # ------------------------------------------------------------------
    # Stale slot reaper
    # ------------------------------------------------------------------

    async def _stale_reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self._stale_reap_interval_s)
            try:
                self._reap()
            except Exception:
                logger.exception("MMP: error in stale reaper")

    def _reap(self) -> None:
        stale = self._pool.stale_slots(self._stale_slot_max_age_s)
        if not stale:
            return
        now_ns        = time.monotonic_ns()
        slot_to_req   = {s: r for r, (_, s, _) in self._pending.items()}
        for slot_id in stale:
            req_id = slot_to_req.get(slot_id)
            try:
                hdr   = self._pool.read_header(slot_id)
                age_s = (now_ns - hdr.timestamp_ns) / 1e9 if hdr.timestamp_ns else 0.0
            except Exception:
                age_s = 0.0
            logger.warning(
                "MMP: reaping stale slot slot_id=%d age_s=%.1f req_id=%s",
                slot_id, age_s, req_id,
            )
            if req_id is not None:
                pending = self._pending.pop(req_id, None)
                if pending:
                    identity, _, _ = pending
                    asyncio.create_task(
                        self._send(identity, T_ERROR,
                                   struct.pack(">QB", req_id, _ERR_STALE))
                    )
            self._pool.free_slot(slot_id)

    # ------------------------------------------------------------------
    # Eviction loop
    # ------------------------------------------------------------------

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(self._evict_check_interval_s)
            try:
                self._check_and_evict()
            except Exception:
                logger.exception("MMP: error in eviction loop")

    def _check_and_evict(self) -> None:
        """Evict LRU flavor if GPU memory exceeds threshold."""
        if self._manager is None:
            return
        gpu_frac = _gpu_used_fraction()
        if gpu_frac < self._evict_threshold:
            return
        lru = self._lru_evictable_flavor()
        if lru is None:
            return
        logger.warning(
            "MMP: GPU %.0f%% > %.0f%% threshold — evicting '%s'",
            gpu_frac * 100, self._evict_threshold * 100, lru,
        )
        try:
            self._manager.sleep(lru)
            fs = self._flavors.get(lru)
            if fs:
                fs.loaded   = False
                fs.sleeping = True
        except Exception:
            logger.warning("MMP: eviction of '%s' failed", lru, exc_info=True)

    def _lru_evictable_flavor(self) -> Optional[str]:
        """LRU flavor that is loaded, not sleeping, and not currently in-flight."""
        in_flight = {flavor for _, _, flavor in self._pending.values()}
        candidates = [
            f for f, fs in self._flavors.items()
            if fs.loaded and not fs.sleeping and f not in in_flight
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda f: self._flavor_access.get(f, 0.0))

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    async def _monitoring_loop(self) -> None:
        while True:
            await asyncio.sleep(self._monitor_interval_s)
            try:
                loop = asyncio.get_running_loop()
                self._stats_snapshot = await loop.run_in_executor(
                    None, self._collect_stats
                )
            except Exception:
                logger.exception("MMP: monitoring loop error")

    def _collect_stats(self) -> dict:
        """Collect system + manager metrics. Runs in executor (may block)."""
        s: dict[str, Any] = {"timestamp_s": time.time()}
        if self._manager is not None:
            try:
                s.update(self._manager.stats())
            except Exception:
                pass
        try:
            import psutil  # type: ignore[import]
            proc = psutil.Process()
            s["process_cpu_pct"] = proc.cpu_percent()
            s["process_rss_mb"]  = proc.memory_info().rss / 1024 / 1024
            s["system_cpu_pct"]  = psutil.cpu_percent()
            s["system_ram_pct"]  = psutil.virtual_memory().percent
        except Exception:
            pass
        s["gpu_used_fraction"] = _gpu_used_fraction()
        return s

    # ------------------------------------------------------------------
    # ZMQ send helper
    # ------------------------------------------------------------------

    async def _send(
        self, identity: bytes, msg_type: bytes, payload: bytes
    ) -> bool:
        """Send a message. Returns True on success, False on ZMQ error."""
        try:
            await self._router.send_multipart([identity, msg_type, payload])
            return True
        except zmq.ZMQError as exc:
            logger.warning("MMP: send failed to %r: %s", identity, exc)
            return False


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ModelManagerProcess")
    parser.add_argument(
        "--n-slots", type=int,
        default=int(os.environ.get("MMP_N_SLOTS", "256")),
    )
    parser.add_argument(
        "--input-mb", type=float,
        default=float(os.environ.get("MMP_INPUT_MB", "20.0")),
    )
    parser.add_argument(
        "--result-mb", type=float,
        default=float(os.environ.get("MMP_RESULT_MB", "4.0")),
    )
    parser.add_argument("--addr", default=None,
                        help="ZMQ bind address (default: platform auto)")
    parser.add_argument("--api-key",
                        default=os.environ.get("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--evict-threshold", type=float, default=0.9)
    args = parser.parse_args()

    from inference_models.model_manager import ModelManager

    mmp = ModelManagerProcess(
        n_slots=args.n_slots,
        input_mb=args.input_mb,
        result_mb=args.result_mb,
        evict_threshold=args.evict_threshold,
        default_api_key=args.api_key,
        manager=ModelManager(),
    )
    asyncio.run(mmp.run(addr=args.addr))


if __name__ == "__main__":
    main()
