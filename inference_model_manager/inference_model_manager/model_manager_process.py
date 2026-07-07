"""ModelManagerProcess (MMP) — orchestrated mode hub.

hot path: ZMQ ROUTER slot routing between uvicorn workers and backends.
cold path: Optional ModelManager embedding, auto-load via executor,
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
  T_LOAD / T_UNLOAD / T_STATS

Run standalone::

    python -m inference_models.model_manager_process \\
        --n-slots 256 --input-mb 20

or embed::

    mmp = ModelManagerProcess(n_slots=256, input_mb=20)
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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import zmq
import zmq.asyncio

from inference_model_manager import configuration as cfg
from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus
from inference_model_manager.backends.utils.transport import zmq_addr
from inference_model_manager.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZMQ message type bytes — must match app.py exactly
# ---------------------------------------------------------------------------

# uvicorn → MMP (hot path)
T_ENSURE_LOADED = b"\x09"
T_ALLOC = b"\x01"
T_SUBMIT = b"\x02"
T_FREE = b"\x03"
T_CANCEL = b"\x04"

# MMP → uvicorn (hot path)
T_MODEL_READY = b"\x0a"
T_LOAD_TIMEOUT = b"\x0b"
T_ALLOC_OK = b"\x11"
T_RESULT_READY = b"\x14"
T_ERROR = b"\xff"

# Lifecycle API (admin → MMP)
T_LOAD = b"\x20"
T_UNLOAD = b"\x21"
T_STATS = b"\x30"
T_SHM_INFO = b"\x31"

# Lifecycle replies (MMP → admin)
T_OK = b"\x40"
T_STATS_RESP = b"\x41"
T_SHM_INFO_RESP = b"\x42"

# Error codes embedded in T_ERROR payload byte
_ERR_POOL_FULL = 1
_ERR_NO_BACKEND = 2
_ERR_STALE = 3
_ERR_BACKEND = 4
_ERR_LOAD_FAILED = 5
_ERR_NOT_LOADED = 6
_ERR_SERVER_FULL = 7


# ---------------------------------------------------------------------------
# GPU memory helper (module level — no dependency on MMP instance)
# ---------------------------------------------------------------------------


# Set once NVML is found unavailable (no libnvidia-ml.so.1 / no NVIDIA GPU, e.g.
# Jetson-Tegra or CPU) so the telemetry loop stops retrying + spamming tracebacks.
_NVML_DISABLED = False

_nvml_handle = None


def _nvml_mem_info():
    """Device-level GPU 0 memory via NVML, or None if unavailable.

    Deliberately NOT torch: torch allocator stats are process-local and MMP
    holds no models in orchestrated mode (always ~0), and torch.cuda calls
    would create a CUDA context inside MMP costing hundreds of MB of the very
    VRAM being budgeted. NVML handle cached after first init.
    """
    global _NVML_DISABLED, _nvml_handle
    if _NVML_DISABLED:
        return None
    try:
        import pynvml  # nvidia-ml-py (drop-in; install nvidia-ml-py not pynvml)

        if _nvml_handle is None:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        _NVML_DISABLED = True
        return None
    try:
        import pynvml

        return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
    except Exception:
        return None


def _gpu_used_fraction() -> float:
    """Fraction of GPU 0 memory in use (0.0-1.0), device-level via NVML.
    Returns 0.0 if unavailable (eviction loop then never triggers)."""
    info = _nvml_mem_info()
    if info is not None and info.total > 0:
        return info.used / info.total
    return 0.0


def _collect_gpu_stats(
    pid_to_flavor: Optional[dict] = None,
) -> dict:
    """Collect per-GPU hardware stats and per-model GPU memory.

    Uses nvidia-ml-py (``import pynvml``).  Returns empty dict on any failure
    (no NVIDIA GPU, driver not loaded, etc.).

    Args:
        pid_to_flavor: mapping of worker PID → model flavor name.
            When provided, per-process GPU memory is attributed to flavors via
            ``nvmlDeviceGetComputeRunningProcesses``.

    Returns dict with keys:
        ``gpus``                  — list of per-GPU dicts (index, memory, util, temp, power)
        ``per_model_gpu_mb``      — dict flavor → GPU memory MB (only populated if pid_to_flavor given)
    """
    global _NVML_DISABLED
    result: dict = {"gpus": [], "per_model_gpu_mb": {}}
    if _NVML_DISABLED:
        return result
    try:
        import pynvml  # nvidia-ml-py

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pid_mem_mb: dict[int, float] = {}

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            util_gpu = util_mem_pct = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_gpu = util.gpu
                util_mem_pct = util.memory
            except Exception:
                pass

            temp_c = None
            try:
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                pass

            power_w = None
            try:
                power_w = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            except Exception:
                pass

            # Per-process GPU memory (for per-model attribution)
            try:
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    pid_mem_mb[p.pid] = p.usedGpuMemory / 1024 / 1024
            except Exception:
                pass

            result["gpus"].append(
                {
                    "index": i,
                    "memory_used_mb": round(mem.used / 1024 / 1024, 1),
                    "memory_total_mb": round(mem.total / 1024 / 1024, 1),
                    "memory_used_pct": (
                        round(mem.used / mem.total * 100, 1) if mem.total else 0.0
                    ),
                    "utilization_pct": util_gpu,
                    "mem_util_pct": util_mem_pct,
                    "temperature_c": temp_c,
                    "power_w": power_w,
                }
            )

        if pid_to_flavor:
            for pid, flavor in pid_to_flavor.items():
                if pid in pid_mem_mb:
                    result["per_model_gpu_mb"][flavor] = round(pid_mem_mb[pid], 1)

    except Exception as exc:
        # Permanent absence (no libnvidia-ml.so.1 — Jetson-Tegra / CPU) → disable so
        # the telemetry loop stops retrying. Transient errors keep logging instead.
        missing = "LibraryNotFound" in type(exc).__name__ or "libnvidia-ml" in str(exc)
        if missing:
            _NVML_DISABLED = True
            logger.warning(
                "_collect_gpu_stats: NVML unavailable (no libnvidia-ml.so.1 / no NVIDIA "
                "GPU); GPU telemetry disabled for this process"
            )
        else:
            logger.error("_collect_gpu_stats: pynvml failed", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# ModelState — per model flavor lifecycle
# ---------------------------------------------------------------------------


@dataclass
class ModelState:
    """Tracks load lifecycle for one model flavor."""

    loaded: bool = False
    loading: bool = False
    # Each waiter: (uvicorn_identity, req_id, deadline_monotonic_s)
    load_waiters: list[tuple[bytes, int, float]] = field(default_factory=list)
    # Admin T_LOAD waiters: (identity, req_id) — replied T_OK / T_ERROR
    admin_waiters: list[tuple[bytes, int]] = field(default_factory=list)
    # Persisted load config — reused by _schedule_reload after worker death.
    api_key: str = ""
    device: str = ""
    # Admin-loaded (preload): never evicted; survives respawn via persisted state.
    pinned: bool = False


# ---------------------------------------------------------------------------
# BackendLike protocol
# ---------------------------------------------------------------------------


class BackendLike(Protocol):
    def signal_slot(
        self, slot_id: int, req_id: int, params_bytes: bytes = ...
    ) -> None: ...


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
        n_slots: int = cfg.MMP_N_SLOTS_DEFAULT,
        input_mb: float = cfg.MMP_INPUT_MB_DEFAULT,
        stale_reap_interval_s: float = cfg.INFERENCE_STALE_REAP_INTERVAL_S,
        stale_slot_max_age_s: float = cfg.INFERENCE_STALE_SLOT_MAX_AGE_S,
        evict_threshold: float = cfg.INFERENCE_GPU_EVICTION_THRESHOLD,
        evict_check_interval_s: float = cfg.INFERENCE_EVICT_CHECK_INTERVAL_S,
        monitor_interval_s: float = cfg.INFERENCE_MONITOR_INTERVAL_S,
        idle_timeout_s: float = cfg.INFERENCE_MODEL_IDLE_TIMEOUT_S,
        load_oom_max_evictions: int = cfg.INFERENCE_LOAD_OOM_MAX_EVICTIONS,
        decoder: str = "imagecodecs",
        batch_max_size: int = 0,
        batch_max_wait_ms: float = 5.0,
        max_pinned_memory_mb: int = cfg.INFERENCE_MAX_PINNED_MEMORY_MB,
        vram_admission: bool = cfg.INFERENCE_VRAM_ADMISSION_CONTROL,
        vram_window_size: int = cfg.INFERENCE_VRAM_WINDOW_SIZE,
        vram_idle_cutoff_s: Optional[float] = cfg.INFERENCE_VRAM_IDLE_CUTOFF_S,
        vram_headroom_mb: float = cfg.INFERENCE_VRAM_HEADROOM_MB,
        vram_recent_window_s: float = cfg.INFERENCE_VRAM_RECENT_WINDOW_S,
    ) -> None:
        """
        Args:
            n_slots: SHM pool slot count.
            input_mb: MB per slot data area.
            decoder: Image decoder for SubprocessBackend workers.
                ``"imagecodecs"`` (default, CPU) or ``"nvjpeg"`` (GPU).
            batch_max_size: Max images per worker batch (0 = model default).
            batch_max_wait_ms: Max ms to wait for a full batch.
            max_pinned_memory_mb: Passed to ModelManager for CPU pinned memory budget.
        """
        self._n_slots = n_slots
        self._input_mb = input_mb
        self._decoder = decoder
        self._batch_max_size = batch_max_size
        self._batch_max_wait_ms = batch_max_wait_ms
        self._stale_reap_interval_s = stale_reap_interval_s
        self._stale_slot_max_age_s = stale_slot_max_age_s
        self._evict_threshold = evict_threshold
        self._evict_check_interval_s = evict_check_interval_s
        self._monitor_interval_s = monitor_interval_s
        self._idle_timeout_s = idle_timeout_s
        self._load_oom_max_evictions = load_oom_max_evictions
        self._vram_admission = vram_admission
        self._vram_window_size = vram_window_size
        self._vram_idle_cutoff_s = (
            vram_idle_cutoff_s if vram_idle_cutoff_s is not None else idle_timeout_s
        )
        self._vram_headroom_mb = vram_headroom_mb
        self._vram_recent_window_s = vram_recent_window_s
        self._manager = ModelManager(
            n_slots=n_slots,
            input_mb=input_mb,
            max_pinned_memory_mb=max_pinned_memory_mb,
        )
        self._manager._shared_death_hook = self._on_shared_worker_death

        # Shared-base bookkeeping.
        self._shared_metadata_cache: dict = {}
        self._shared_heads: dict[str, set[str]] = {}  # base_key → head_ids
        self._head_base_key: dict[str, str] = {}  # head_id → base_key
        self._preloaded_shared_bases: dict[str, str] = {}  # model_id → base_key

        self._pool: Optional[SHMPool] = None
        self._router: Optional[zmq.asyncio.Socket] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # req_id → (uvicorn_identity, slot_id, flavor)
        self._pending: dict[int, tuple[bytes, int, str]] = {}

        self._rejects_pool_full = 0

        # slot_id → flavor for slots signalled to a worker and not yet
        # resulted. The reaper must never free these: the worker may be
        # mid-batch on them, and freeing allows re-allocation (torn slot).
        self._inflight: dict[int, str] = {}

        # Strong refs for fire-and-forget tasks (GC'd mid-flight otherwise)
        self._bg_tasks: set = set()

        # flavor → ModelState
        self._models: dict[str, ModelState] = {}

        # model_id → BackendLike (registered by register_backend or _load_model)
        self._backends: dict[str, BackendLike] = {}

        # flavor → monotonic timestamp of last T_SUBMIT (LRU eviction + hot/cold)
        self._model_access: dict[str, float] = {}

        # flavor → list of request timestamps (sliding window for request rate)
        self._model_request_times: dict[str, list[float]] = {}

        # VRAM admission control state (only used when _vram_admission is on)
        # flavor → sliding window of measured GPU MB samples (footprint = max)
        self._vram_window: dict[str, deque] = {}
        # model_id → static per-batch VRAM MB (0 = no data); fetched once, cached
        self._vram_meta_cache: dict[str, int] = {}
        # flavors currently being evicted by an admission plan (race guard)
        self._unloading: set[str] = set()

        # Cache hit/miss counters (model already loaded vs triggered load)
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Latest monitoring snapshot (updated by _monitoring_loop)
        self._stats_snapshot: dict[str, Any] = {}

        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def manager(self) -> ModelManager:
        return self._manager

    @property
    def shm_name(self) -> Optional[str]:
        """SHM pool name — valid after run() has started."""
        return self._pool.name if self._pool else None

    @property
    def bound_addr(self) -> Optional[str]:
        """Actual bound ZMQ address — valid after run() has started."""
        return getattr(self, "_bound_addr", None)

    def register_backend(self, model_id: str, backend: BackendLike) -> None:
        """Mark model_id loaded and register its backend.

        Called by ModelManager (or tests) after a model is ready.
        Thread-safe: may be called from any thread before or during run().
        """
        self._backends[model_id] = backend
        if hasattr(backend, "set_on_result_callback"):
            backend.set_on_result_callback(self.on_result)
        if hasattr(backend, "set_on_death_callback"):
            backend.set_on_death_callback(self._on_backend_death)
        fs = self._models.setdefault(model_id, ModelState())
        fs.loading = False
        fs.loaded = True
        # Prevent immediate eviction — treat (re)load time as first access.
        # Unconditional: a stale pre-crash timestamp surviving a reload would
        # classify the fresh model cold and re-evict it (load/evict flapping).
        self._model_access[model_id] = time.monotonic()

        if self._loop is not None and (fs.load_waiters or fs.admin_waiters):
            self._loop.call_soon_threadsafe(self._flush_load_waiters, model_id)

    def on_result(self, req_id: int, slot_id: int, result_sz: int) -> None:
        """Called by SubprocessBackend recv thread when a slot completes.

        Thread-safe. ``result_sz == 0`` signals inference error.
        """
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._on_result_on_loop, req_id, slot_id, result_sz
            )

    def _on_backend_death(self, model_id: str) -> None:
        """Called by SubprocessBackend recv thread when worker dies.

        Thread-safe. Schedules reload on the event loop.
        """
        logger.warning("MMP: backend '%s' died, scheduling reload", model_id)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._schedule_reload, model_id)

    def _on_shared_worker_death(self, base_key: str) -> None:
        """Called (off-loop) when a shared-base worker dies, taking all its heads
        with it. Schedules per-head cleanup on the event loop."""
        logger.warning("MMP: shared-base worker '%s' died", base_key)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._cleanup_dead_shared_base, base_key)

    def _untrack_shared_head(self, model_id: str) -> None:
        """Drop a head from shared-base bookkeeping on graceful unload/eviction."""
        base_key = self._head_base_key.pop(model_id, None)
        if base_key is None:
            return
        heads = self._shared_heads.get(base_key)
        if heads is not None:
            heads.discard(model_id)
            if not heads:
                self._shared_heads.pop(base_key, None)

    def _cleanup_dead_shared_base(self, base_key: str) -> None:
        """Mark every head hosted by a dead shared-base worker not-loaded and drop its
        state. Heads reload lazily on their next request via the normal cold path
        (which re-resolves the base and spawns a fresh worker). Runs on the loop."""
        head_ids = self._shared_heads.pop(base_key, set())
        for model_id, preload_base_key in list(self._preloaded_shared_bases.items()):
            if preload_base_key != base_key:
                continue
            self._preloaded_shared_bases.pop(model_id, None)
            fs = self._models.get(model_id)
            if fs is not None:
                fs.loaded = False
                fs.loading = False
        for head_id in head_ids:
            self._head_base_key.pop(head_id, None)
            fs = self._models.get(head_id)
            if fs is not None:
                fs.loaded = False
                fs.loading = False
            for slot_id in [s for s, m in self._inflight.items() if m == head_id]:
                del self._inflight[slot_id]
            self._backends.pop(head_id, None)
            # Owner is already torn down; drop the dead view directly.
            self._manager._backends.pop(head_id, None)

    def _kick_pending_load(self, model_id: str) -> None:
        """Start a load deferred by the _unloading gate, if waiters accumulated
        while the model was being unloaded/evicted. Runs on the event loop."""
        fs = self._models.get(model_id)
        if fs is None or fs.loading or fs.loaded:
            return
        if not (fs.load_waiters or fs.admin_waiters):
            return
        fs.loading = True
        self._spawn(
            self._load_model(model_id, api_key=fs.api_key, device=fs.device),
            name=f"mmp-load-{model_id}",
        )

    def _schedule_reload(self, model_id: str) -> None:
        """Mark model as loading and trigger _load_model. Runs on event loop."""
        fs = self._models.get(model_id)
        if fs is None:
            return
        # Worker is dead: its tickets are void. Drop them so the reaper can
        # reclaim the slots once stale.
        for slot_id in [s for s, m in self._inflight.items() if m == model_id]:
            del self._inflight[slot_id]
        if fs.loading:
            return  # already reloading
        if model_id in self._unloading:
            return  # being unloaded; _kick_pending_load restarts if waiters exist
        fs.loaded = False
        fs.loading = True
        logger.info("MMP: reloading '%s' after worker death", model_id)
        asyncio.create_task(
            self._load_model(model_id, api_key=fs.api_key, device=fs.device),
            name=f"mmp-reload-{model_id}",
        )

    # ------------------------------------------------------------------
    # Main async entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        addr: Optional[str] = None,
        ready_event: Optional[threading.Event] = None,
    ) -> None:
        """Bind ROUTER, create SHM pool, dispatch messages until stopped."""
        self._loop = asyncio.get_running_loop()
        self._running = True

        # SHM pool — owned by ModelManager, shared with MMP and workers.
        # One pool, one owner (ModelManager), one source of truth.
        self._manager._ensure_pool()
        self._pool = self._manager._pool
        logger.info(
            "MMP: SHM pool ready  name=%s  slots=%d  data=%.0fMB",
            self._pool.name,
            self._n_slots,
            self._input_mb,
        )

        # Bind ROUTER
        ctx = zmq.asyncio.Context()
        self._router = ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._router.setsockopt(zmq.SNDHWM, 0)
        self._router.setsockopt(zmq.RCVHWM, 0)
        self._router.setsockopt(zmq.LINGER, 0)
        bind_addr = addr or zmq_addr("mmprocess")
        if bind_addr.startswith("ipc://"):
            # libzmq does not unlink stale ipc socket files — after SIGKILL the
            # next bind fails EADDRINUSE without this.
            try:
                os.unlink(bind_addr[len("ipc://") :])
            except OSError:
                pass
        self._router.bind(bind_addr)
        self._bound_addr = self._router.getsockopt_string(zmq.LAST_ENDPOINT)
        logger.info("MMP: ROUTER bound on %s", self._bound_addr)

        try:
            self._loop.add_signal_handler(signal.SIGTERM, self._on_sigterm)
        except (NotImplementedError, RuntimeError):
            pass

        self._stop_event: asyncio.Event = asyncio.Event()

        if ready_event is not None:
            ready_event.set()

        tasks = [
            asyncio.create_task(self._recv_loop(), name="mmp-recv"),
            asyncio.create_task(self._stale_reaper_loop(), name="mmp-stale-reaper"),
            asyncio.create_task(self._eviction_loop(), name="mmp-evictor"),
            asyncio.create_task(self._monitoring_loop(), name="mmp-monitor"),
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
            self._pool = None
            self._router = None
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
                # A dead ROUTER means the hub can never serve again — shut down
                # fully so the supervisor restarts us, instead of running as a
                # zombie (background loops alive, no message processing).
                logger.error("MMP: recv error, shutting down: %s", exc)
                self._set_stop_event()
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
        data = frames[2:]

        try:
            if msg_type == T_ENSURE_LOADED:
                await self._handle_ensure_loaded(identity, data)
            elif msg_type == T_ALLOC:
                await self._handle_alloc(identity, data)
            elif msg_type == T_SUBMIT:
                await self._handle_submit(identity, data)
            elif msg_type == T_FREE:
                self._handle_free(data)
            elif msg_type == T_CANCEL:
                self._handle_cancel(data)
            elif msg_type == T_LOAD:
                await self._handle_load(identity, data)
            elif msg_type == T_UNLOAD:
                await self._handle_unload(identity, data)
            elif msg_type == T_STATS:
                await self._handle_stats(identity, data)
            elif msg_type == T_SHM_INFO:
                await self._handle_shm_info(identity, data)
            else:
                logger.debug("MMP: unknown msg_type=%r from %r", msg_type, identity)
        except Exception:
            logger.exception("MMP: unhandled error dispatching %r", msg_type)

    # ------------------------------------------------------------------
    # T_ENSURE_LOADED  →  T_MODEL_READY | T_LOAD_TIMEOUT
    # wire: Q I H N H M H D
    #   req_id(8) wait_ms(4) model_id_len(2) model_id(N)
    #   key_len(2) key(M) device_len(2) device(D)
    # ------------------------------------------------------------------

    async def _handle_ensure_loaded(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 14:
            return
        frame = data[0]
        req_id, wait_ms, mid_len = struct.unpack_from(">QIH", frame)
        off = 14
        model_id = frame[off : off + mid_len].decode(errors="replace")
        off += mid_len
        api_key = ""
        if len(frame) >= off + 2:
            klen = struct.unpack_from(">H", frame, off)[0]
            api_key = frame[off + 2 : off + 2 + klen].decode(errors="replace")
            off += 2 + klen
        device = ""
        if len(frame) >= off + 2:
            dlen = struct.unpack_from(">H", frame, off)[0]
            device = frame[off + 2 : off + 2 + dlen].decode(errors="replace")
        deadline = time.monotonic() + wait_ms / 1000.0

        fs = self._models.get(model_id)
        if fs is not None and fs.loaded:
            self._cache_hits += 1
            await self._send(identity, T_MODEL_READY, struct.pack(">Q", req_id))
            return

        self._cache_misses += 1

        if fs is None:
            fs = ModelState()
            self._models[model_id] = fs

        fs.load_waiters.append((identity, req_id, deadline))
        asyncio.create_task(
            self._expire_waiter(model_id, identity, req_id, deadline),
            name=f"mmp-waiter-timeout-{req_id}",
        )

        if not fs.loading:
            fs.api_key = api_key
            fs.device = device
            # Gated on _unloading: a load starting mid-drain races the unload's
            # backend pop (bricked model). _kick_pending_load restarts after.
            if model_id not in self._unloading:
                fs.loading = True
                asyncio.create_task(
                    self._load_model(model_id, api_key=api_key, device=device),
                    name=f"mmp-load-{model_id}",
                )

    # ------------------------------------------------------------------
    # T_ALLOC  →  T_ALLOC_OK | T_ERROR
    # wire: Q H N   req_id(8) mid_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_alloc(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, mid_len = struct.unpack_from(">QH", frame)
        model_id = frame[10 : 10 + mid_len].decode(errors="replace")

        # Optional fail-fast backpressure: cap in-flight requests per model so
        # a slow model sheds load immediately instead of queueing into the
        # 30s reaper window.
        cap = cfg.INFERENCE_MAX_INFLIGHT_PER_MODEL
        if cap > 0:
            inflight = sum(1 for _, _, m in self._pending.values() if m == model_id)
            if inflight >= cap:
                self._rejects_pool_full += 1
                await self._send(
                    identity, T_ERROR, struct.pack(">QB", req_id, _ERR_POOL_FULL)
                )
                return

        try:
            slot_id = self._pool.alloc_slot(timeout=0)
        except TimeoutError:
            self._rejects_pool_full += 1
            await self._send(
                identity, T_ERROR, struct.pack(">QB", req_id, _ERR_POOL_FULL)
            )
            return

        self._pool.mark_allocated(slot_id, req_id)
        await self._send(identity, T_ALLOC_OK, struct.pack(">QI", req_id, slot_id))

    # ------------------------------------------------------------------
    # T_SUBMIT  (no reply — result delivered via on_result callback)
    # wire: Q I I H N   req_id(8) slot_id(4) input_sz(4) mid_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_submit(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 18:
            return
        frame = data[0]
        req_id, slot_id, input_sz, mid_len = struct.unpack_from(">QIIH", frame)
        model_id = frame[18 : 18 + mid_len].decode(errors="replace")
        params_bytes = data[1] if len(data) > 1 else b"{}"

        hdr = self._pool.read_header(slot_id)
        if hdr.status != SlotStatus.ALLOCATED or hdr.request_id != req_id:
            logger.warning(
                "MMP: rejecting submit for slot %d — ownership mismatch "
                "(status=%d hdr_req=%d submit_req=%d)",
                slot_id,
                hdr.status,
                hdr.request_id,
                req_id,
            )
            await self._send(identity, T_ERROR, struct.pack(">QB", req_id, _ERR_STALE))
            return

        self._pending[req_id] = (identity, slot_id, model_id)
        self._pool.mark_written(slot_id, input_sz)
        self._forward_to_backend(model_id, slot_id, req_id, params_bytes)

    # ------------------------------------------------------------------
    # T_FREE
    # wire: Q I   req_id(8) slot_id(4)
    # ------------------------------------------------------------------

    def _handle_free(self, data: list[bytes]) -> None:
        if not data or len(data[0]) < 12:
            return
        req_id, slot_id = struct.unpack_from(">QI", data[0])
        try:
            self._pool.free_slot(slot_id, request_id=req_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # T_CANCEL — client gave up (timeout / disconnect)
    # wire: Q   req_id(8)
    # ------------------------------------------------------------------

    def _handle_cancel(self, data: list[bytes]) -> None:
        """Client abandoned req_id. Drop its reply target but do NOT free the
        slot — the worker may still hold a ticket for it. The slot is freed when
        the worker's T_RESULT lands (the pending-None branch in
        _on_result_on_loop) or when the reaper reclaims it. This is what stops a
        slot being reused while a live worker ticket still references it."""
        if not data or len(data[0]) < 8:
            return
        req_id = struct.unpack_from(">Q", data[0])[0]
        self._pending.pop(req_id, None)

    # ------------------------------------------------------------------
    # T_LOAD (admin) — trigger model load + reply T_OK immediately
    # wire: Q H N H N   req_id(8) mid_len(2) flavor(N) api_key_len(2) api_key(M)
    # ------------------------------------------------------------------

    async def _handle_load(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, mid_len = struct.unpack_from(">QH", frame)
        off = 10
        model_id = frame[off : off + mid_len].decode(errors="replace")
        off += mid_len
        api_key = ""
        if len(frame) >= off + 2:
            klen = struct.unpack_from(">H", frame, off)[0]
            api_key = frame[off + 2 : off + 2 + klen].decode(errors="replace")

        fs = self._models.setdefault(model_id, ModelState())
        fs.pinned = True  # admin/preload load is non-evictable
        if fs.loaded:
            await self._send(identity, T_OK, struct.pack(">Q", req_id))
            return

        # Join the load (in progress or started here); replied on flush/fail.
        fs.admin_waiters.append((identity, req_id))
        if not fs.loading:
            fs.api_key = api_key
            if model_id not in self._unloading:
                fs.loading = True
                self._spawn(
                    self._load_model(model_id, api_key=api_key),
                    name=f"mmp-admin-load-{model_id}",
                )

    # ------------------------------------------------------------------
    # T_UNLOAD (admin)
    # wire: Q H N   req_id(8) mid_len(2) flavor(N)
    # ------------------------------------------------------------------

    async def _handle_unload(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 10:
            return
        frame = data[0]
        req_id, mid_len = struct.unpack_from(">QH", frame)
        model_id = frame[10 : 10 + mid_len].decode(errors="replace")

        # Off the dispatch path — a drain can take up to 30s and must not
        # block _recv_loop for other clients.
        self._spawn(
            self._do_unload(identity, req_id, model_id),
            name=f"mmp-unload-{model_id}",
        )

    async def _do_unload(self, identity: bytes, req_id: int, model_id: str) -> None:
        """Admin unload off the dispatch path: drain in executor, reply after."""
        loop = asyncio.get_running_loop()
        fs = self._models.get(model_id)
        if fs:
            fs.loaded = False
            fs.loading = False
        self._unloading.add(model_id)
        victim = self._backends.get(model_id)
        try:
            if self._manager is not None:
                if model_id in self._preloaded_shared_bases:
                    await loop.run_in_executor(
                        None, lambda: self._manager.unload_shared_base(model_id)
                    )
                else:
                    await loop.run_in_executor(
                        None, lambda: self._manager.unload(model_id, drain=True)
                    )
            cur = self._backends.pop(model_id, None)
            if cur is not None and cur is not victim:
                # A load that was already in flight registered a fresh backend
                # mid-drain — keep it, only the drained victim is gone.
                self._backends[model_id] = cur
            else:
                self._preloaded_shared_bases.pop(model_id, None)
                self._untrack_shared_head(model_id)
            await self._send(identity, T_OK, struct.pack(">Q", req_id))
        except Exception:
            logger.exception("MMP: T_UNLOAD '%s' failed", model_id)
            await self._send(
                identity, T_ERROR, struct.pack(">QB", req_id, _ERR_NOT_LOADED)
            )
        finally:
            self._unloading.discard(model_id)
            self._kick_pending_load(model_id)

    # ------------------------------------------------------------------
    # T_STATS — snapshot reply
    # wire: Q   req_id(8)
    # reply: T_STATS_RESP  Q I N   req_id(8) json_len(4) json(N)
    # ------------------------------------------------------------------

    async def _handle_stats(self, identity: bytes, data: list[bytes]) -> None:
        req_id = (
            struct.unpack_from(">Q", data[0])[0] if (data and len(data[0]) >= 8) else 0
        )
        # Start from cached monitoring snapshot (includes psutil/GPU from last poll)
        snapshot: dict[str, Any] = dict(self._stats_snapshot)
        # Overlay with manager stats (skip 'models' — mmp_models is the source of
        # truth; skip 'gpus' — NVML snapshot wins, manager's torch view is empty in MMP)
        if self._manager is not None:
            try:
                mgr_stats = self._manager.stats()
                mgr_stats.pop("models", None)
                mgr_stats.pop("gpus", None)
                snapshot.update(mgr_stats)
            except Exception:
                pass
        now = time.monotonic()
        # Per-model pending count from _pending
        pending_per_model: dict[str, int] = {}
        for _, _, mid in self._pending.values():
            pending_per_model[mid] = pending_per_model.get(mid, 0) + 1
        model_stats: dict[str, Any] = {}
        for f, fs in self._models.items():
            last_access = self._model_access.get(f)
            idle_s = (now - last_access) if last_access else None
            req_times = self._model_request_times.get(f, [])
            # Trim stale entries for accurate rate
            cutoff = now - self._vram_recent_window_s
            while req_times and req_times[0] < cutoff:
                req_times.pop(0)

            backend = self._backends.get(f)

            if fs.loading:
                model_state = "loading"
            elif backend is not None:
                try:
                    model_state = backend.state
                except Exception:
                    model_state = "loaded" if fs.loaded else "unloaded"
            elif fs.loaded:
                model_state = "loaded"
            else:
                model_state = "unloaded"

            device = None
            if backend is not None:
                try:
                    device = backend.device
                except Exception:
                    pass

            entry: dict[str, Any] = {
                "state": model_state,
                "device": device,
                "loaded": fs.loaded,
                "loading": fs.loading,
                "last_access_ts": last_access,
                "idle_s": round(idle_s, 1) if idle_s is not None else None,
                "is_cold": idle_s is not None and idle_s > self._idle_timeout_s,
                "request_rate_60s": len(req_times),
            }
            entry["queue_depth"] = pending_per_model.get(f, 0)
            if backend is not None:
                try:
                    entry["worker_alive"] = backend.is_healthy
                    entry["worker_pid"] = getattr(backend, "worker_pid", None)
                    entry["model_class_name"] = getattr(
                        backend, "_model_class_name", None
                    )
                    entry = {**backend.stats(), **entry}
                except Exception:
                    pass
            if self._manager is not None:
                try:
                    entry["tasks"] = self._manager.get_supported_tasks(f)
                except Exception:
                    logger.debug("MMP: failed to get tasks for '%s'", f, exc_info=True)
            if backend is not None:
                try:
                    entry["class_names"] = backend.class_names
                except Exception:
                    logger.debug(
                        "MMP: failed to get class_names for '%s'", f, exc_info=True
                    )
            if f in self._preloaded_shared_bases and self._manager is not None:
                base_key = self._preloaded_shared_bases[f]
                owner = self._manager.shared_owners().get(base_key)
                entry["backend_type"] = "shared-base"
                entry["base_key"] = base_key
                if owner is not None:
                    entry["worker_alive"] = owner.is_healthy
                    entry["worker_pid"] = getattr(owner, "worker_pid", None)
                    entry["device"] = owner.device
            model_stats[f] = entry

        snapshot.update(
            {
                "mmp_free_slots": self._pool.free_count if self._pool else 0,
                "mmp_total_slots": self._n_slots,
                "mmp_pending": len(self._pending),
                "mmp_rejects_pool_full": self._rejects_pool_full,
                "mmp_cache_hits": self._cache_hits,
                "mmp_cache_misses": self._cache_misses,
                "mmp_idle_timeout_s": self._idle_timeout_s,
                "mmp_models": model_stats,
            }
        )
        payload = json.dumps(snapshot, default=str).encode()
        await self._send(
            identity,
            T_STATS_RESP,
            struct.pack(">QI", req_id, len(payload)) + payload,
        )

    async def _handle_shm_info(self, identity: bytes, data: list[bytes]) -> None:
        if not data or len(data[0]) < 8:
            return
        req_id = struct.unpack_from(">Q", data[0])[0]
        name = (self.shm_name or "").encode()
        data_size = int(self._input_mb * 1024 * 1024)
        await self._send(
            identity,
            T_SHM_INFO_RESP,
            struct.pack(">QIQH", req_id, self._n_slots, data_size, len(name)) + name,
        )

    # ------------------------------------------------------------------
    # Backend routing
    # ------------------------------------------------------------------

    def _forward_to_backend(
        self, model_id: str, slot_id: int, req_id: int, params_bytes: bytes = b"{}"
    ) -> None:
        now = time.monotonic()
        self._model_access[model_id] = now
        # Sliding window: append timestamp, trim old entries
        times = self._model_request_times.setdefault(model_id, [])
        times.append(now)
        # Keep only the recent-window of timestamps
        cutoff = now - self._vram_recent_window_s
        while times and times[0] < cutoff:
            times.pop(0)
        backend = self._backends.get(model_id)
        if backend is None:
            logger.warning("MMP: no backend for '%s', req_id=%d", model_id, req_id)
            self._on_result_on_loop(req_id, slot_id, 0)
            return
        if not backend.is_healthy:
            logger.warning(
                "MMP: backend '%s' unhealthy, triggering reload (req_id=%d)",
                model_id,
                req_id,
            )
            self._schedule_reload(model_id)
            self._on_result_on_loop(req_id, slot_id, 0)
            return
        try:
            backend.signal_slot(slot_id, req_id, params_bytes)
        except Exception:
            logger.exception("MMP: signal_slot raised for '%s'", model_id)
            self._schedule_reload(model_id)
            self._on_result_on_loop(req_id, slot_id, 0)
            return
        self._inflight[slot_id] = model_id

    # ------------------------------------------------------------------
    # on_result — event-loop side
    # ------------------------------------------------------------------

    def _on_result_on_loop(self, req_id: int, slot_id: int, result_sz: int) -> None:
        """Must be called on the event loop thread."""
        self._inflight.pop(slot_id, None)
        pending = self._pending.pop(req_id, None)
        if pending is None:
            # Stale reaper already freed this, or duplicate callback — free
            # slot only if it is still bound to this request (it may have been
            # re-allocated since).
            self._pool.free_slot(slot_id, request_id=req_id)
            return
        identity, _, _ = pending
        asyncio.create_task(
            self._reply_result(identity, req_id, slot_id, result_sz),
            name=f"mmp-reply-{req_id}",
        )

    async def _reply_result(
        self,
        identity: bytes,
        req_id: int,
        slot_id: int,
        result_sz: int,
    ) -> None:
        if result_sz == 0:
            await self._send(
                identity, T_ERROR, struct.pack(">QB", req_id, _ERR_BACKEND)
            )
            self._pool.free_slot(slot_id, request_id=req_id)
            return
        self._pool.mark_done(slot_id, result_sz)
        sent = await self._send(
            identity,
            T_RESULT_READY,
            struct.pack(">QII", req_id, slot_id, result_sz),
        )
        if not sent:
            # Peer disconnected — result will never be read; free slot now
            # (backend already finished; we just drop the result silently)
            logger.debug(
                "MMP: peer gone for req_id=%d slot=%d — freeing slot",
                req_id,
                slot_id,
            )
            self._pool.free_slot(slot_id, request_id=req_id)

    # ------------------------------------------------------------------
    # Cold path — load
    # ------------------------------------------------------------------

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        """Check if exception is a CUDA out-of-memory error."""
        msg = str(exc).lower()
        return "cuda" in msg and ("out of memory" in msg or "oom" in msg)

    async def _load_model(
        self, model_id: str, api_key: str = "", device: str = ""
    ) -> None:
        """Latch backstop: _load_model_inner reports failures via _fail_load
        itself; this wrapper guarantees an unexpected exception can never
        leave fs.loading latched True forever."""
        try:
            await self._load_model_inner(model_id, api_key=api_key, device=device)
        except Exception:
            logger.exception("MMP: _load_model('%s') crashed", model_id)
            self._fail_load(model_id, self._models.get(model_id), _ERR_LOAD_FAILED)

    async def _load_model_inner(
        self, model_id: str, api_key: str = "", device: str = ""
    ) -> None:
        """Load model via ModelManager with reactive admission loop.

        On CUDA OOM: evicts coldest model, retries. Repeats until success,
        no cold models left (_ERR_SERVER_FULL), or non-OOM failure (_ERR_LOAD_FAILED).

        model_id is the full routing key (may include ":instance" suffix).
        model_id_or_path strips the suffix to fetch the correct weights.
        """
        loop = asyncio.get_running_loop()
        fs = self._models.get(model_id)

        if self._manager is None:
            # Stub mode: no manager
            self._stub_load(model_id, fs)
            return

        model_id_or_path = model_id.rsplit(":", 1)[0]

        # Detect a shareable head before admission (detect-before-admit). Resolution is
        # a cached, weight-free provider lookup; failure falls back to a normal load.
        from inference_model_manager.shared_base_resolution import (
            resolve_shared_base,
            resolve_shared_base_model,
        )

        resolution = await loop.run_in_executor(
            None,
            lambda: resolve_shared_base(
                model_id_or_path, api_key, device, self._shared_metadata_cache
            ),
        )
        base_preload_resolution = None
        if resolution is None and fs is not None and fs.pinned:
            base_preload_resolution = await loop.run_in_executor(
                None,
                lambda: resolve_shared_base_model(
                    model_id_or_path, api_key, device, self._shared_metadata_cache
                ),
            )
        # The base counts once: when its worker is already resident, admit the head on
        # its marginal footprint (whole-model minus the resident base), not the full
        # model. First head / non-shared loads admit on the full footprint.
        base_resident = resolution is not None and self._manager.has_shared_base(
            resolution.base_key
        )

        # A loading head must never evict the base it is about to share.
        shared_resolution = resolution or base_preload_resolution
        shared_exclude = (
            {shared_resolution.base_key} if shared_resolution is not None else None
        )

        if self._vram_admission:
            need_mb = None
            if base_resident:
                need_mb = await loop.run_in_executor(
                    None,
                    self._shared_head_marginal_mb,
                    resolution,
                    model_id_or_path,
                    api_key,
                )
            decision, victims, deficit = await self._vram_admission_plan(
                model_id, api_key, need_mb=need_mb, exclude=shared_exclude
            )
            if decision == "no_capacity":
                logger.warning(
                    "MMP: VRAM admission denied '%s' — no evictable capacity",
                    model_id,
                )
                self._fail_load(model_id, fs, _ERR_SERVER_FULL)
                return
            if decision == "evict":
                logger.warning(
                    "MMP: VRAM admission evicting %s to load '%s' (deficit=%.0f MB)",
                    victims,
                    model_id,
                    deficit,
                )
                secured = await self._execute_eviction_plan(
                    victims, deficit, exclude=shared_exclude
                )
                if not secured:
                    logger.warning(
                        "MMP: VRAM admission aborted for '%s' — capacity lost mid-evict",
                        model_id,
                    )
                    self._fail_load(model_id, fs, _ERR_SERVER_FULL)
                    return

        max_retries = self._load_oom_max_evictions  # safety cap — never loop forever

        # Reload after worker death/unhealthy leaves a stale entry in the
        # ModelManager. Clear it (freeing its resources) so the load below
        # won't be rejected with "already loaded".
        if model_id in self._manager:
            try:
                await loop.run_in_executor(None, lambda: self._manager.unload(model_id))
            except Exception:
                self._manager._backends.pop(model_id, None)

        for attempt in range(max_retries):
            logger.info(
                "MMP: loading '%s' (weights=%s device=%s attempt=%d)",
                model_id,
                model_id_or_path,
                device or "default",
                attempt + 1,
            )
            try:
                if resolution is not None:
                    await loop.run_in_executor(
                        None,
                        lambda: self._manager.load_shared_head(
                            model_id,
                            api_key,
                            resolution,
                            model_id_or_path=model_id_or_path,
                            device=device or None,
                            batch_max_size=self._batch_max_size,
                            batch_max_delay_ms=self._batch_max_wait_ms,
                            decoder=self._decoder,
                        ),
                    )
                elif base_preload_resolution is not None:
                    await loop.run_in_executor(
                        None,
                        lambda: self._manager.load_shared_base(
                            model_id,
                            api_key,
                            base_preload_resolution,
                            device=device or None,
                            batch_max_size=self._batch_max_size,
                            batch_max_delay_ms=self._batch_max_wait_ms,
                            decoder=self._decoder,
                        ),
                    )
                    if not await self._enforce_vram_headroom_after_load(
                        model_id,
                        fs,
                        is_shared_base_preload=True,
                        exclude=shared_exclude,
                    ):
                        return
                    self._preloaded_shared_bases[model_id] = (
                        base_preload_resolution.base_key
                    )
                    fs = self._models.setdefault(model_id, ModelState())
                    fs.loading = False
                    fs.loaded = True
                    self._model_access[model_id] = time.monotonic()
                    self._flush_load_waiters(model_id)
                    return
                else:
                    await loop.run_in_executor(
                        None,
                        lambda: self._manager.load(
                            model_id,
                            api_key,
                            model_id_or_path=model_id_or_path,
                            backend="subprocess",
                            device=device or None,
                            decoder=self._decoder,
                            batch_max_size=self._batch_max_size,
                            batch_max_delay_ms=self._batch_max_wait_ms,
                        ),
                    )
                if not await self._enforce_vram_headroom_after_load(
                    model_id, fs, exclude=shared_exclude
                ):
                    return
                backend = self._manager.get_backend(model_id)
                if backend is not None and hasattr(backend, "signal_slot"):
                    # Track BEFORE register_backend exposes the head as loaded, so a
                    # worker death can never see an untracked-but-loaded head.
                    if resolution is not None:
                        self._shared_heads.setdefault(resolution.base_key, set()).add(
                            model_id
                        )
                        self._head_base_key[model_id] = resolution.base_key
                    self.register_backend(model_id, backend)
                    return
                # Loaded but no backend — fall through to stub
                break

            except Exception as exc:
                if not self._is_cuda_oom(exc):
                    # Non-OOM failure — don't retry
                    logger.exception("MMP: load '%s' failed (non-OOM)", model_id)
                    self._fail_load(model_id, fs, _ERR_LOAD_FAILED)
                    return

                # CUDA OOM — evict a cold unit and retry. For a shared head, never
                # evict the base it is loading onto (head-load OOM must free OTHER
                # units, not the dependency this load needs).
                candidate = self._pick_eviction_candidate(exclude=shared_exclude)
                if candidate is None:
                    logger.error(
                        "MMP: OOM loading '%s' — all models hot, cannot evict",
                        model_id,
                    )
                    self._fail_load(model_id, fs, _ERR_SERVER_FULL)
                    return

                logger.warning(
                    "MMP: OOM loading '%s' — evicting cold unit '%s' and retrying",
                    model_id,
                    candidate,
                )
                # candidate may be a shared base key — _evict_target drops all its
                # heads so the worker (and its VRAM) is actually reclaimed.
                await self._evict_target(candidate)

                # Force-clear partial load so next attempt doesn't hit "already loaded"
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: (
                            self._manager.unload_shared_base(model_id)
                            if base_preload_resolution is not None
                            else self._manager.unload(model_id)
                        ),
                    )
                except Exception:
                    # unload failed — force-remove from backends dict
                    self._manager._backends.pop(model_id, None)
                    self._preloaded_shared_bases.pop(model_id, None)

        # Exhausted retries — should not normally reach here
        logger.error("MMP: load '%s' exhausted %d retries", model_id, max_retries)
        self._fail_load(model_id, fs, _ERR_LOAD_FAILED)

    def _stub_load(self, model_id: str, fs: Optional[ModelState]) -> None:
        """Mark model as stub-loaded (no real backend). Flushes waiters."""
        logger.info("MMP: '%s' stub-loaded (no real model)", model_id)
        if fs is None:
            fs = self._models.setdefault(model_id, ModelState())
        fs.loading = False
        fs.loaded = True
        self._flush_load_waiters(model_id)

    def _fail_load(
        self, model_id: str, fs: Optional[ModelState], err_code: int
    ) -> None:
        """Clean up ModelState and notify waiters with T_ERROR on load failure."""
        if fs is None:
            fs = self._models.get(model_id)
        if fs is not None:
            waiters, fs.load_waiters = fs.load_waiters, []
            fs.loading = False
            fs.loaded = False
            for identity, req_id, _ in waiters:
                asyncio.create_task(
                    self._send(identity, T_ERROR, struct.pack(">QB", req_id, err_code))
                )
            admin, fs.admin_waiters = fs.admin_waiters, []
            for identity, req_id in admin:
                asyncio.create_task(
                    self._send(identity, T_ERROR, struct.pack(">QB", req_id, err_code))
                )
        # Clean up any partial state
        self._backends.pop(model_id, None)
        if model_id in self._preloaded_shared_bases and self._manager is not None:
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.run_in_executor(
                    None, lambda: self._manager.unload_shared_base(model_id)
                )
            else:
                self._manager.unload_shared_base(model_id)
        self._preloaded_shared_bases.pop(model_id, None)
        self._model_access.pop(model_id, None)
        self._model_request_times.pop(model_id, None)

    def _flush_load_waiters(self, model_id: str) -> None:
        """Notify all T_ENSURE_LOADED waiters for this model_id.

        Must be called on the event loop thread.
        """
        fs = self._models.get(model_id)
        if not fs:
            return
        waiters, fs.load_waiters = fs.load_waiters, []
        now = time.monotonic()
        for identity, req_id, deadline in waiters:
            if now <= deadline:
                asyncio.create_task(
                    self._send(identity, T_MODEL_READY, struct.pack(">Q", req_id))
                )
            else:
                asyncio.create_task(
                    self._send(identity, T_LOAD_TIMEOUT, struct.pack(">QI", req_id, 1))
                )
        admin, fs.admin_waiters = fs.admin_waiters, []
        for identity, req_id in admin:
            self._spawn(self._send(identity, T_OK, struct.pack(">Q", req_id)))

    async def _expire_waiter(
        self, model_id: str, identity: bytes, req_id: int, deadline: float
    ) -> None:
        """Send T_LOAD_TIMEOUT if this waiter is still pending at its deadline."""
        delay = deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        fs = self._models.get(model_id)
        if fs is None:
            return
        remaining = [
            w for w in fs.load_waiters if not (w[0] == identity and w[1] == req_id)
        ]
        if len(remaining) == len(fs.load_waiters):
            return  # already flushed by _flush_load_waiters / _fail_load
        fs.load_waiters = remaining
        await self._send(identity, T_LOAD_TIMEOUT, struct.pack(">QI", req_id, 1))

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
        now_ns = time.monotonic_ns()
        slot_to_req = {s: r for r, (_, s, _) in self._pending.items()}
        for slot_id in stale:
            if slot_id in self._inflight:
                # A worker holds a live ticket for this slot (slow batch or
                # cancelled request retained for the worker) — never free it
                # under the worker; it would be re-allocated mid-batch.
                continue
            req_id = slot_to_req.get(slot_id)
            hdr = None
            try:
                hdr = self._pool.read_header(slot_id)
                age_s = (now_ns - hdr.timestamp_ns) / 1e9 if hdr.timestamp_ns else 0.0
            except Exception:
                age_s = 0.0
            logger.warning(
                "MMP: reaping stale slot slot_id=%d age_s=%.1f req_id=%s",
                slot_id,
                age_s,
                req_id,
            )
            if req_id is not None:
                pending = self._pending.pop(req_id, None)
                if pending:
                    identity, _, _ = pending
                    asyncio.create_task(
                        self._send(
                            identity, T_ERROR, struct.pack(">QB", req_id, _ERR_STALE)
                        )
                    )
            # Free under the request the header was bound to when sampled — if
            # the request completed (and the slot was rebound) between
            # stale_slots() and here, the free is a no-op.
            if hdr is not None:
                self._pool.free_slot(slot_id, request_id=hdr.request_id)
            else:
                self._pool.free_slot(slot_id)

    # ------------------------------------------------------------------
    # Eviction loop
    # ------------------------------------------------------------------

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(self._evict_check_interval_s)
            try:
                await self._check_and_evict()
            except Exception:
                logger.exception("MMP: error in eviction loop")

    async def _check_and_evict(self) -> None:
        """Evict cold models if GPU memory exceeds threshold.

        Cold-first: models idle > idle_timeout_s are evicted first (oldest first).
        If all models are hot (recent traffic), no eviction — _ERR_SERVER_FULL
        will be returned by _load_model when it can't free space.

        Uses drain_and_unload (graceful: stop accepting, finish in-flight, then kill).
        Next request for evicted model triggers _load_model which reloads from disk cache.
        """
        if self._manager is None:
            return
        loop = asyncio.get_running_loop()
        gpu_frac = await loop.run_in_executor(None, _gpu_used_fraction)
        if gpu_frac < self._evict_threshold:
            return
        candidate = self._pick_eviction_candidate()
        if candidate is None:
            logger.warning(
                "MMP: GPU %.0f%% > %.0f%% threshold but all models are hot — no eviction",
                gpu_frac * 100,
                self._evict_threshold * 100,
            )
            return
        logger.warning(
            "MMP: GPU %.0f%% > %.0f%% threshold — evicting cold unit '%s'",
            gpu_frac * 100,
            self._evict_threshold * 100,
            candidate,
        )
        # candidate may be a base key — _evict_target drops all its heads so the
        # worker's VRAM is actually reclaimed (_evict_model would KeyError on it).
        await self._evict_target(candidate)

    def _unload_blocking(self, model_id: str) -> bool:
        """Drain + unload via ModelManager. BLOCKING (up to drain timeout) —
        call only from an executor thread."""
        try:
            self._manager.unload(model_id, drain=True)
            return True
        except Exception:
            logger.warning("MMP: eviction of '%s' failed", model_id, exc_info=True)
            return False

    async def _evict_model(self, model_id: str) -> bool:
        """Evict one model. State marks on the event loop, drain in executor.

        Marks the victim (loaded=False, _unloading) BEFORE draining so
        concurrent submit routing, ENSURE_LOADED, and eviction pickers skip
        it — prevents the double-unload of the same victim.
        """
        fs = self._models.get(model_id)
        if fs:
            fs.loaded = False
        self._unloading.add(model_id)
        victim = self._backends.get(model_id)
        try:
            ok = await asyncio.get_running_loop().run_in_executor(
                None, self._unload_blocking, model_id
            )
        finally:
            self._unloading.discard(model_id)
        if ok:
            cur = self._backends.pop(model_id, None)
            if cur is not None and cur is not victim:
                # A load that was already in flight registered a fresh backend
                # mid-drain — keep it, only the drained victim is gone.
                self._backends[model_id] = cur
            else:
                self._untrack_shared_head(model_id)
                self._model_access.pop(model_id, None)
                self._model_request_times.pop(model_id, None)
        self._kick_pending_load(model_id)
        return ok

    # ------------------------------------------------------------------
    # VRAM-aware admission control
    # ------------------------------------------------------------------

    def _record_vram_samples(self, samples: dict) -> None:
        """Append measured per-model GPU MB into each model's sliding window."""
        for flavor, mb in samples.items():
            window = self._vram_window.get(flavor)
            if window is None:
                window = deque(maxlen=self._vram_window_size)
                self._vram_window[flavor] = window
            window.append(mb)

    def _footprint_mb(self, flavor: str) -> float:
        """Resolved VRAM footprint of a loaded model or shared base worker.

        Measured peak (max of sliding window) wins; falls back to the static
        per-batch figure from MemoryProfile; 0 when neither is known.

        A shared head reports 0 — its VRAM lives in the shared base worker and is
        reclaimed only by tearing the whole worker down (keyed by base_key), never by
        dropping one head.
        """
        if flavor in self._head_base_key:
            return 0.0
        window = self._vram_window.get(flavor)
        if window:
            return max(window)
        return self._vram_meta_cache.get(flavor, 0)

    def _fetch_vram_mb(self, model_id: str, api_key: str, batch: int) -> Optional[int]:
        """Metadata-only Roboflow lookup: max static VRAM across packages (MB).

        Returns 0 when no MemoryProfile data exists, None on fetch failure
        (so transient failures are not cached).
        """
        try:
            from inference_models.weights_providers.roboflow import get_model_metadata

            meta = get_model_metadata(model_id, api_key or None)
            values = []
            for pkg in meta.model_packages:
                profile = getattr(pkg, "memory_profile", None)
                if profile is None:
                    continue
                vram = profile.vram_for_batch(batch)
                if vram:
                    values.append(vram)
            return max(values) if values else 0
        except Exception:
            logger.debug(
                "MMP: VRAM metadata fetch failed for '%s'", model_id, exc_info=True
            )
            return None

    def _vram_metadata_model_id(self, model_id: str) -> str:
        """Strip multi-instance routing suffix for Roboflow metadata lookup."""
        return model_id.rsplit(":", 1)[0]

    def _required_mb(self, model_id: str, api_key: str = "", batch: int = 1) -> int:
        """VRAM needed to admit an incoming model (MB). Cached on success;
        transient fetch failures (None) are NOT cached so the next load retries."""
        metadata_model_id = self._vram_metadata_model_id(model_id)
        if metadata_model_id in self._vram_meta_cache:
            return self._vram_meta_cache[metadata_model_id]
        mb = self._fetch_vram_mb(metadata_model_id, api_key, batch)
        if mb is None:
            return 0
        self._vram_meta_cache[metadata_model_id] = mb
        return mb

    async def _enforce_vram_headroom_after_load(
        self,
        model_id: str,
        fs: Optional[ModelState],
        *,
        is_shared_base_preload: bool = False,
        exclude: Optional[set] = None,
    ) -> bool:
        """Rollback a successful load if it violated the configured free-VRAM floor."""
        if not self._vram_admission or self._vram_headroom_mb <= 0:
            return True
        free_mb = await asyncio.get_running_loop().run_in_executor(
            None, self._gpu_free_mb
        )
        if free_mb is None or free_mb >= self._vram_headroom_mb:
            return True
        deficit = self._vram_headroom_mb - free_mb
        exclude = set(exclude or ())
        exclude.add(model_id)
        plan = self._plan_evictions(deficit, exclude=exclude)
        if plan is not None:
            logger.warning(
                "MMP: load '%s' left %.0f MB free below %.0f MB headroom — "
                "evicting %s",
                model_id,
                free_mb,
                self._vram_headroom_mb,
                plan,
            )
            if await self._execute_eviction_plan(
                plan, deficit, exclude=exclude
            ):
                free_mb = await asyncio.get_running_loop().run_in_executor(
                    None, self._gpu_free_mb
                )
                if free_mb is None or free_mb >= self._vram_headroom_mb:
                    return True

        logger.warning(
            "MMP: load '%s' left %.0f MB free below %.0f MB headroom — unloading",
            model_id,
            free_mb,
            self._vram_headroom_mb,
        )
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: (
                    self._manager.unload_shared_base(model_id)
                    if is_shared_base_preload
                    else self._manager.unload(model_id)
                ),
            )
        except Exception:
            logger.warning(
                "MMP: failed to rollback '%s' after VRAM headroom breach",
                model_id,
                exc_info=True,
            )
            if is_shared_base_preload:
                self._manager._shared_base_preloads.pop(model_id, None)
            else:
                self._manager._backends.pop(model_id, None)
        self._preloaded_shared_bases.pop(model_id, None)
        self._fail_load(model_id, fs, _ERR_SERVER_FULL)
        return False

    def _recent_count(self, flavor: str, window_s: float) -> int:
        """Number of inferences for a model within the last ``window_s`` seconds."""
        times = self._model_request_times.get(flavor)
        if not times:
            return 0
        cutoff = time.monotonic() - window_s
        return sum(1 for t in times if t >= cutoff)

    def _gpu_free_mb(self) -> Optional[float]:
        """Free GPU 0 memory in MB via NVML. None if telemetry unavailable
        (overridable in tests)."""
        info = _nvml_mem_info()
        if info is None:
            return None
        return info.free / 1024 / 1024

    def _is_shared_head(self, flavor: str) -> bool:
        return flavor in self._head_base_key

    def _is_base_key(self, key: str) -> bool:
        return key in self._shared_heads

    def _is_preloaded_base_key(self, key: str) -> bool:
        return key in self._preloaded_shared_bases.values()

    def _evictable_units(self) -> list:
        """Eviction units: normal models (excluding shared heads) plus shared base
        keys. Evicting a base key reclaims its whole worker's VRAM; a single head
        reclaims nothing, so heads are never units."""
        units = [f for f in self._models if not self._is_shared_head(f)]
        units.extend(self._shared_heads.keys())
        return units

    def _flavor_loaded_idle(
        self, flavor: str, now: float, in_flight: set, cutoff_s: float
    ) -> bool:
        fs = self._models.get(flavor)
        return bool(
            fs
            and fs.loaded
            and not fs.pinned
            and flavor not in in_flight
            and flavor not in self._unloading
            and (now - self._model_access.get(flavor, 0.0)) > cutoff_s
        )

    def _unit_evictable(
        self, key: str, now: float, in_flight: set, cutoff_s: float
    ) -> bool:
        if self._is_base_key(key):
            if self._is_preloaded_base_key(key):
                return False
            heads = self._shared_heads.get(key, set())
            return bool(heads) and all(
                self._flavor_loaded_idle(h, now, in_flight, cutoff_s) for h in heads
            )
        return self._flavor_loaded_idle(key, now, in_flight, cutoff_s)

    def _unit_access(self, key: str) -> float:
        if self._is_base_key(key):
            heads = self._shared_heads.get(key, set())
            return max((self._model_access.get(h, 0.0) for h in heads), default=0.0)
        return self._model_access.get(key, 0.0)

    def _unit_recent(self, key: str, window_s: float) -> int:
        if self._is_base_key(key):
            return sum(
                self._recent_count(h, window_s)
                for h in self._shared_heads.get(key, set())
            )
        return self._recent_count(key, window_s)

    async def _evict_target(self, key: str) -> bool:
        """Evict one unit. A base key drops every head (the last drop reaps the worker
        and frees its VRAM); a normal model unloads directly."""
        if self._is_base_key(key):
            ok = True
            for head in list(self._shared_heads.get(key, set())):
                ok = await self._evict_model(head) and ok
            return ok
        return await self._evict_model(key)

    def _plan_evictions(
        self, deficit_mb: float, exclude: Optional[set] = None
    ) -> Optional[list]:
        """Pick coldest eviction units whose combined footprint covers deficit_mb.

        Excludes hot (idle <= cutoff), in-flight, and already-unloading units, plus any
        in ``exclude`` (e.g. the base a loading head depends on). A shared base key is
        evictable only when ALL its heads are cold/idle/free. Coldest-first by (oldest
        last access, fewest recent inferences). Returns the ordered victim list, or
        None if the cold set cannot cover it.
        """
        now = time.monotonic()
        in_flight = {mid for _, _, mid in self._pending.values()}
        exclude = exclude or set()
        units = [
            u
            for u in self._evictable_units()
            if u not in exclude
            and self._unit_evictable(u, now, in_flight, self._vram_idle_cutoff_s)
        ]
        units.sort(
            key=lambda u: (
                self._unit_access(u),
                self._unit_recent(u, self._vram_recent_window_s),
            )
        )
        plan = []
        secured = 0.0
        for unit in units:
            plan.append(unit)
            secured += self._footprint_mb(unit)
            if secured >= deficit_mb:
                return plan
        return None

    def _shared_head_marginal_mb(
        self, resolution: Any, head_model_id: str, api_key: str
    ) -> int:
        """Marginal VRAM of a head joining a RESIDENT base: the whole-model footprint
        minus the base's own footprint. Clamped at 0; the OOM-retry loop backstops any
        under-estimate from noisy metadata."""
        whole = self._required_mb(head_model_id, api_key)
        base = self._required_mb(resolution.dep_model_id, api_key)
        return max(0, whole - base)

    async def _vram_admission_plan(
        self,
        model_id: str,
        api_key: str = "",
        need_mb: Optional[int] = None,
        exclude: Optional[set] = None,
    ) -> tuple:
        """Decide admission for an incoming model without mutating state.

        Returns (decision, victims, deficit_mb); decision is "admit" | "evict"
        | "no_capacity". ``need_mb`` overrides the metadata footprint (used to admit a
        shared head on its marginal cost); ``exclude`` keeps planning from evicting the
        base a loading head depends on. Missing footprint data (need == 0) still
        enforces the headroom floor; GPU telemetry unavailable admits and lets the
        OOM-retry loop in _load_model backstop.
        Metadata fetch (HTTP) and GPU probe run in the executor, never on the loop.
        """
        loop = asyncio.get_running_loop()
        if need_mb is not None:
            need = need_mb
        else:
            need = await loop.run_in_executor(
                None, self._required_mb, model_id, api_key
            )
        free_mb = await loop.run_in_executor(None, self._gpu_free_mb)
        if free_mb is None:
            return "admit", [], 0.0
        free = free_mb - self._vram_headroom_mb
        if need == 0:
            if free >= 0:
                return "admit", [], 0.0
            deficit = -free
            plan = self._plan_evictions(deficit, exclude=exclude)
            if plan is None:
                return "no_capacity", [], deficit
            return "evict", plan, deficit
        if free >= need:
            return "admit", [], 0.0
        deficit = need - free
        plan = self._plan_evictions(deficit, exclude=exclude)
        if plan is None:
            return "no_capacity", [], deficit
        return "evict", plan, deficit

    async def _execute_eviction_plan(
        self, plan: list, deficit_mb: float, exclude: Optional[set] = None
    ) -> bool:
        """Unload victims one at a time, re-validating each step (revocable).

        If a planned victim turned hot/in-flight, drop it and elect a replacement
        covering the remaining deficit (honouring ``exclude``). If no replacement
        exists, abort (leaving already-unloaded victims unloaded) and return False.
        Returns True once secured >= deficit_mb.
        """
        secured = 0.0
        queue = list(plan)
        while secured < deficit_mb and queue:
            victim = queue.pop(0)
            now = time.monotonic()
            in_flight = {mid for _, _, mid in self._pending.values()}
            if not self._unit_evictable(
                victim, now, in_flight, self._vram_idle_cutoff_s
            ):
                replacement = self._plan_evictions(
                    deficit_mb - secured, exclude=exclude
                )
                if replacement is None:
                    return False
                queue = replacement
                continue
            footprint = self._footprint_mb(victim)
            ok = await self._evict_target(victim)
            if ok:
                secured += footprint
        return secured >= deficit_mb

    def _pick_eviction_candidate(self, exclude: Optional[set] = None) -> Optional[str]:
        """Pick best eviction unit (cold-first, LRU). ``exclude`` skips units a caller
        must not evict (e.g. the base a loading head depends on).

        Returns None if no evictable unit (all hot and in-flight, or nothing loaded).
        """
        now = time.monotonic()
        in_flight = {mid for _, _, mid in self._pending.values()}
        # Cold eviction units (a shared base key counts only when all its heads are
        # cold). ``exclude`` protects a base a loading head still needs. Oldest last
        # access first; all-hot → None.
        exclude = exclude or set()
        cold = [
            u
            for u in self._evictable_units()
            if u not in exclude
            and self._unit_evictable(u, now, in_flight, self._idle_timeout_s)
        ]
        if not cold:
            return None
        return min(cold, key=self._unit_access)

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
                if self._vram_admission:
                    self._record_vram_samples(
                        self._stats_snapshot.get("per_model_gpu_mb", {})
                    )
            except Exception:
                logger.exception("MMP: monitoring loop error")

    def _collect_stats(self) -> dict:
        """Collect system + manager + GPU metrics. Runs in executor (may block)."""
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
            s["process_rss_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1)
            s["system_cpu_pct"] = psutil.cpu_percent()
            s["system_ram_pct"] = psutil.virtual_memory().percent
        except Exception:
            pass

        # Build pid→flavor map from registered backends for per-model GPU attribution.
        pid_to_flavor: dict[int, str] = {}
        for model_id, backend in list(self._backends.items()):
            pid = getattr(backend, "worker_pid", None)
            if pid is not None:
                pid_to_flavor[pid] = model_id

        # Shared heads report worker_pid=None — their base worker's whole VRAM
        # (base + every head, one process) is attributed to the base_key instead.
        if self._manager is not None:
            for base_key, owner in self._manager.shared_owners().items():
                pid = getattr(owner, "worker_pid", None)
                if pid is not None:
                    pid_to_flavor[pid] = base_key

        gpu_stats = _collect_gpu_stats(pid_to_flavor)
        s.update(gpu_stats)

        return s

    # ------------------------------------------------------------------
    # ZMQ send helper
    # ------------------------------------------------------------------

    def _spawn(self, coro, name: str = "") -> asyncio.Task:
        """create_task with a strong reference (fire-and-forget tasks can be
        GC'd mid-flight otherwise)."""
        task = asyncio.create_task(coro, name=name or None)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def _send(self, identity: bytes, msg_type: bytes, payload: bytes) -> bool:
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
        "--n-slots",
        type=int,
        default=cfg.MMP_N_SLOTS_DEFAULT,
    )
    parser.add_argument(
        "--input-mb",
        type=float,
        default=cfg.MMP_INPUT_MB_DEFAULT,
    )
    parser.add_argument(
        "--addr", default=None, help="ZMQ bind address (default: platform auto)"
    )
    parser.add_argument(
        "--evict-threshold", type=float, default=cfg.INFERENCE_GPU_EVICTION_THRESHOLD
    )
    args = parser.parse_args()

    mmp = ModelManagerProcess(
        n_slots=args.n_slots,
        input_mb=args.input_mb,
        evict_threshold=args.evict_threshold,
    )
    asyncio.run(mmp.run(addr=args.addr))


if __name__ == "__main__":
    main()
