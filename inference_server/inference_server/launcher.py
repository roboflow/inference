"""Inference stack wiring.

Two launch paths:

  launch_inprocess()     → ModelManager
      Caller loads backends directly via mm.load(..., backend="direct"|"subprocess").
      Inference via mm.infer_sync() / mm.submit() / mm.infer_async().
      One model per ModelManager; no ZMQ involved.

  launch_orchestrated()  → LaunchHandle
      Starts ModelManagerProcess (MMP) in a background daemon thread.
      MMP owns a ModelManager (subprocess backends only) + ZMQ ROUTER + SHMPool.
      FastAPI workers connect as ZMQ DEALER clients and share the SHMPool:
          worker allocs slot → writes image bytes → T_SUBMIT → MMP routes to backend
          backend infers → writes result to slot → MMP sends T_RESULT_READY to worker
      handle.mmp_addr  — address FastAPI DEALER workers connect() to
      handle.shm_name  — SHMPool name FastAPI workers attach to

Select via env var:  INFERENCE_DEPLOYMENT_MODE=bundled|mmp
or pass mode= to launch().
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional, Union

from inference_model_manager import configuration as mmp_config
from inference_model_manager.model_manager import ModelManager
from inference_server import configuration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LaunchHandle — returned by launch_orchestrated()
# ---------------------------------------------------------------------------


class LaunchHandle:
    """Handle to a running ModelManagerProcess.

    Attributes:
        manager:  ModelManager the MMP is wrapping.  Lifecycle (load/unload/
                  sleep/wake) can be driven either via this object directly or
                  via ZMQ T_LOAD / T_UNLOAD messages to mmp_addr.
        mmp:      The ModelManagerProcess instance.
        mmp_addr: ZMQ address DEALER clients connect() to.
        shm_name: SHMPool shared-memory name clients attach to.
    """

    def __init__(
        self,
        manager: ModelManager,
        mmp,  # ModelManagerProcess (imported lazily)
        mmp_addr: str,
        shm_name: str,
        _thread: threading.Thread,
    ) -> None:
        self.manager = manager
        self.mmp = mmp
        self.mmp_addr = mmp_addr
        self.shm_name = shm_name
        self._thread = _thread

    def __repr__(self) -> str:
        return (
            f"LaunchHandle(mmp_addr={self.mmp_addr!r}, " f"shm_name={self.shm_name!r})"
        )

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop MMP and shut down the ModelManager."""
        self.mmp.stop()
        self._thread.join(timeout=timeout)
        self.manager.shutdown()


# ---------------------------------------------------------------------------
# launch_inprocess
# ---------------------------------------------------------------------------


def launch_inprocess(
    *,
    max_pinned_memory_mb: int = 0,
) -> ModelManager:
    """Return a ModelManager for in-process use.

    Example::

        mm = launch_inprocess()
        mm.load("yolov8n-640", api_key=key, backend="subprocess")
        result = mm.infer_sync("yolov8n-640", image_bytes)
    """
    return ModelManager(max_pinned_memory_mb=max_pinned_memory_mb)


# ---------------------------------------------------------------------------
# launch_orchestrated
# ---------------------------------------------------------------------------


def launch_orchestrated(
    *,
    max_pinned_memory_mb: int = mmp_config.INFERENCE_MAX_PINNED_MEMORY_MB,
    n_slots: int = mmp_config.MMP_N_SLOTS_DEFAULT,
    input_mb: float = mmp_config.MMP_INPUT_MB_DEFAULT,
    mmp_addr: Optional[str] = None,
    gpu_eviction_threshold: float = mmp_config.INFERENCE_GPU_EVICTION_THRESHOLD,
    evict_check_interval_s: float = mmp_config.INFERENCE_EVICT_CHECK_INTERVAL_S,
    monitor_interval_s: float = mmp_config.INFERENCE_MONITOR_INTERVAL_S,
    stale_reap_interval_s: float = mmp_config.INFERENCE_STALE_REAP_INTERVAL_S,
    stale_slot_max_age_s: float = mmp_config.INFERENCE_STALE_SLOT_MAX_AGE_S,
    load_oom_max_evictions: int = mmp_config.INFERENCE_LOAD_OOM_MAX_EVICTIONS,
    mmp_start_timeout: float = mmp_config.INFERENCE_MMP_START_TIMEOUT_S,
    decoder: str = "imagecodecs",
    batch_max_size: int = 0,
    batch_max_wait_ms: float = 5.0,
    idle_timeout_s: float = mmp_config.INFERENCE_MODEL_IDLE_TIMEOUT_S,
) -> LaunchHandle:
    """Start a ModelManagerProcess and return a LaunchHandle.

    The MMP runs in a background daemon thread.  Call handle.shutdown() to
    stop it gracefully.

    Example (server startup)::

        handle = launch_orchestrated(n_slots=128, input_mb=20)
        # then in each uvicorn worker:
        #   ctx  = zmq.asyncio.Context()
        #   sock = ctx.socket(zmq.DEALER)
        #   sock.connect(handle.mmp_addr)
        #   pool = SHMPool.attach(handle.shm_name, n_slots=128, ...)
    """
    from inference_model_manager.backends.utils.transport import zmq_addr as _zmq_addr
    from inference_model_manager.model_manager_process import ModelManagerProcess

    bind_addr = mmp_addr or _zmq_addr("mmprocess")

    mmp = ModelManagerProcess(
        n_slots=n_slots,
        input_mb=input_mb,
        max_pinned_memory_mb=max_pinned_memory_mb,
        evict_threshold=gpu_eviction_threshold,
        evict_check_interval_s=evict_check_interval_s,
        monitor_interval_s=monitor_interval_s,
        stale_reap_interval_s=stale_reap_interval_s,
        stale_slot_max_age_s=stale_slot_max_age_s,
        load_oom_max_evictions=load_oom_max_evictions,
        decoder=decoder,
        batch_max_size=batch_max_size,
        batch_max_wait_ms=batch_max_wait_ms,
        idle_timeout_s=idle_timeout_s,
        vram_admission=mmp_config.INFERENCE_VRAM_ADMISSION_CONTROL,
        vram_window_size=mmp_config.INFERENCE_VRAM_WINDOW_SIZE,
        vram_idle_cutoff_s=mmp_config.INFERENCE_VRAM_IDLE_CUTOFF_S,
        vram_headroom_mb=mmp_config.INFERENCE_VRAM_HEADROOM_MB,
        vram_recent_window_s=mmp_config.INFERENCE_VRAM_RECENT_WINDOW_S,
    )

    ready = threading.Event()
    thread = threading.Thread(
        target=lambda: asyncio.run(mmp.run(addr=bind_addr, ready_event=ready)),
        daemon=True,
        name="mmp-main",
    )
    thread.start()

    if not ready.wait(timeout=mmp_start_timeout):
        mmp.stop()
        thread.join(timeout=5)
        raise RuntimeError(
            f"ModelManagerProcess did not start within {mmp_start_timeout}s"
        )

    shm_name = mmp.shm_name
    logger.info("MMP started: addr=%s  shm=%s  slots=%d", bind_addr, shm_name, n_slots)

    return LaunchHandle(
        manager=mmp.manager,
        mmp=mmp,
        mmp_addr=bind_addr,
        shm_name=shm_name,
        _thread=thread,
    )


# ---------------------------------------------------------------------------
# launch — unified entry point
# ---------------------------------------------------------------------------


def launch(mode: str, **kwargs) -> Union[ModelManager, LaunchHandle]:
    """Wire up the inference stack.

    Mode is required — callers read env at their own boundary (e.g.
    `app.py` lifespan, `server.main`) so this function stays testable
    without env manipulation.

    Args:
        mode: ``configuration.MODE_BUNDLED`` — return a ModelManager for
            in-process use (Workflows, InferencePipeline, dev).
            ``configuration.MODE_MMP`` — start a ModelManagerProcess in a
            daemon thread, return a LaunchHandle exposing its ZMQ address
            and SHM name (used by `server.main` to wire uvicorn workers).
        **kwargs: Forwarded to :func:`launch_inprocess` or
                  :func:`launch_orchestrated`.
    """
    if mode == configuration.MODE_MMP:
        return launch_orchestrated(**kwargs)
    if mode == configuration.MODE_BUNDLED:
        return launch_inprocess(
            max_pinned_memory_mb=kwargs.get("max_pinned_memory_mb", 0),
        )
    raise ValueError(
        f"Unknown deployment mode {mode!r}. "
        f"Choose {configuration.MODE_BUNDLED!r} or {configuration.MODE_MMP!r}."
    )
