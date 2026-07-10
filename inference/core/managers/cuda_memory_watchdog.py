import threading
import time
from typing import Optional

from inference.core import logger

BYTES_IN_MB = 1024 * 1024
MIN_RECLAMATION_INTERVAL_SECONDS = 5.0


def reclaim_cuda_memory() -> None:
    """Return cached-but-unused CUDA blocks to the driver via ``torch.cuda.empty_cache()``.

    PyTorch's CUDA caching allocator keeps freed device blocks in its own pool and
    never returns them to the OS on its own. On a long-running inference server this
    makes the high-water mark of concurrent/batched inference sticky - reserved VRAM
    only ever grows. This call releases the *unused* portion of that pool back to the
    driver. Live allocations are unaffected, so it is safe to call at any time (only
    the reclaimable slack is freed).
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free_before, total = torch.cuda.mem_get_info()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free_after, _ = torch.cuda.mem_get_info()
        reclaimed_mb = max(0, free_after - free_before) / BYTES_IN_MB
        logger.info(
            "CUDA memory reclamation cycle complete - reclaimed %.2fMB "
            "(free: %.2fMB -> %.2fMB of %.2fMB total)",
            reclaimed_mb,
            free_before / BYTES_IN_MB,
            free_after / BYTES_IN_MB,
            total / BYTES_IN_MB,
        )
    except Exception as error:
        logger.warning(
            f"Attempted to reclaim CUDA memory but failed with error: {error}"
        )
    return None


class CudaMemoryReclamationWatchdog:
    """Daemon thread that periodically returns cached CUDA memory to the driver.

    Mirrors :class:`InferenceModelsCacheWatchdog` in nature: a background daemon
    thread running a fixed-interval loop. It exists because the only in-process CUDA
    reclamation calls (``try_releasing_cuda_memory`` on model eviction, and the
    memory-pressure check in ``WithFixedSizeCache``) fire only on model-lifecycle
    events - never on the inference hot path - so a server serving a fixed set of
    already-loaded models under load never gives reserved VRAM back.

    Disabled by default; enable and tune via the
    ``ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG`` and
    ``CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS`` environment variables.
    """

    def __init__(self, interval_seconds: float):
        if interval_seconds < MIN_RECLAMATION_INTERVAL_SECONDS:
            logger.warning(
                f"Requested CUDA memory reclamation interval {interval_seconds}s is below "
                f"minimum {MIN_RECLAMATION_INTERVAL_SECONDS}s - falling back to minimum."
            )
            interval_seconds = MIN_RECLAMATION_INTERVAL_SECONDS
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("CUDA memory reclamation daemon is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="cuda-memory-reclamation-daemon",
        )
        self._thread.start()
        logger.info(
            f"CUDA memory reclamation daemon started - interval: {self._interval_seconds}s"
        )

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("CUDA memory reclamation daemon stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                reclaim_cuda_memory()
            except Exception as e:
                logger.error(
                    f"CUDA memory reclamation cycle failed: {e}", exc_info=True
                )
            elapsed = time.monotonic() - cycle_start
            remaining = self._interval_seconds - elapsed
            if remaining <= 0:
                remaining = self._interval_seconds
            self._stop_event.wait(timeout=remaining)
