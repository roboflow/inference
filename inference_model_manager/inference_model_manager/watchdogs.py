"""Background daemons for disk cache purging and CUDA memory reclamation.

Both watchdogs are fixed-interval daemon threads sharing the ``_IntervalDaemon``
base: ``InferenceModelsCacheWatchdog`` purges the on-disk inference-models cache
when it exceeds a size budget, and ``CudaMemoryReclamationWatchdog`` returns
cached-but-unused CUDA blocks to the driver on a timer. ``start_enabled_watchdogs``
reads ``configuration.py`` and starts whichever of the two are enabled.
"""

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from filelock import FileLock

from inference_model_manager import configuration as cfg

logger = logging.getLogger(__name__)

SHARED_BLOBS_DIR = "shared-blobs"
MODELS_CACHE_DIR = "models-cache"
BYTES_IN_MB = 1024 * 1024
SECONDS_IN_DAY = 60 * 60 * 24
SECONDS_IN_MIN = 60
LOCK_POSTFIX = ".lock"
RECENT_THRESHOLD_DAYS = 1
WARM_THRESHOLD_DAYS = 7
STALE_THRESHOLD_DAYS = 30
MIN_PURGE_INTERVAL_MINUTES = 15
MIN_RECLAMATION_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class FileInfo:
    path: str
    size_mb: float
    modified_at: datetime


class StalenessGroup(Enum):
    ABANDONED = 0
    STALE = 1
    WARM = 2
    RECENT = 3


class _IntervalDaemon(ABC):
    """Daemon thread running ``_cycle()`` on a fixed interval until stopped."""

    def __init__(self, name: str, interval_seconds: float, min_interval_seconds: float):
        if interval_seconds < min_interval_seconds:
            logger.warning(
                f"Requested {name} interval {interval_seconds}s is below minimum "
                f"{min_interval_seconds}s - falling back to minimum."
            )
            interval_seconds = min_interval_seconds
        self._name = name
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @abstractmethod
    def _cycle(self) -> None:
        """Run a single watchdog cycle."""

    def _can_start(self) -> bool:
        """Override to add a pre-flight check, run after the already-running
        guard and before the daemon thread is spawned."""
        return True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"{self._name} daemon is already running")
            return
        if not self._can_start():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=self._name,
        )
        self._thread.start()
        logger.info(
            f"{self._name} daemon started - interval: {self._interval_seconds}s"
        )

    def stop(self, timeout: Optional[float] = None) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info(f"{self._name} daemon stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                self._cycle()
            except Exception as e:
                logger.error(f"{self._name} cycle failed: {e}", exc_info=True)
            elapsed = time.monotonic() - cycle_start
            remaining = self._interval_seconds - elapsed
            if remaining <= 0:
                logger.warning(
                    f"{self._name} cycle took {elapsed}s, exceeding interval "
                    f"of {self._interval_seconds}s - skipping next cycle"
                )
                remaining = self._interval_seconds
            self._stop_event.wait(timeout=remaining)


class InferenceModelsCacheWatchdog(_IntervalDaemon):

    def __init__(
        self,
        inference_home: str,
        max_cache_size_mb: int,
        interval_minutes: float,
    ):
        super().__init__(
            name="cache-purge-daemon",
            interval_seconds=interval_minutes * SECONDS_IN_MIN,
            min_interval_seconds=MIN_PURGE_INTERVAL_MINUTES * SECONDS_IN_MIN,
        )
        self._inference_home = inference_home
        self._max_cache_size_mb = max_cache_size_mb

    def _cycle(self) -> None:
        purge_inference_models_cache(
            inference_home=self._inference_home,
            max_cache_size_mb=self._max_cache_size_mb,
        )


def purge_inference_models_cache(
    inference_home: str,
    max_cache_size_mb: int,
) -> None:
    directories_to_investigate = [
        os.path.abspath(os.path.join(inference_home, SHARED_BLOBS_DIR)),
        os.path.abspath(os.path.join(inference_home, MODELS_CACHE_DIR)),
    ]
    cache_index = build_current_cache_index(
        directories_to_investigate=directories_to_investigate
    )
    current_cache_size = summarize_disk_size(files_info=cache_index)
    if current_cache_size <= max_cache_size_mb:
        logger.info(
            f"Purging inference models cache skipped - current {round(current_cache_size, 2)}MB, "
            f"limit: {round(max_cache_size_mb, 2)}MB"
        )
        return None
    to_be_reclaimed = current_cache_size - max_cache_size_mb
    cache_index_ranked = rank_for_deletion(files=cache_index)
    nominated_for_deletion = nominate_files_for_deletion(
        files=cache_index_ranked, to_be_reclaimed=to_be_reclaimed
    )
    purged = purge_files(files=nominated_for_deletion)
    if purged < to_be_reclaimed:
        logger.warning(
            "Could not fully purge inference-models cache - expected size to be reclaimed: "
            f"{round(to_be_reclaimed, 2)}MB, actual reclaimed: {round(purged, 2)}MB"
        )
    else:
        logger.info(f"Purge complete - reclaimed {round(purged, 2)}MB.")
    return None


def build_current_cache_index(
    directories_to_investigate: List[str],
) -> List[FileInfo]:
    results = []
    for directory_path in directories_to_investigate:
        results.extend(list_files(path=directory_path))
    return results


def list_files(path: str) -> List[FileInfo]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return []
    if os.path.islink(path):
        return []
    if os.path.isfile(path):
        if path.endswith(LOCK_POSTFIX):
            return []
        stat = os.stat(path)
        return [
            FileInfo(
                path=path,
                size_mb=stat.st_size / BYTES_IN_MB,
                modified_at=datetime.fromtimestamp(stat.st_mtime),
            )
        ]
    results = []
    for directory_path, directory_names, file_names in os.walk(path, followlinks=False):
        for file_name in file_names:
            file_path = os.path.join(directory_path, file_name)
            if os.path.islink(file_path) or file_path.endswith(LOCK_POSTFIX):
                continue
            try:
                stat = os.stat(file_path)
            except OSError:
                continue
            results.append(
                FileInfo(
                    path=file_path,
                    size_mb=stat.st_size / BYTES_IN_MB,
                    modified_at=datetime.fromtimestamp(stat.st_mtime),
                )
            )
    return results


def summarize_disk_size(files_info: List[FileInfo]) -> float:
    result = 0
    for file_info in files_info:
        result += file_info.size_mb
    return result


def rank_for_deletion(
    files: List[FileInfo],
    now: Optional[datetime] = None,
    recent_threshold_days: float = RECENT_THRESHOLD_DAYS,
    warm_threshold_days: float = WARM_THRESHOLD_DAYS,
    stale_threshold_days: float = STALE_THRESHOLD_DAYS,
) -> List[FileInfo]:
    if not files:
        return []
    if now is None:
        now = datetime.now()

    if not (recent_threshold_days < warm_threshold_days < stale_threshold_days):
        raise ValueError(
            f"Thresholds must be in ascending order: "
            f"recent ({recent_threshold_days}) < warm ({warm_threshold_days}) < stale ({stale_threshold_days})"
        )

    def staleness_group(file_info: FileInfo) -> int:
        age_days = (now - file_info.modified_at).total_seconds() / SECONDS_IN_DAY
        if age_days > stale_threshold_days:
            return StalenessGroup.ABANDONED.value
        if age_days > warm_threshold_days:
            return StalenessGroup.STALE.value
        if age_days > recent_threshold_days:
            return StalenessGroup.WARM.value
        return StalenessGroup.RECENT.value

    return sorted(files, key=lambda f: (staleness_group(f), -f.size_mb))


def nominate_files_for_deletion(
    files: List[FileInfo], to_be_reclaimed: float
) -> List[FileInfo]:
    reclaimed = 0
    to_delete = []
    for file in files:
        if reclaimed >= to_be_reclaimed:
            break
        to_delete.append(file)
        reclaimed += file.size_mb
    return to_delete


def purge_files(files: List[FileInfo], file_lock_acquire_timeout: int = 3) -> float:
    result = 0
    for file in files:
        try:
            file_absolute_path = os.path.abspath(file.path)
            file_directory = os.path.dirname(file_absolute_path)
            file_name = os.path.basename(file_absolute_path)
            lock_path = os.path.join(file_directory, f".{file_name}{LOCK_POSTFIX}")
            with FileLock(lock_path, timeout=file_lock_acquire_timeout):
                os.remove(file.path)
            result += file.size_mb
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to purge cache file {file.path}: {e}")
    return result


def cuda_is_available() -> bool:
    """Return True only if torch is importable *and* a CUDA device is present.

    Used to short-circuit the watchdog before it ever starts: on a CPU-only or
    torch-less deployment there is nothing to reclaim, so we must not spin a daemon
    that wakes up every interval only to no-op.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception as error:
        logger.warning(
            f"Could not determine CUDA availability for the memory reclamation "
            f"watchdog: {error}"
        )
        return False


def reclaim_cuda_memory() -> None:
    """Return cached-but-unused CUDA blocks to the driver via ``torch.cuda.empty_cache()``.

    PyTorch's CUDA caching allocator keeps freed device blocks in its own pool and
    never returns them to the OS on its own. On a long-running inference server this
    makes the high-water mark of concurrent/batched inference sticky - reserved VRAM
    only ever grows. This call releases the *unused* portion of that pool back to the
    driver. Live allocations are unaffected, so it is safe to call at any time (only
    the reclaimable slack is freed).
    """
    if not cuda_is_available():
        return None
    try:
        import torch

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


class CudaMemoryReclamationWatchdog(_IntervalDaemon):
    """Daemon thread that periodically returns cached CUDA memory to the driver.

    Shares the ``_IntervalDaemon`` base with ``InferenceModelsCacheWatchdog``: a
    background daemon thread running a fixed-interval loop. It exists because the
    only in-process CUDA reclamation calls (``try_releasing_cuda_memory`` on model
    eviction, and the memory-pressure check in ``ModelManager``) fire only on
    model-lifecycle events - never on the inference hot path - so a server serving
    a fixed set of already-loaded models under load never gives reserved VRAM back.

    Disabled by default; enable and tune via the
    ``ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG`` and
    ``CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS`` environment variables.
    """

    def __init__(self, interval_seconds: float):
        super().__init__(
            name="cuda-memory-reclamation-daemon",
            interval_seconds=interval_seconds,
            min_interval_seconds=MIN_RECLAMATION_INTERVAL_SECONDS,
        )

    def _can_start(self) -> bool:
        if not cuda_is_available():
            logger.info(
                "CUDA memory reclamation watchdog was enabled but no CUDA device is "
                "available (torch missing or torch.cuda.is_available() is False) - "
                "the daemon will not start, so it never wakes up to do nothing."
            )
            return False
        return True

    def _cycle(self) -> None:
        reclaim_cuda_memory()


def start_enabled_watchdogs() -> list:
    """Start every watchdog its config enables; returns the started daemons
    (caller stops them on shutdown)."""
    started = []
    if cfg.MAX_INFERENCE_MODELS_CACHE_SIZE_MB > 0:
        from inference_models.configuration import INFERENCE_HOME

        d = InferenceModelsCacheWatchdog(
            inference_home=INFERENCE_HOME,
            max_cache_size_mb=cfg.MAX_INFERENCE_MODELS_CACHE_SIZE_MB,
            interval_minutes=cfg.INFERENCE_MODELS_CACHE_WATCHDOG_INTERVAL_MINUTES,
        )
        d.start()
        started.append(d)
    if cfg.ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG:
        d = CudaMemoryReclamationWatchdog(
            interval_seconds=cfg.CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS
        )
        d.start()
        started.append(d)
    return started
