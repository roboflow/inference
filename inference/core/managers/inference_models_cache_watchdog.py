import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from filelock import FileLock

from inference.core import logger

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


class InferenceModelsCacheWatchdog:

    def __init__(
        self,
        inference_home: str,
        max_cache_size_mb: int,
        interval_minutes: float,
    ):
        if interval_minutes < MIN_PURGE_INTERVAL_MINUTES:
            logger.warning(
                f"Requested purge interval {interval_minutes}min is below minimum "
                f"{MIN_PURGE_INTERVAL_MINUTES}min - falling back to minimum"
            )
            interval_minutes = MIN_PURGE_INTERVAL_MINUTES
        self._inference_home = inference_home
        self._max_cache_size_mb = max_cache_size_mb
        self._interval_seconds = interval_minutes * SECONDS_IN_MIN
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Cache purge daemon is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="cache-purge-daemon",
        )
        self._thread.start()
        logger.info(
            f"Cache purge daemon started - interval: {self._interval_seconds}s, "
            f"max cache size: {self._max_cache_size_mb}MB"
        )

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("Cache purge daemon stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                purge_inference_models_cache(
                    inference_home=self._inference_home,
                    max_cache_size_mb=self._max_cache_size_mb,
                )
            except Exception as e:
                logger.error(f"Cache purge cycle failed: {e}", exc_info=True)
            elapsed = time.monotonic() - cycle_start
            remaining = self._interval_seconds - elapsed
            if remaining <= 0:
                logger.warning(
                    f"Cache purge took {elapsed}s, exceeding interval "
                    f"of {self._interval_seconds}s - skipping next cycle"
                )
                remaining = self._interval_seconds
            self._stop_event.wait(timeout=remaining)


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
            lock_path = f"{file.path}{LOCK_POSTFIX}"
            with FileLock(lock_path, timeout=file_lock_acquire_timeout):
                os.remove(file.path)
            result += file.size_mb
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to purge cache file {file.path}: {e}")
    return result
