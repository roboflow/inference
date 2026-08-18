import hashlib
import os
import queue
import re
import threading
from abc import ABC, abstractmethod
from time import monotonic
from typing import Any, Callable, Optional

from inference_models.logger import LOGGER
from inference_models.utils.blob_storage import BlobStorage
from inference_models.utils.file_system import remove_file_if_exists

_UPLOAD_WORKERS = 2
_UPLOAD_QUEUE_SIZE = 8
_MD5_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


class ContentAddressedArtifactCache(ABC):
    @abstractmethod
    def restore(self, content_hash: Optional[str], target_path: str) -> bool:
        """Populate `target_path` from the cache, returning False on any miss.

        A falsy `content_hash` identifies content that cannot be addressed and
        is always reported as a miss.
        """

    @abstractmethod
    def schedule_store(self, content_hash: Optional[str], source_path: str) -> bool:
        """Queue `source_path` for upload, returning False when it was dropped.

        Callers must already have verified that `source_path` hashes to
        `content_hash`; the upload does not re-read the file to confirm it.
        """


class NullContentAddressedArtifactCache(ContentAddressedArtifactCache):
    def restore(self, content_hash: Optional[str], target_path: str) -> bool:
        return False

    def schedule_store(self, content_hash: Optional[str], source_path: str) -> bool:
        return False


class _BoundedDaemonExecutor:
    def __init__(
        self, max_workers: int, max_pending: int, thread_name_prefix: str
    ) -> None:
        self._tasks: queue.Queue = queue.Queue()
        self._available_slots = threading.BoundedSemaphore(max_workers + max_pending)
        for index in range(max_workers):
            threading.Thread(
                target=self._work,
                name=f"{thread_name_prefix}-{index}",
                daemon=True,
            ).start()

    def submit(self, function: Callable, *args: Any) -> bool:
        if not self._available_slots.acquire(blocking=False):
            return False
        try:
            self._tasks.put_nowait((function, args))
        except Exception:
            self._available_slots.release()
            raise
        return True

    def _work(self) -> None:
        while True:
            function, args = self._tasks.get()
            try:
                function(*args)
            except Exception:
                LOGGER.exception("Unexpected artifact cache background task error")
            finally:
                self._tasks.task_done()
                self._available_slots.release()


class _CircuitBreaker:
    """Fail-fast gate with a leaky failure count and epoch-tagged outcomes.

    `admit()` hands back the current epoch alongside its go/no-go decision;
    callers report their outcome against that token via `record_success`/
    `record_failure`. A call admitted before a trip can finish long after -
    concurrent calls have wildly different durations - so without the token,
    its outcome would land against whatever state the breaker is in *now*,
    not the state that was true when it started. Tagging every outcome with
    the epoch it was admitted under means a stale completion from before a
    trip can only ever match a stale epoch, so it can neither re-close a
    circuit another call already opened nor pollute the failure count of the
    window that opened it.

    The failure count leaks by one on success rather than resetting to zero.
    With many concurrent, variable-duration calls, completions interleave in
    an order unrelated to when they started, so a strict "N consecutive
    failures" counter can be reset by a single unrelated success arbitrarily
    often and never reach threshold even under a genuinely high failure
    rate. Leaking means an occasional success can no longer erase an entire
    run of failures - the count still tracks overall failure pressure.
    """

    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._failure_count = 0
        self._open_until = 0.0
        self._epoch = 0

    def admit(self) -> Optional[int]:
        """Returns the epoch to report the outcome under, or None if open."""
        with self._lock:
            if monotonic() < self._open_until:
                return None
            self._open_until = 0.0
            return self._epoch

    def record_success(self, epoch: int) -> None:
        with self._lock:
            if epoch != self._epoch:
                return  # Outcome of a call admitted before the last trip.
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, epoch: int) -> None:
        with self._lock:
            if epoch != self._epoch:
                return  # Outcome of a call admitted before the last trip.
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._open_until = monotonic() + self._cooldown_seconds
                self._failure_count = 0
                self._epoch += 1


def _discard_partial_download(path: str) -> None:
    try:
        remove_file_if_exists(path=path)
    except Exception as error:
        LOGGER.warning(
            "Could not clean up partial artifact cache download %s: %s", path, error
        )


class VerifiedContentAddressedArtifactCache(ContentAddressedArtifactCache):
    """Fail-open, integrity-verifying cache over an untrusted blob transport."""

    def __init__(
        self,
        storage: BlobStorage,
        prefix: str,
        failure_threshold: int,
        cooldown_seconds: float,
        upload_executor: Optional[_BoundedDaemonExecutor] = None,
    ) -> None:
        self._storage = storage
        self._prefix = prefix
        self._upload_executor = upload_executor or _BoundedDaemonExecutor(
            _UPLOAD_WORKERS, _UPLOAD_QUEUE_SIZE, "artifact-cache-upload"
        )
        self._read_circuit = _CircuitBreaker(failure_threshold, cooldown_seconds)
        self._write_circuit = _CircuitBreaker(failure_threshold, cooldown_seconds)

    def restore(self, content_hash: Optional[str], target_path: str) -> bool:
        """Download and verify `content_hash` into `target_path`, or report a miss.

        The transfer runs on the calling thread; a stalled or hung read is
        bounded by the storage client's own connect/read timeouts, not by
        this method. `target_path` is left absent unless it holds verified
        content, which lets the caller fall back to the origin freely.
        """
        if not content_hash or not _MD5_PATTERN.fullmatch(content_hash):
            LOGGER.warning("Artifact cache request has an invalid MD5 hash")
            return False
        epoch = self._read_circuit.admit()
        if epoch is None:
            return False
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            found = self._storage.download(self._blob_key(content_hash), target_path)
        except Exception as error:
            _discard_partial_download(target_path)
            self._read_circuit.record_failure(epoch)
            LOGGER.warning("Artifact cache read failed: %s", error)
            return False
        if not found:
            _discard_partial_download(target_path)
            self._read_circuit.record_success(epoch)
            return False
        try:
            verified = self._md5(target_path) == content_hash.lower()
        except Exception as error:
            _discard_partial_download(target_path)
            self._read_circuit.record_failure(epoch)
            LOGGER.warning("Could not verify artifact cache download: %s", error)
            return False
        if not verified:
            _discard_partial_download(target_path)
            self._read_circuit.record_failure(epoch)
            LOGGER.warning("Artifact cache returned content with an invalid MD5 hash")
            return False
        self._read_circuit.record_success(epoch)
        return True

    def schedule_store(self, content_hash: Optional[str], source_path: str) -> bool:
        if not content_hash or not _MD5_PATTERN.fullmatch(content_hash):
            LOGGER.warning("Artifact cache store has an invalid MD5 hash")
            return False
        epoch = self._write_circuit.admit()
        if epoch is None:
            return False
        try:
            submitted = self._upload_executor.submit(
                self._store_best_effort, content_hash, source_path, epoch
            )
        except Exception as error:
            self._write_circuit.record_failure(epoch)
            LOGGER.warning("Could not schedule artifact cache upload: %s", error)
            return False
        if not submitted:
            LOGGER.warning("Artifact cache upload queue is full; dropping upload")
        return submitted

    def _store_best_effort(
        self, content_hash: str, source_path: str, epoch: int
    ) -> None:
        # Uploads are only scheduled for content whose MD5 was verified during
        # the download, and `restore` re-verifies everything it reads back, so
        # re-hashing the whole file here would cost a full read and buy nothing.
        try:
            self._storage.upload(self._blob_key(content_hash), source_path)
        except FileNotFoundError as error:
            # The source can be evicted between scheduling and upload. That is a
            # local race, not a signal about cache health, so the circuit stays shut.
            LOGGER.warning("Artifact cache upload source disappeared: %s", error)
        except Exception as error:
            self._write_circuit.record_failure(epoch)
            LOGGER.warning("Artifact cache upload failed: %s", error)
        else:
            self._write_circuit.record_success(epoch)

    def _blob_key(self, content_hash: str) -> str:
        prefix = self._prefix.strip("/")
        normalized_hash = content_hash.lower()
        return f"{prefix}/{normalized_hash}" if prefix else normalized_hash

    @staticmethod
    def _md5(path: str) -> str:
        digest = hashlib.md5()
        with open(path, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
