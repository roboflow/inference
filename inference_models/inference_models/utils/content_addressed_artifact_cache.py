import hashlib
import os
import queue
import re
import threading
from abc import ABC, abstractmethod
from time import monotonic
from typing import Any, Callable, Optional
from uuid import uuid4

from inference_models.logger import LOGGER
from inference_models.utils.blob_storage import BlobStorage
from inference_models.utils.file_system import remove_file_if_exists

_RESTORE_WORKERS = 8
_UPLOAD_WORKERS = 2
_UPLOAD_QUEUE_SIZE = 8
_MD5_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


class ContentAddressedArtifactCache(ABC):
    @abstractmethod
    def restore(self, content_hash: str, target_path: str) -> bool:
        pass

    @abstractmethod
    def schedule_store(self, content_hash: str, source_path: str) -> bool:
        pass


class NullContentAddressedArtifactCache(ContentAddressedArtifactCache):
    def restore(self, content_hash: str, target_path: str) -> bool:
        return False

    def schedule_store(self, content_hash: str, source_path: str) -> bool:
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
    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        with self._lock:
            if monotonic() >= self._open_until:
                self._open_until = 0.0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._open_until = monotonic() + self._cooldown_seconds
                self._consecutive_failures = 0


def _remove_staging_file_best_effort(path: str) -> None:
    try:
        remove_file_if_exists(path=path)
    except Exception as error:
        LOGGER.warning(
            "Could not clean up artifact cache staging file %s: %s", path, error
        )


class _RestoreAttempt:
    def __init__(self, staging_path: str) -> None:
        self.completed = threading.Event()
        self.cancelled = threading.Event()
        self.outcome = "failure"
        self._staging_path = staging_path
        self._ownership_lock = threading.Lock()

    def finish_worker(self, outcome: str) -> None:
        with self._ownership_lock:
            worker_owns_staging = self.cancelled.is_set() or outcome != "hit"
            if not self.cancelled.is_set():
                self.outcome = outcome
                self.completed.set()
        if worker_owns_staging:
            _remove_staging_file_best_effort(self._staging_path)

    def cancel_if_pending(self) -> bool:
        with self._ownership_lock:
            if self.completed.is_set():
                return False
            self.cancelled.set()
            return True


class VerifiedContentAddressedArtifactCache(ContentAddressedArtifactCache):
    """Fail-open, integrity-verifying cache over an untrusted blob transport."""

    def __init__(
        self,
        storage: BlobStorage,
        prefix: str,
        read_deadline_seconds: float,
        failure_threshold: int,
        cooldown_seconds: float,
        restore_executor: Optional[_BoundedDaemonExecutor] = None,
        upload_executor: Optional[_BoundedDaemonExecutor] = None,
    ) -> None:
        self._storage = storage
        self._prefix = prefix
        self._read_deadline_seconds = read_deadline_seconds
        self._restore_executor = restore_executor or _BoundedDaemonExecutor(
            _RESTORE_WORKERS, 0, "artifact-cache-download"
        )
        self._upload_executor = upload_executor or _BoundedDaemonExecutor(
            _UPLOAD_WORKERS, _UPLOAD_QUEUE_SIZE, "artifact-cache-upload"
        )
        self._read_circuit = _CircuitBreaker(failure_threshold, cooldown_seconds)
        self._write_circuit = _CircuitBreaker(failure_threshold, cooldown_seconds)

    def restore(self, content_hash: str, target_path: str) -> bool:
        if not _MD5_PATTERN.fullmatch(content_hash):
            LOGGER.warning("Artifact cache request has an invalid MD5 hash")
            return False
        if self._read_circuit.is_open():
            return False
        staging_path = f"{target_path}.artifact-cache-{uuid4()}"
        attempt = _RestoreAttempt(staging_path)
        try:
            submitted = self._restore_executor.submit(
                self._restore_to_staging, content_hash, staging_path, attempt
            )
        except Exception as error:
            self._read_circuit.record_failure()
            LOGGER.warning("Could not start artifact cache read: %s", error)
            return False
        if not submitted:
            LOGGER.warning("Artifact cache download workers are saturated")
            return False
        if not attempt.completed.wait(self._read_deadline_seconds):
            if attempt.cancel_if_pending():
                self._read_circuit.record_failure()
                LOGGER.warning(
                    "Artifact cache read exceeded %.2fs", self._read_deadline_seconds
                )
                return False
        if attempt.outcome == "hit":
            try:
                os.replace(staging_path, target_path)
                self._read_circuit.record_success()
                return True
            except Exception as error:
                _remove_staging_file_best_effort(staging_path)
                self._read_circuit.record_failure()
                LOGGER.warning("Could not promote artifact cache download: %s", error)
                return False
        if attempt.outcome == "miss":
            self._read_circuit.record_success()
        else:
            self._read_circuit.record_failure()
        return False

    def schedule_store(self, content_hash: str, source_path: str) -> bool:
        if not _MD5_PATTERN.fullmatch(content_hash):
            LOGGER.warning("Artifact cache store has an invalid MD5 hash")
            return False
        if self._write_circuit.is_open():
            return False
        try:
            submitted = self._upload_executor.submit(
                self._store_best_effort, content_hash, source_path
            )
        except Exception as error:
            self._write_circuit.record_failure()
            LOGGER.warning("Could not schedule artifact cache upload: %s", error)
            return False
        if not submitted:
            LOGGER.warning("Artifact cache upload queue is full; dropping upload")
        return submitted

    def _restore_to_staging(
        self, content_hash: str, staging_path: str, attempt: _RestoreAttempt
    ) -> None:
        outcome = "failure"
        try:
            if attempt.cancelled.is_set():
                return
            os.makedirs(os.path.dirname(os.path.abspath(staging_path)), exist_ok=True)
            found = self._storage.download(self._blob_key(content_hash), staging_path)
            if attempt.cancelled.is_set():
                return
            if not found:
                outcome = "miss"
            elif self._md5(staging_path) == content_hash.lower():
                outcome = "hit"
            else:
                LOGGER.warning(
                    "Artifact cache returned content with an invalid MD5 hash"
                )
        except Exception as error:
            LOGGER.warning("Artifact cache read failed: %s", error)
        finally:
            attempt.finish_worker(outcome)

    def _store_best_effort(self, content_hash: str, source_path: str) -> None:
        try:
            if self._md5(source_path) != content_hash.lower():
                raise ValueError("source content does not match its MD5 hash")
            self._storage.upload(self._blob_key(content_hash), source_path)
            self._write_circuit.record_success()
        except Exception as error:
            self._write_circuit.record_failure()
            LOGGER.warning("Artifact cache upload failed: %s", error)

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
