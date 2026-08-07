import hashlib
import os
import queue
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, Dict, Optional, cast
from uuid import uuid4

from inference_models.configuration import (
    MODEL_BLOB_CACHE_ACCESS_KEY_ID,
    MODEL_BLOB_CACHE_ADDRESSING_STYLE,
    MODEL_BLOB_CACHE_BUCKET,
    MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS,
    MODEL_BLOB_CACHE_COOLDOWN_SECONDS,
    MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS,
    MODEL_BLOB_CACHE_ENABLED,
    MODEL_BLOB_CACHE_ENDPOINT_URL,
    MODEL_BLOB_CACHE_FAILURE_THRESHOLD,
    MODEL_BLOB_CACHE_PREFIX,
    MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS,
    MODEL_BLOB_CACHE_REGION,
    MODEL_BLOB_CACHE_SECRET_ACCESS_KEY,
)
from inference_models.logger import LOGGER
from inference_models.utils.file_system import remove_file_if_exists

_STREAM_CHUNK_SIZE = 1024 * 1024
_RESTORE_WORKERS = 8
_RESTORE_QUEUE_SIZE = 0
_UPLOAD_WORKERS = 2
_UPLOAD_QUEUE_SIZE = 8
_SUPPORTED_ADDRESSING_STYLES = {"auto", "path", "virtual"}


@dataclass(frozen=True)
class ModelBlobCacheConfig:
    bucket: str
    prefix: str = "model-blobs"
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    addressing_style: str = "auto"
    connect_timeout_seconds: float = 1.0
    read_timeout_seconds: float = 2.0
    download_timeout_seconds: float = 30.0
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0


class _BoundedDaemonExecutor:
    """A small non-blocking work queue that cannot delay interpreter shutdown."""

    def __init__(
        self, max_workers: int, max_pending: int, thread_name_prefix: str
    ) -> None:
        self._tasks: queue.Queue = queue.Queue()
        self._available_slots = threading.BoundedSemaphore(
            value=max_workers + max_pending
        )
        for worker_index in range(max_workers):
            worker = threading.Thread(
                target=self._work,
                name=f"{thread_name_prefix}-{worker_index}",
                daemon=True,
            )
            worker.start()

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
                # Tasks are expected to handle their own exceptions. This guard keeps a
                # future programming error from terminating a queue worker.
                LOGGER.exception("Unexpected model blob cache background task error")
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
        now = monotonic()
        with self._lock:
            if now >= self._open_until:
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


def _remove_staging_file_best_effort(staging_path: str) -> None:
    try:
        remove_file_if_exists(path=staging_path)
    except Exception as error:
        LOGGER.warning(
            "Could not clean up model blob cache staging file %s: %s",
            staging_path,
            error,
        )


class _RestoreAttempt:
    """Coordinate staging-file ownership between a worker and its caller."""

    def __init__(self, staging_path: str) -> None:
        self.completed = threading.Event()
        self.cancelled = threading.Event()
        self.outcome = "failure"
        self._staging_path = staging_path
        self._ownership_lock = threading.Lock()

    def finish_worker(self, outcome: str) -> None:
        """Publish the result, then clean staging when the worker owns it."""
        with self._ownership_lock:
            if self.cancelled.is_set():
                worker_owns_staging = True
            else:
                self.outcome = outcome
                worker_owns_staging = outcome != "hit"
                self.completed.set()
        if worker_owns_staging:
            _remove_staging_file_best_effort(staging_path=self._staging_path)

    def cancel_if_pending(self) -> bool:
        """Claim timeout ownership unless the worker already published completion."""
        with self._ownership_lock:
            if self.completed.is_set():
                return False
            self.cancelled.set()
            return True


class S3ModelBlobCache:
    """Best-effort content-addressed cache backed by an S3-compatible service."""

    def __init__(
        self,
        client: Any,
        config: ModelBlobCacheConfig,
        upload_executor: Optional[_BoundedDaemonExecutor] = None,
        restore_executor: Optional[_BoundedDaemonExecutor] = None,
    ) -> None:
        self._client = client
        self._config = config
        self._upload_executor = (
            upload_executor
            if upload_executor is not None
            else _BoundedDaemonExecutor(
                max_workers=_UPLOAD_WORKERS,
                max_pending=_UPLOAD_QUEUE_SIZE,
                thread_name_prefix="model-blob-cache-upload",
            )
        )
        self._restore_executor = (
            restore_executor
            if restore_executor is not None
            else _BoundedDaemonExecutor(
                max_workers=_RESTORE_WORKERS,
                max_pending=_RESTORE_QUEUE_SIZE,
                thread_name_prefix="model-blob-cache-download",
            )
        )
        self._read_circuit = _CircuitBreaker(
            failure_threshold=config.failure_threshold,
            cooldown_seconds=config.cooldown_seconds,
        )
        self._write_circuit = _CircuitBreaker(
            failure_threshold=config.failure_threshold,
            cooldown_seconds=config.cooldown_seconds,
        )

    def restore(self, content_hash: str, target_path: str) -> bool:
        """Restore a verified blob, returning False on misses and all cache errors.

        The S3 operation runs in a bounded daemon worker pool so the caller observes a
        hard overall deadline even if an SDK or network stack does not honor its socket
        timeout. The worker writes to a cache-specific temporary path; a timed-out
        worker can therefore never race with the source download or poison its output.
        """
        if self._read_circuit.is_open():
            return False

        staging_path = f"{target_path}.model-blob-cache-{uuid4()}"
        attempt = _RestoreAttempt(staging_path=staging_path)
        try:
            submitted = self._restore_executor.submit(
                self._restore_to_staging_path,
                content_hash,
                staging_path,
                attempt,
            )
        except Exception as error:
            self._read_circuit.record_failure()
            LOGGER.warning(
                "Could not start model blob cache read; using the original model "
                "source: %s",
                error,
            )
            return False
        if not submitted:
            LOGGER.warning(
                "Model blob cache download workers are saturated; using the original "
                "model source"
            )
            return False
        completed_in_time = attempt.completed.wait(
            self._config.download_timeout_seconds
        )
        if not completed_in_time and attempt.cancel_if_pending():
            self._read_circuit.record_failure()
            LOGGER.warning(
                "Model blob cache read exceeded %.2fs; using the original model source",
                self._config.download_timeout_seconds,
            )
            return False

        outcome = attempt.outcome
        if outcome == "hit":
            try:
                os.replace(staging_path, target_path)
                self._read_circuit.record_success()
                return True
            except Exception as error:
                _remove_staging_file_best_effort(staging_path=staging_path)
                self._read_circuit.record_failure()
                LOGGER.warning(
                    "Could not promote model blob cache download; using the original "
                    "model source: %s",
                    error,
                )
                return False

        if outcome == "miss":
            self._read_circuit.record_success()
        else:
            self._read_circuit.record_failure()
        return False

    def schedule_store(self, content_hash: str, source_path: str) -> bool:
        """Schedule an upload without blocking; drop it when the queue is full."""
        if self._write_circuit.is_open():
            return False
        submitted = self._upload_executor.submit(
            self._store_best_effort, content_hash, source_path
        )
        if not submitted:
            LOGGER.warning("Model blob cache upload queue is full; dropping upload")
        return submitted

    def _restore_to_staging_path(
        self,
        content_hash: str,
        staging_path: str,
        attempt: _RestoreAttempt,
    ) -> None:
        body = None
        outcome = "failure"
        try:
            if attempt.cancelled.is_set():
                return
            response = self._client.get_object(
                Bucket=self._config.bucket,
                Key=self._object_key(content_hash),
            )
            if attempt.cancelled.is_set():
                return
            body = response["Body"]
            computed_hash = hashlib.md5()
            os.makedirs(os.path.dirname(os.path.abspath(staging_path)), exist_ok=True)
            with open(staging_path, "wb") as target_file:
                while not attempt.cancelled.is_set():
                    chunk = body.read(_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    target_file.write(chunk)
                    computed_hash.update(chunk)
            if attempt.cancelled.is_set():
                return
            if computed_hash.hexdigest().lower() != content_hash.lower():
                LOGGER.warning(
                    "Model blob cache returned content with an invalid MD5 hash; "
                    "using the original model source"
                )
                return
            outcome = "hit"
        except Exception as error:
            if _is_missing_object_error(error):
                outcome = "miss"
                LOGGER.debug("Model blob cache miss for %s", content_hash)
            else:
                LOGGER.warning(
                    "Model blob cache read failed; using the original model source: %s",
                    error,
                )
        finally:
            if body is not None:
                try:
                    body.close()
                except Exception:
                    pass
            attempt.finish_worker(outcome)

    def _store_best_effort(self, content_hash: str, source_path: str) -> None:
        try:
            from boto3.s3.transfer import TransferConfig

            self._client.upload_file(
                source_path,
                self._config.bucket,
                self._object_key(content_hash),
                Config=TransferConfig(use_threads=False),
            )
            self._write_circuit.record_success()
        except Exception as error:
            self._write_circuit.record_failure()
            LOGGER.warning("Model blob cache upload failed: %s", error)

    def _object_key(self, content_hash: str) -> str:
        prefix = self._config.prefix.strip("/")
        normalized_hash = content_hash.lower()
        return f"{prefix}/{normalized_hash}" if prefix else normalized_hash


def _is_missing_object_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    error_details = response.get("Error", {})
    return str(error_details.get("Code")) in {"404", "NoSuchKey", "NotFound"}


def _build_s3_client(config: ModelBlobCacheConfig) -> Any:
    import boto3
    from botocore.config import Config

    client_config = Config(
        connect_timeout=config.connect_timeout_seconds,
        read_timeout=config.read_timeout_seconds,
        retries={"total_max_attempts": 1, "mode": "standard"},
        s3={"addressing_style": config.addressing_style},
    )
    client_kwargs: Dict[str, Any] = {"config": client_config}
    if config.endpoint_url:
        client_kwargs["endpoint_url"] = config.endpoint_url
    if config.region:
        client_kwargs["region_name"] = config.region
    if config.access_key_id and config.secret_access_key:
        client_kwargs["aws_access_key_id"] = config.access_key_id
        client_kwargs["aws_secret_access_key"] = config.secret_access_key
    return boto3.client("s3", **client_kwargs)


def _configuration_from_environment() -> ModelBlobCacheConfig:
    if not MODEL_BLOB_CACHE_BUCKET:
        raise ValueError(
            "MODEL_BLOB_CACHE_BUCKET must be set when the cache is enabled"
        )
    if MODEL_BLOB_CACHE_ADDRESSING_STYLE not in _SUPPORTED_ADDRESSING_STYLES:
        raise ValueError(
            "MODEL_BLOB_CACHE_ADDRESSING_STYLE must be one of: auto, path, virtual"
        )
    credentials = (
        MODEL_BLOB_CACHE_ACCESS_KEY_ID,
        MODEL_BLOB_CACHE_SECRET_ACCESS_KEY,
    )
    if any(credentials) and not all(credentials):
        raise ValueError(
            "MODEL_BLOB_CACHE_ACCESS_KEY_ID and "
            "MODEL_BLOB_CACHE_SECRET_ACCESS_KEY must be set together"
        )
    if MODEL_BLOB_CACHE_FAILURE_THRESHOLD < 1:
        raise ValueError("MODEL_BLOB_CACHE_FAILURE_THRESHOLD must be at least 1")
    for name, value in (
        (
            "MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS",
            MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS,
        ),
        (
            "MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS",
            MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS,
        ),
        (
            "MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS",
            MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS,
        ),
        ("MODEL_BLOB_CACHE_COOLDOWN_SECONDS", MODEL_BLOB_CACHE_COOLDOWN_SECONDS),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    return ModelBlobCacheConfig(
        bucket=MODEL_BLOB_CACHE_BUCKET,
        prefix=MODEL_BLOB_CACHE_PREFIX,
        endpoint_url=MODEL_BLOB_CACHE_ENDPOINT_URL,
        region=MODEL_BLOB_CACHE_REGION,
        access_key_id=MODEL_BLOB_CACHE_ACCESS_KEY_ID,
        secret_access_key=MODEL_BLOB_CACHE_SECRET_ACCESS_KEY,
        addressing_style=MODEL_BLOB_CACHE_ADDRESSING_STYLE,
        connect_timeout_seconds=MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds=MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS,
        download_timeout_seconds=MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS,
        failure_threshold=MODEL_BLOB_CACHE_FAILURE_THRESHOLD,
        cooldown_seconds=MODEL_BLOB_CACHE_COOLDOWN_SECONDS,
    )


def _initialize_model_blob_cache() -> Optional[S3ModelBlobCache]:
    if not MODEL_BLOB_CACHE_ENABLED:
        return None
    try:
        config = _configuration_from_environment()
        return S3ModelBlobCache(client=_build_s3_client(config), config=config)
    except Exception as error:
        LOGGER.warning(
            "Could not initialize model blob cache; using original model sources: %s",
            error,
        )
        return None


_MODEL_BLOB_CACHE_UNINITIALIZED = object()
_model_blob_cache_instance: object = _MODEL_BLOB_CACHE_UNINITIALIZED
_model_blob_cache_initialization_lock = threading.Lock()


def get_model_blob_cache() -> Optional[S3ModelBlobCache]:
    global _model_blob_cache_instance

    if _model_blob_cache_instance is _MODEL_BLOB_CACHE_UNINITIALIZED:
        with _model_blob_cache_initialization_lock:
            if _model_blob_cache_instance is _MODEL_BLOB_CACHE_UNINITIALIZED:
                _model_blob_cache_instance = _initialize_model_blob_cache()
    return cast(Optional[S3ModelBlobCache], _model_blob_cache_instance)
