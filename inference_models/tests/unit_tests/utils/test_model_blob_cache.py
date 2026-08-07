import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from unittest import mock

from inference_models.utils import model_blob_cache as model_blob_cache_module
from inference_models.utils.model_blob_cache import (
    ModelBlobCacheConfig,
    S3ModelBlobCache,
    _BoundedDaemonExecutor,
    _build_s3_client,
)


class _StreamingBody:
    def __init__(self, content: bytes) -> None:
        self._content = content
        self._returned_content = False
        self.closed = False

    def read(self, _: int) -> bytes:
        if self._returned_content:
            return b""
        self._returned_content = True
        return self._content

    def close(self) -> None:
        self.closed = True


class _MissingObjectError(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class _InlineExecutor:
    def submit(self, function, *args) -> bool:
        function(*args)
        return True


def _cache(
    client, restore_executor=None, upload_executor=None, **overrides
) -> S3ModelBlobCache:
    config = ModelBlobCacheConfig(bucket="models", **overrides)
    return S3ModelBlobCache(
        client=client,
        config=config,
        upload_executor=upload_executor or mock.MagicMock(),
        restore_executor=restore_executor or _InlineExecutor(),
    )


def test_restore_returns_verified_cache_hit(tmp_path) -> None:
    content = b"cached model weights"
    content_hash = hashlib.md5(content).hexdigest()
    body = _StreamingBody(content)
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": body}
    target_path = tmp_path / "weights.onnx"

    restored = _cache(client).restore(content_hash, str(target_path))

    assert restored is True
    assert target_path.read_bytes() == content
    client.get_object.assert_called_once_with(
        Bucket="models", Key=f"model-blobs/{content_hash}"
    )
    assert body.closed is True


def test_promotion_and_cleanup_failures_remain_fail_open(tmp_path) -> None:
    content = b"cached model weights"
    content_hash = hashlib.md5(content).hexdigest()
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _StreamingBody(content)}
    cache = _cache(client, failure_threshold=1, cooldown_seconds=60)
    target_path = tmp_path / "weights.onnx"
    promotion_error = OSError("promotion failed")
    cleanup_error = OSError("cleanup failed")

    with mock.patch.object(
        model_blob_cache_module.os,
        "replace",
        side_effect=promotion_error,
    ) as replace_mock, mock.patch.object(
        model_blob_cache_module,
        "remove_file_if_exists",
        side_effect=cleanup_error,
    ) as remove_mock, mock.patch.object(
        model_blob_cache_module.LOGGER,
        "warning",
    ) as warning_mock:
        restored = cache.restore(content_hash, str(target_path))
        bypassed_after_failure = cache.restore(content_hash, str(target_path))

    assert restored is False
    assert bypassed_after_failure is False
    replace_mock.assert_called_once()
    remove_mock.assert_called_once()
    assert client.get_object.call_count == 1
    warning_mock.assert_any_call(
        "Could not clean up model blob cache staging file %s: %s",
        mock.ANY,
        cleanup_error,
    )
    warning_mock.assert_any_call(
        "Could not promote model blob cache download; using the original "
        "model source: %s",
        promotion_error,
    )


def test_restore_returns_false_for_cache_miss(tmp_path) -> None:
    client = mock.MagicMock()
    client.get_object.side_effect = _MissingObjectError()
    target_path = tmp_path / "weights.onnx"

    restored = _cache(client).restore("abc", str(target_path))

    assert restored is False
    assert not target_path.exists()


def test_restore_rejects_mismatched_content(tmp_path) -> None:
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _StreamingBody(b"corrupt")}
    target_path = tmp_path / "weights.onnx"

    restored = _cache(client).restore(
        hashlib.md5(b"expected").hexdigest(), str(target_path)
    )

    assert restored is False
    assert not target_path.exists()


def test_restore_observes_hard_overall_deadline(tmp_path) -> None:
    release_read = threading.Event()

    class _BlockingBody:
        def read(self, _: int) -> bytes:
            release_read.wait()
            return b""

        def close(self) -> None:
            pass

    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _BlockingBody()}
    restore_executor = _BoundedDaemonExecutor(
        max_workers=1,
        max_pending=0,
        thread_name_prefix="test-model-blob-cache-deadline",
    )
    started_at = time.monotonic()

    try:
        restored = _cache(
            client,
            restore_executor=restore_executor,
            download_timeout_seconds=0.02,
        ).restore(hashlib.md5(b"").hexdigest(), str(tmp_path / "weights.onnx"))
    finally:
        release_read.set()

    elapsed = time.monotonic() - started_at
    assert restored is False
    assert elapsed < 0.2


def test_restore_executor_caps_stuck_reads_and_rejects_additional_work(
    tmp_path,
) -> None:
    release_read = threading.Event()

    class _BlockingBody:
        def read(self, _: int) -> bytes:
            release_read.wait()
            return b""

        def close(self) -> None:
            pass

    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _BlockingBody()}
    restore_executor = _BoundedDaemonExecutor(
        max_workers=2,
        max_pending=0,
        thread_name_prefix="test-model-blob-cache-bounded",
    )
    cache = _cache(
        client,
        restore_executor=restore_executor,
        download_timeout_seconds=0.02,
        failure_threshold=100,
    )

    try:
        for index in range(10):
            assert cache.restore("abc", str(tmp_path / str(index))) is False
    finally:
        release_read.set()

    assert client.get_object.call_count == 2


def test_timed_out_restore_is_cleaned_by_worker_without_cleanup_thread(
    tmp_path,
) -> None:
    release_read = threading.Event()
    second_read_started = threading.Event()

    class _PartiallyBlockingBody:
        def __init__(self) -> None:
            self._first_read = True

        def read(self, _: int) -> bytes:
            if self._first_read:
                self._first_read = False
                return b"partial"
            second_read_started.set()
            release_read.wait()
            return b""

        def close(self) -> None:
            pass

    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _PartiallyBlockingBody()}
    restore_executor = _BoundedDaemonExecutor(
        max_workers=1,
        max_pending=0,
        thread_name_prefix="test-model-blob-cache-cleanup",
    )
    target_path = tmp_path / "weights.onnx"

    try:
        restored = _cache(
            client,
            restore_executor=restore_executor,
            download_timeout_seconds=0.1,
        ).restore("abc", str(target_path))
        staging_paths = list(tmp_path.glob("weights.onnx.model-blob-cache-*"))
        cleanup_threads = [
            thread
            for thread in threading.enumerate()
            if thread.name == "model-blob-cache-cleanup"
        ]
    finally:
        release_read.set()

    cleanup_deadline = time.monotonic() + 1
    while (
        list(tmp_path.glob("weights.onnx.model-blob-cache-*"))
        and time.monotonic() < cleanup_deadline
    ):
        time.sleep(0.01)

    assert restored is False
    assert second_read_started.is_set()
    assert staging_paths
    assert cleanup_threads == []
    assert list(tmp_path.glob("weights.onnx.model-blob-cache-*")) == []


def test_worker_completion_claim_wins_timeout_race_without_orphaning_staging(
    tmp_path,
) -> None:
    content = b"cached model weights"
    content_hash = hashlib.md5(content).hexdigest()
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _StreamingBody(content)}
    restore_executor = _BoundedDaemonExecutor(
        max_workers=1,
        max_pending=0,
        thread_name_prefix="test-model-blob-cache-ownership",
    )
    cache = _cache(
        client,
        restore_executor=restore_executor,
        download_timeout_seconds=0.1,
    )
    target_path = tmp_path / "weights.onnx"
    completion_claimed = threading.Event()
    timeout_claim_attempted = threading.Event()
    release_completion = threading.Event()

    class _GatedCompletionEvent(threading.Event):
        def set(self) -> None:
            completion_claimed.set()
            release_completion.wait()
            super().set()

    attempt = model_blob_cache_module._RestoreAttempt(staging_path="unused-for-hit")
    attempt.completed = _GatedCompletionEvent()
    original_cancel_if_pending = attempt.cancel_if_pending

    def observed_cancel_if_pending() -> bool:
        timeout_claim_attempted.set()
        return original_cancel_if_pending()

    with mock.patch.object(
        model_blob_cache_module,
        "_RestoreAttempt",
        return_value=attempt,
    ), mock.patch.object(
        attempt,
        "cancel_if_pending",
        side_effect=observed_cancel_if_pending,
    ):
        with ThreadPoolExecutor(max_workers=1) as caller_executor:
            restore_result = caller_executor.submit(
                cache.restore, content_hash, str(target_path)
            )
            worker_reached_boundary = completion_claimed.wait(timeout=1)
            caller_reached_timeout = timeout_claim_attempted.wait(timeout=1)
            caller_waited_for_owner = not restore_result.done()
            release_completion.set()
            restored = restore_result.result(timeout=1)

    assert worker_reached_boundary is True
    assert caller_reached_timeout is True
    assert caller_waited_for_owner is True
    assert restored is True
    assert target_path.read_bytes() == content
    assert list(tmp_path.glob("weights.onnx.model-blob-cache-*")) == []


def test_slow_non_hit_cleanup_does_not_block_or_add_a_second_owner(tmp_path) -> None:
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _StreamingBody(b"corrupt")}
    worker_name_prefix = "test-model-blob-cache-non-hit-cleanup"
    restore_executor = _BoundedDaemonExecutor(
        max_workers=1,
        max_pending=0,
        thread_name_prefix=worker_name_prefix,
    )
    cache = _cache(
        client,
        restore_executor=restore_executor,
        download_timeout_seconds=0.02,
    )
    target_path = tmp_path / "weights.onnx"
    worker_cleanup_started = threading.Event()
    worker_cleanup_finished = threading.Event()
    release_worker_cleanup = threading.Event()
    cleanup_call_threads = []
    original_remove_file_if_exists = model_blob_cache_module.remove_file_if_exists

    def fail_on_caller_cleanup(*, path: str) -> None:
        if not path.startswith(f"{target_path}.model-blob-cache-"):
            original_remove_file_if_exists(path=path)
            return
        cleanup_call_threads.append(threading.current_thread().name)
        if threading.current_thread().name.startswith(worker_name_prefix):
            worker_cleanup_started.set()
            release_worker_cleanup.wait()
            original_remove_file_if_exists(path=path)
            worker_cleanup_finished.set()
            return
        raise FileNotFoundError("caller attempted a second staging cleanup")

    with mock.patch.object(
        model_blob_cache_module,
        "remove_file_if_exists",
        side_effect=fail_on_caller_cleanup,
    ):
        with ThreadPoolExecutor(max_workers=1) as caller_executor:
            restore_result = caller_executor.submit(
                cache.restore,
                hashlib.md5(b"expected").hexdigest(),
                str(target_path),
            )
            worker_reached_cleanup = worker_cleanup_started.wait(timeout=1)
            try:
                restored = restore_result.result(timeout=0.15)
                caller_exceeded_deadline = False
            except FutureTimeoutError:
                restored = None
                caller_exceeded_deadline = True
            finally:
                release_worker_cleanup.set()
            if caller_exceeded_deadline:
                restore_result.result(timeout=1)
            cleanup_finished = worker_cleanup_finished.wait(timeout=1)

    assert worker_reached_cleanup is True
    assert caller_exceeded_deadline is False
    assert restored is False
    assert cleanup_finished is True
    assert cleanup_call_threads == [f"{worker_name_prefix}-0"]
    assert list(tmp_path.glob("weights.onnx.model-blob-cache-*")) == []


def test_non_hit_cleanup_exception_does_not_suppress_fallback(tmp_path) -> None:
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": _StreamingBody(b"corrupt")}
    restore_executor = _BoundedDaemonExecutor(
        max_workers=1,
        max_pending=0,
        thread_name_prefix="test-model-blob-cache-cleanup-error",
    )
    cache = _cache(
        client,
        restore_executor=restore_executor,
        download_timeout_seconds=0.5,
    )
    target_path = tmp_path / "weights.onnx"
    cleanup_error = RuntimeError("cleanup failed")
    cleanup_attempted = threading.Event()

    def fail_cleanup(*, path: str) -> None:
        cleanup_attempted.set()
        raise cleanup_error

    with mock.patch.object(
        model_blob_cache_module,
        "remove_file_if_exists",
        side_effect=fail_cleanup,
    ) as remove_mock, mock.patch.object(
        model_blob_cache_module.LOGGER,
        "warning",
    ) as warning_mock, mock.patch.object(
        model_blob_cache_module.LOGGER,
        "exception",
    ) as exception_mock:
        with ThreadPoolExecutor(max_workers=1) as caller_executor:
            restore_result = caller_executor.submit(
                cache.restore,
                hashlib.md5(b"expected").hexdigest(),
                str(target_path),
            )
            cleanup_was_attempted = cleanup_attempted.wait(timeout=1)
            try:
                restored = restore_result.result(timeout=0.15)
                cleanup_suppressed_fallback = False
            except FutureTimeoutError:
                restored = None
                cleanup_suppressed_fallback = True
                restore_result.result(timeout=1)

    assert cleanup_was_attempted is True
    assert cleanup_suppressed_fallback is False
    assert restored is False
    remove_mock.assert_called_once()
    warning_mock.assert_any_call(
        "Could not clean up model blob cache staging file %s: %s",
        mock.ANY,
        cleanup_error,
    )
    exception_mock.assert_not_called()


def test_circuit_breaker_bypasses_reads_after_repeated_failures(tmp_path) -> None:
    client = mock.MagicMock()
    client.get_object.side_effect = RuntimeError("unavailable")
    cache = _cache(client, failure_threshold=2, cooldown_seconds=60)

    assert cache.restore("abc", str(tmp_path / "one")) is False
    assert cache.restore("abc", str(tmp_path / "two")) is False
    assert cache.restore("abc", str(tmp_path / "three")) is False

    assert client.get_object.call_count == 2


def test_read_failures_do_not_open_write_circuit(tmp_path) -> None:
    client = mock.MagicMock()
    client.get_object.side_effect = RuntimeError("read failed")
    upload_executor = mock.MagicMock()
    upload_executor.submit.return_value = True
    cache = _cache(
        client,
        upload_executor=upload_executor,
        failure_threshold=2,
        cooldown_seconds=60,
    )

    assert cache.restore("abc", str(tmp_path / "one")) is False
    assert cache.restore("abc", str(tmp_path / "two")) is False
    assert cache.schedule_store("abc", "/model") is True

    upload_executor.submit.assert_called_once()


def test_write_failures_do_not_open_read_circuit(tmp_path) -> None:
    client = mock.MagicMock()
    client.upload_file.side_effect = RuntimeError("write failed")
    cache = _cache(client, failure_threshold=2, cooldown_seconds=60)

    cache._store_best_effort("abc", "/model")
    cache._store_best_effort("abc", "/model")
    client.get_object.side_effect = _MissingObjectError()

    assert cache.restore("abc", str(tmp_path / "weights.onnx")) is False
    client.get_object.assert_called_once()


def test_read_success_does_not_reset_write_failure_count(tmp_path) -> None:
    client = mock.MagicMock()
    client.upload_file.side_effect = RuntimeError("write failed")
    client.get_object.side_effect = _MissingObjectError()
    upload_executor = mock.MagicMock()
    upload_executor.submit.return_value = True
    cache = _cache(
        client,
        upload_executor=upload_executor,
        failure_threshold=2,
        cooldown_seconds=60,
    )

    cache._store_best_effort("abc", "/model")
    assert cache.restore("abc", str(tmp_path / "weights.onnx")) is False
    cache._store_best_effort("abc", "/model")

    assert cache.schedule_store("abc", "/model") is False
    upload_executor.submit.assert_not_called()


def test_get_model_blob_cache_initializes_once_under_concurrent_load() -> None:
    caller_count = 8
    callers_ready = threading.Barrier(caller_count)
    initialization_started = threading.Event()
    release_initialization = threading.Event()
    initialized_cache = mock.sentinel.initialized_cache

    def initialize():
        initialization_started.set()
        release_initialization.wait()
        return initialized_cache

    def get_cache():
        callers_ready.wait()
        return model_blob_cache_module.get_model_blob_cache()

    with mock.patch.object(
        model_blob_cache_module,
        "_model_blob_cache_instance",
        model_blob_cache_module._MODEL_BLOB_CACHE_UNINITIALIZED,
    ), mock.patch.object(
        model_blob_cache_module,
        "_initialize_model_blob_cache",
        side_effect=initialize,
    ) as initialize_mock:
        with ThreadPoolExecutor(max_workers=caller_count) as executor:
            futures = [executor.submit(get_cache) for _ in range(caller_count)]
            initialization_started_in_time = initialization_started.wait(timeout=1)
            time.sleep(0.02)
            calls_while_initializing = initialize_mock.call_count
            release_initialization.set()
            results = [future.result(timeout=1) for future in futures]

    assert initialization_started_in_time is True
    assert calls_while_initializing == 1
    assert all(result is initialized_cache for result in results)


def test_model_blob_cache_config_uses_fail_fast_default_timeouts() -> None:
    config = ModelBlobCacheConfig(bucket="models")

    assert config.connect_timeout_seconds == 1.0
    assert config.read_timeout_seconds == 2.0
    assert config.download_timeout_seconds == 30.0


def test_s3_client_disables_retries_and_applies_provider_options() -> None:
    config = ModelBlobCacheConfig(
        bucket="models",
        endpoint_url="https://objects.example.com",
        region="region-1",
        access_key_id="access",
        secret_access_key="secret",
        addressing_style="path",
        connect_timeout_seconds=2,
        read_timeout_seconds=10,
    )

    with mock.patch("boto3.client") as client_factory:
        _build_s3_client(config)

    kwargs = client_factory.call_args.kwargs
    assert kwargs["endpoint_url"] == "https://objects.example.com"
    assert kwargs["region_name"] == "region-1"
    assert kwargs["aws_access_key_id"] == "access"
    assert kwargs["aws_secret_access_key"] == "secret"
    assert kwargs["config"].connect_timeout == 2
    assert kwargs["config"].read_timeout == 10
    assert kwargs["config"].retries["total_max_attempts"] == 1
    assert kwargs["config"].s3["addressing_style"] == "path"


def test_upload_failure_is_swallowed() -> None:
    client = mock.MagicMock()
    client.upload_file.side_effect = RuntimeError("write failed")
    cache = _cache(client)

    cache._store_best_effort("abc", "/missing/file")

    transfer_config = client.upload_file.call_args.kwargs["Config"]
    assert transfer_config.use_threads is False


def test_upload_queue_drops_work_when_saturated() -> None:
    executor = _BoundedDaemonExecutor(
        max_workers=0,
        max_pending=1,
        thread_name_prefix="test-model-blob-cache-upload",
    )

    assert executor.submit(lambda: None) is True
    assert executor.submit(lambda: None) is False
