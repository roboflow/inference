import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from unittest import mock

from inference_models.utils import content_addressed_artifact_cache as cache_module
from inference_models.utils.content_addressed_artifact_cache import (
    NullContentAddressedArtifactCache,
    VerifiedContentAddressedArtifactCache,
    _BoundedDaemonExecutor,
)


class _InlineExecutor:
    def submit(self, function, *args) -> bool:
        function(*args)
        return True


def _cache(storage, restore_executor=None, upload_executor=None, **overrides):
    return VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        read_deadline_seconds=overrides.pop("read_deadline_seconds", 1),
        failure_threshold=overrides.pop("failure_threshold", 3),
        cooldown_seconds=overrides.pop("cooldown_seconds", 60),
        restore_executor=restore_executor or _InlineExecutor(),
        upload_executor=upload_executor or _InlineExecutor(),
        **overrides,
    )


def _write_download(content: bytes):
    def download(_: str, target_path: str) -> bool:
        with open(target_path, "wb") as target_file:
            target_file.write(content)
        return True

    return download


def test_null_cache_restore_is_a_side_effect_free_miss(tmp_path) -> None:
    target_path = tmp_path / "weights.onnx"

    restored = NullContentAddressedArtifactCache().restore(
        content_hash="c770e3485f6f6cd5bf2f78504bd56c50",
        target_path=str(target_path),
    )

    assert restored is False
    assert not target_path.exists()


def test_null_cache_store_is_a_side_effect_free_drop(tmp_path) -> None:
    source_path = tmp_path / "weights.onnx"
    source_path.write_bytes(b"weights")

    scheduled = NullContentAddressedArtifactCache().schedule_store(
        content_hash="c770e3485f6f6cd5bf2f78504bd56c50",
        source_path=str(source_path),
    )

    assert scheduled is False
    assert source_path.read_bytes() == b"weights"


def test_verified_cache_constructs_normalized_key_and_verifies_download(
    tmp_path,
) -> None:
    content = b"cached weights"
    content_hash = hashlib.md5(content).hexdigest().upper()
    storage = mock.MagicMock()

    def download(blob_key: str, target_path: str) -> bool:
        assert blob_key == f"prefix/{content_hash.lower()}"
        with open(target_path, "wb") as target_file:
            target_file.write(content)
        return True

    storage.download.side_effect = download
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="/prefix/",
        read_deadline_seconds=1,
        failure_threshold=3,
        cooldown_seconds=60,
    )
    target_path = tmp_path / "weights.onnx"

    assert cache.restore(content_hash, str(target_path)) is True
    assert target_path.read_bytes() == content


def test_cache_miss_counts_as_read_success(tmp_path) -> None:
    content_hash = hashlib.md5(b"expected").hexdigest()
    storage = mock.MagicMock()
    storage.download.side_effect = [
        RuntimeError("failed"),
        False,
        RuntimeError("failed"),
        False,
    ]
    cache = _cache(storage, failure_threshold=2)

    assert cache.restore(content_hash, str(tmp_path / "one")) is False
    assert cache.restore(content_hash, str(tmp_path / "two")) is False
    assert cache.restore(content_hash, str(tmp_path / "three")) is False
    assert cache.restore(content_hash, str(tmp_path / "four")) is False

    assert storage.download.call_count == 4


def test_verified_cache_rejects_invalid_md5_without_touching_storage(tmp_path) -> None:
    storage = mock.MagicMock()
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        read_deadline_seconds=1,
        failure_threshold=3,
        cooldown_seconds=60,
    )

    assert cache.restore("not-an-md5", str(tmp_path / "weights")) is False
    assert cache.schedule_store("not-an-md5", str(tmp_path / "source")) is False
    storage.download.assert_not_called()
    storage.upload.assert_not_called()


def test_verified_cache_rejects_corrupt_download_and_cleans_staging(tmp_path) -> None:
    storage = mock.MagicMock()

    def download(_: str, target_path: str) -> bool:
        with open(target_path, "wb") as target_file:
            target_file.write(b"corrupt")
        return True

    storage.download.side_effect = download
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        read_deadline_seconds=1,
        failure_threshold=3,
        cooldown_seconds=60,
    )
    target_path = tmp_path / "weights"

    restored = cache.restore(hashlib.md5(b"expected").hexdigest(), str(target_path))
    assert restored is False
    assert not target_path.exists()
    assert list(tmp_path.glob("weights.artifact-cache-*")) == []


def test_promotion_and_cleanup_failures_remain_fail_open(tmp_path) -> None:
    content = b"cached model weights"
    content_hash = hashlib.md5(content).hexdigest()
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(content)
    cache = _cache(storage, failure_threshold=1)
    target_path = tmp_path / "weights.onnx"
    promotion_error = OSError("promotion failed")
    cleanup_error = OSError("cleanup failed")

    with mock.patch.object(
        cache_module.os, "replace", side_effect=promotion_error
    ) as replace_mock, mock.patch.object(
        cache_module, "remove_file_if_exists", side_effect=cleanup_error
    ) as remove_mock, mock.patch.object(
        cache_module.LOGGER, "warning"
    ) as warning_mock:
        assert cache.restore(content_hash, str(target_path)) is False
        assert cache.restore(content_hash, str(target_path)) is False

    replace_mock.assert_called_once()
    remove_mock.assert_called_once()
    assert storage.download.call_count == 1
    warning_mock.assert_any_call(
        "Could not clean up artifact cache staging file %s: %s",
        mock.ANY,
        cleanup_error,
    )
    warning_mock.assert_any_call(
        "Could not promote artifact cache download: %s", promotion_error
    )


def test_verified_cache_enforces_hard_read_deadline(tmp_path) -> None:
    release = threading.Event()
    storage = mock.MagicMock()

    def download(_: str, __: str) -> bool:
        release.wait()
        return False

    storage.download.side_effect = download
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        read_deadline_seconds=0.02,
        failure_threshold=3,
        cooldown_seconds=60,
        restore_executor=_BoundedDaemonExecutor(1, 0, "test-artifact-deadline"),
    )
    started_at = time.monotonic()
    try:
        restored = cache.restore(
            hashlib.md5(b"expected").hexdigest(), str(tmp_path / "weights")
        )
    finally:
        release.set()

    assert restored is False
    assert time.monotonic() - started_at < 0.2


def test_restore_executor_caps_stuck_reads(tmp_path) -> None:
    release = threading.Event()
    storage = mock.MagicMock()

    def download(_: str, __: str) -> bool:
        release.wait()
        return False

    storage.download.side_effect = download
    cache = _cache(
        storage,
        restore_executor=_BoundedDaemonExecutor(2, 0, "test-artifact-bounded"),
        read_deadline_seconds=0.02,
        failure_threshold=100,
    )
    content_hash = hashlib.md5(b"expected").hexdigest()
    try:
        for index in range(10):
            assert cache.restore(content_hash, str(tmp_path / str(index))) is False
    finally:
        release.set()

    assert storage.download.call_count == 2


def test_timed_out_restore_is_cleaned_by_worker(tmp_path) -> None:
    release = threading.Event()
    download_started = threading.Event()
    storage = mock.MagicMock()

    def download(_: str, target_path: str) -> bool:
        with open(target_path, "wb") as target_file:
            target_file.write(b"partial")
        download_started.set()
        release.wait()
        return True

    storage.download.side_effect = download
    cache = _cache(
        storage,
        restore_executor=_BoundedDaemonExecutor(1, 0, "test-artifact-cleanup"),
        read_deadline_seconds=0.05,
    )
    target_path = tmp_path / "weights"
    try:
        assert (
            cache.restore(hashlib.md5(b"partial").hexdigest(), str(target_path))
            is False
        )
        assert download_started.is_set()
        assert list(tmp_path.glob("weights.artifact-cache-*"))
    finally:
        release.set()

    deadline = time.monotonic() + 1
    while (
        list(tmp_path.glob("weights.artifact-cache-*")) and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert list(tmp_path.glob("weights.artifact-cache-*")) == []


def test_worker_completion_claim_wins_timeout_ownership_race(tmp_path) -> None:
    content = b"cached model weights"
    content_hash = hashlib.md5(content).hexdigest()
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(content)
    cache = _cache(
        storage,
        restore_executor=_BoundedDaemonExecutor(1, 0, "test-artifact-ownership"),
        read_deadline_seconds=0.05,
    )
    target_path = tmp_path / "weights"
    completion_claimed = threading.Event()
    timeout_claim_attempted = threading.Event()
    release_completion = threading.Event()

    class _GatedCompletionEvent(threading.Event):
        def set(self) -> None:
            completion_claimed.set()
            release_completion.wait()
            super().set()

    attempt = cache_module._RestoreAttempt(staging_path="unused-for-hit")
    attempt.completed = _GatedCompletionEvent()
    original_cancel = attempt.cancel_if_pending

    def observed_cancel() -> bool:
        timeout_claim_attempted.set()
        return original_cancel()

    with mock.patch.object(
        cache_module, "_RestoreAttempt", return_value=attempt
    ), mock.patch.object(attempt, "cancel_if_pending", side_effect=observed_cancel):
        with ThreadPoolExecutor(max_workers=1) as callers:
            result = callers.submit(cache.restore, content_hash, str(target_path))
            assert completion_claimed.wait(timeout=1)
            assert timeout_claim_attempted.wait(timeout=1)
            assert not result.done()
            release_completion.set()
            assert result.result(timeout=1) is True

    assert target_path.read_bytes() == content
    assert list(tmp_path.glob("weights.artifact-cache-*")) == []


def test_slow_non_hit_cleanup_has_only_worker_ownership(tmp_path) -> None:
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(b"corrupt")
    worker_prefix = "test-artifact-non-hit-cleanup"
    cache = _cache(
        storage,
        restore_executor=_BoundedDaemonExecutor(1, 0, worker_prefix),
        read_deadline_seconds=0.02,
    )
    target_path = tmp_path / "weights"
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    release_cleanup = threading.Event()
    cleanup_threads = []
    original_remove = cache_module.remove_file_if_exists

    def slow_remove(*, path: str) -> None:
        cleanup_threads.append(threading.current_thread().name)
        if threading.current_thread().name.startswith(worker_prefix):
            cleanup_started.set()
            release_cleanup.wait()
            original_remove(path=path)
            cleanup_finished.set()
            return
        raise AssertionError("caller attempted staging cleanup")

    with mock.patch.object(
        cache_module, "remove_file_if_exists", side_effect=slow_remove
    ):
        with ThreadPoolExecutor(max_workers=1) as callers:
            result = callers.submit(
                cache.restore,
                hashlib.md5(b"expected").hexdigest(),
                str(target_path),
            )
            assert cleanup_started.wait(timeout=1)
            try:
                assert result.result(timeout=0.15) is False
            except FutureTimeoutError:
                raise AssertionError("worker-owned cleanup blocked fallback")
            finally:
                release_cleanup.set()
            assert cleanup_finished.wait(timeout=1)

    assert cleanup_threads == [f"{worker_prefix}-0"]


def test_non_hit_cleanup_exception_does_not_suppress_fallback(tmp_path) -> None:
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(b"corrupt")
    cache = _cache(
        storage,
        restore_executor=_BoundedDaemonExecutor(1, 0, "test-artifact-cleanup-error"),
    )
    cleanup_error = RuntimeError("cleanup failed")
    with mock.patch.object(
        cache_module, "remove_file_if_exists", side_effect=cleanup_error
    ) as remove_mock, mock.patch.object(cache_module.LOGGER, "warning") as warning_mock:
        assert (
            cache.restore(
                hashlib.md5(b"expected").hexdigest(), str(tmp_path / "weights")
            )
            is False
        )

    remove_mock.assert_called_once()
    warning_mock.assert_any_call(
        "Could not clean up artifact cache staging file %s: %s",
        mock.ANY,
        cleanup_error,
    )


def test_verified_cache_validates_source_before_upload(tmp_path) -> None:
    source_path = tmp_path / "weights"
    source_path.write_bytes(b"actual")
    storage = mock.MagicMock()
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        read_deadline_seconds=1,
        failure_threshold=3,
        cooldown_seconds=60,
        upload_executor=_InlineExecutor(),
    )

    assert (
        cache.schedule_store(hashlib.md5(b"different").hexdigest(), str(source_path))
        is True
    )
    storage.upload.assert_not_called()


def test_read_circuit_bypasses_storage_after_repeated_failures(tmp_path) -> None:
    storage = mock.MagicMock()
    storage.download.side_effect = RuntimeError("unavailable")
    cache = _cache(storage, failure_threshold=2)
    content_hash = hashlib.md5(b"expected").hexdigest()

    assert cache.restore(content_hash, str(tmp_path / "one")) is False
    assert cache.restore(content_hash, str(tmp_path / "two")) is False
    assert cache.restore(content_hash, str(tmp_path / "three")) is False

    assert storage.download.call_count == 2


def test_read_failures_do_not_open_write_circuit(tmp_path) -> None:
    content = b"weights"
    content_hash = hashlib.md5(content).hexdigest()
    source_path = tmp_path / "source"
    source_path.write_bytes(content)
    storage = mock.MagicMock()
    storage.download.side_effect = RuntimeError("read failed")
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        read_deadline_seconds=1,
        failure_threshold=2,
        cooldown_seconds=60,
        restore_executor=_InlineExecutor(),
        upload_executor=_InlineExecutor(),
    )

    assert cache.restore(content_hash, str(tmp_path / "one")) is False
    assert cache.restore(content_hash, str(tmp_path / "two")) is False
    assert cache.schedule_store(content_hash, str(source_path)) is True
    storage.upload.assert_called_once()


def test_write_failures_do_not_open_read_circuit(tmp_path) -> None:
    content_hash = hashlib.md5(b"expected").hexdigest()
    source_path = tmp_path / "source"
    source_path.write_bytes(b"wrong")
    storage = mock.MagicMock()
    storage.download.return_value = False
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        read_deadline_seconds=1,
        failure_threshold=2,
        cooldown_seconds=60,
        restore_executor=_InlineExecutor(),
        upload_executor=_InlineExecutor(),
    )

    assert cache.schedule_store(content_hash, str(source_path)) is True
    assert cache.schedule_store(content_hash, str(source_path)) is True
    assert cache.schedule_store(content_hash, str(source_path)) is False
    assert cache.restore(content_hash, str(tmp_path / "target")) is False
    storage.download.assert_called_once()


def test_read_success_does_not_reset_write_failures(tmp_path) -> None:
    content = b"weights"
    content_hash = hashlib.md5(content).hexdigest()
    source_path = tmp_path / "source"
    source_path.write_bytes(content)
    storage = mock.MagicMock()
    storage.upload.side_effect = RuntimeError("write failed")
    storage.download.return_value = False
    cache = _cache(storage, failure_threshold=2)

    assert cache.schedule_store(content_hash, str(source_path)) is True
    assert cache.restore(content_hash, str(tmp_path / "target")) is False
    assert cache.schedule_store(content_hash, str(source_path)) is True
    assert cache.schedule_store(content_hash, str(source_path)) is False

    assert storage.upload.call_count == 2
    storage.download.assert_called_once()


def test_upload_queue_drops_work_when_saturated(tmp_path) -> None:
    content = b"weights"
    source_path = tmp_path / "source"
    source_path.write_bytes(content)
    executor = _BoundedDaemonExecutor(
        max_workers=0,
        max_pending=1,
        thread_name_prefix="test-artifact-upload-saturation",
    )
    cache = _cache(mock.MagicMock(), upload_executor=executor)
    content_hash = hashlib.md5(content).hexdigest()

    assert cache.schedule_store(content_hash, str(source_path)) is True
    assert cache.schedule_store(content_hash, str(source_path)) is False


def test_upload_worker_failure_is_fail_open(tmp_path) -> None:
    content = b"weights"
    source_path = tmp_path / "source"
    source_path.write_bytes(content)
    storage = mock.MagicMock()
    storage.upload.side_effect = RuntimeError("write failed")
    cache = _cache(storage)

    assert (
        cache.schedule_store(hashlib.md5(content).hexdigest(), str(source_path)) is True
    )
    storage.upload.assert_called_once()


def test_missing_upload_source_is_fail_open(tmp_path) -> None:
    storage = mock.MagicMock()
    cache = _cache(storage)

    assert (
        cache.schedule_store(
            hashlib.md5(b"missing").hexdigest(), str(tmp_path / "missing")
        )
        is True
    )
    storage.upload.assert_not_called()
