import hashlib
from unittest import mock

from inference_models.utils import content_addressed_artifact_cache as cache_module
from inference_models.utils.content_addressed_artifact_cache import (
    NullContentAddressedArtifactCache,
    VerifiedContentAddressedArtifactCache,
    _BoundedDaemonExecutor,
    _CircuitBreaker,
)


class _InlineExecutor:
    def submit(self, function, *args) -> bool:
        function(*args)
        return True


def _cache(storage, upload_executor=None, **overrides):
    storage.exists.return_value = overrides.pop("object_exists", False)
    return VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        failure_threshold=overrides.pop("failure_threshold", 3),
        cooldown_seconds=overrides.pop("cooldown_seconds", 60),
        upload_executor=upload_executor or _InlineExecutor(),
        **overrides,
    )


def _write_download(content: bytes):
    def download(_: str, target_path: str, max_bytes=None) -> bool:
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

    def download(blob_key: str, target_path: str, max_bytes=None) -> bool:
        assert blob_key == f"prefix/{content_hash.lower()}"
        with open(target_path, "wb") as target_file:
            target_file.write(content)
        return True

    storage.download.side_effect = download
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="/prefix/",
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
        failure_threshold=3,
        cooldown_seconds=60,
    )

    assert cache.restore("not-an-md5", str(tmp_path / "weights")) is False
    assert cache.schedule_store("not-an-md5", str(tmp_path / "source")) is False
    storage.download.assert_not_called()
    storage.upload.assert_not_called()


def test_verified_cache_rejects_corrupt_download_and_removes_target(tmp_path) -> None:
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(b"corrupt")
    cache = _cache(storage)
    target_path = tmp_path / "weights"

    restored = cache.restore(hashlib.md5(b"expected").hexdigest(), str(target_path))

    assert restored is False
    # The caller downloads to this path next, so a rejected blob must not survive.
    assert not target_path.exists()


def test_restore_passes_max_object_bytes_through_to_storage(tmp_path) -> None:
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(b"weights")
    cache = _cache(storage, max_object_bytes=123)
    content_hash = hashlib.md5(b"weights").hexdigest()

    assert cache.restore(content_hash, str(tmp_path / "weights")) is True

    assert storage.download.call_args.kwargs["max_bytes"] == 123


def test_download_failure_discards_partial_download(tmp_path) -> None:
    storage = mock.MagicMock()
    target_path = tmp_path / "weights"

    def download(_: str, path: str, max_bytes=None) -> bool:
        with open(path, "wb") as target_file:
            target_file.write(b"partial")
        raise RuntimeError("connection reset")

    storage.download.side_effect = download
    cache = _cache(storage, failure_threshold=1)

    with mock.patch.object(cache_module.LOGGER, "warning") as warning_mock:
        assert (
            cache.restore(hashlib.md5(b"expected").hexdigest(), str(target_path))
            is False
        )

    # The abandoned transfer must leave the caller's download path clean...
    assert not target_path.exists()
    warning_mock.assert_any_call("Artifact cache read failed: %s", mock.ANY)
    # ...and count against the read circuit so a failing endpoint gets bypassed.
    storage.download.reset_mock()
    assert (
        cache.restore(hashlib.md5(b"expected").hexdigest(), str(target_path)) is False
    )
    storage.download.assert_not_called()


def test_cleanup_exception_does_not_suppress_fallback(tmp_path) -> None:
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(b"corrupt")
    cache = _cache(storage)
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
        "Could not clean up partial artifact cache download %s: %s",
        mock.ANY,
        cleanup_error,
    )


def test_verified_cache_uploads_without_re_reading_the_source(tmp_path) -> None:
    source_path = tmp_path / "weights"
    source_path.write_bytes(b"actual")
    content_hash = hashlib.md5(b"different").hexdigest()
    storage = mock.MagicMock()
    cache = _cache(storage)

    assert cache.schedule_store(content_hash, str(source_path)) is True

    # The caller vouches for the hash it verified during the download, so the
    # upload skips a second full-file read; `restore` re-verifies on the way back.
    storage.exists.assert_called_once_with(f"model-blobs/{content_hash}")
    storage.upload.assert_called_once_with(
        f"model-blobs/{content_hash}", str(source_path)
    )


def test_upload_worker_skips_an_existing_blob(tmp_path) -> None:
    source_path = tmp_path / "weights"
    source_path.write_bytes(b"weights")
    content_hash = hashlib.md5(b"weights").hexdigest()
    storage = mock.MagicMock()
    cache = _cache(storage, object_exists=True)

    assert cache.schedule_store(content_hash, str(source_path)) is True

    storage.exists.assert_called_once_with(f"model-blobs/{content_hash}")
    storage.upload.assert_not_called()


def test_upload_worker_replaces_a_corrupt_existing_blob(tmp_path) -> None:
    content = b"weights"
    content_hash = hashlib.md5(content).hexdigest()
    source_path = tmp_path / "source"
    source_path.write_bytes(content)
    storage = mock.MagicMock()
    storage.download.side_effect = _write_download(b"corrupt")
    cache = _cache(storage, object_exists=True)

    assert cache.restore(content_hash, str(tmp_path / "target")) is False
    assert cache.schedule_store(content_hash, str(source_path)) is True

    storage.exists.assert_not_called()
    storage.upload.assert_called_once_with(
        f"model-blobs/{content_hash}", str(source_path)
    )


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
    storage.exists.return_value = False
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        failure_threshold=2,
        cooldown_seconds=60,
        upload_executor=_InlineExecutor(),
    )

    assert cache.restore(content_hash, str(tmp_path / "one")) is False
    assert cache.restore(content_hash, str(tmp_path / "two")) is False
    assert cache.schedule_store(content_hash, str(source_path)) is True
    storage.upload.assert_called_once()


def test_write_failures_do_not_open_read_circuit(tmp_path) -> None:
    content = b"weights"
    content_hash = hashlib.md5(content).hexdigest()
    source_path = tmp_path / "source"
    source_path.write_bytes(content)
    storage = mock.MagicMock()
    storage.upload.side_effect = RuntimeError("write failed")
    storage.download.return_value = False
    storage.exists.return_value = False
    cache = VerifiedContentAddressedArtifactCache(
        storage=storage,
        prefix="model-blobs",
        failure_threshold=2,
        cooldown_seconds=60,
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


def test_missing_upload_source_does_not_open_write_circuit(tmp_path) -> None:
    storage = mock.MagicMock()
    storage.upload.side_effect = FileNotFoundError("source evicted")
    cache = _cache(storage, failure_threshold=1)
    content_hash = hashlib.md5(b"missing").hexdigest()
    missing_path = str(tmp_path / "missing")

    assert cache.schedule_store(content_hash, missing_path) is True
    assert cache.schedule_store(content_hash, missing_path) is True

    # An evicted source is a local race, not a cache-health signal, so a single
    # failure threshold still leaves the write circuit closed for the second call.
    assert storage.upload.call_count == 2


def test_circuit_breaker_leaky_counter_needs_more_than_one_success_to_erase_failures() -> (
    None
):
    breaker = _CircuitBreaker(failure_threshold=3, cooldown_seconds=60)

    # fail, fail, success, fail -> count goes 1, 2, 1, 2: the interleaved
    # success only cancels one failure, not the whole streak, so the circuit
    # is still closed here even though 3 failures have happened overall.
    for _ in range(2):
        breaker.record_failure(breaker.admit())
    breaker.record_success(breaker.admit())
    breaker.record_failure(breaker.admit())
    assert breaker.admit() is not None

    # A 4th failure (count -> 3) is what actually trips it.
    breaker.record_failure(breaker.admit())
    assert breaker.admit() is None


def test_circuit_breaker_single_success_does_not_reset_a_healthy_run() -> None:
    # The old "reset to 0 on any success" behaviour is gone: a lone success
    # among failures must not fully erase accumulated failure pressure.
    breaker = _CircuitBreaker(failure_threshold=2, cooldown_seconds=60)

    breaker.record_failure(breaker.admit())
    breaker.record_success(breaker.admit())  # leaks by 1 (count -> 0), not a reset
    breaker.record_failure(breaker.admit())
    breaker.record_failure(breaker.admit())
    assert breaker.admit() is None


def test_circuit_breaker_ignores_a_stale_success_admitted_before_the_trip() -> None:
    breaker = _CircuitBreaker(failure_threshold=1, cooldown_seconds=60)

    # A slow call is admitted while the circuit is still closed...
    stale_epoch = breaker.admit()
    assert stale_epoch is not None

    # ...then an unrelated, faster call fails and trips the circuit.
    breaker.record_failure(breaker.admit())
    assert breaker.admit() is None

    # The slow call finally finishes as a success, reporting against the
    # epoch it was admitted under - which the trip has since superseded.
    breaker.record_success(stale_epoch)

    # The stale success must not have re-closed the circuit.
    assert breaker.admit() is None


def test_circuit_breaker_ignores_a_stale_failure_from_before_the_trip() -> None:
    breaker = _CircuitBreaker(failure_threshold=2, cooldown_seconds=60)

    stale_epoch = breaker.admit()

    # Two unrelated failures trip the circuit and start a new epoch.
    breaker.record_failure(breaker.admit())
    breaker.record_failure(breaker.admit())
    assert breaker.admit() is None

    # The stale call also fails, but it belongs to the epoch before the trip
    # and must not double-count against - or otherwise disturb - the new one.
    breaker.record_failure(stale_epoch)

    # Let the cooldown lapse and confirm a single new failure alone hasn't
    # already re-tripped a freshly reset window (threshold is 2).
    breaker._open_until = 0.0
    epoch = breaker.admit()
    assert epoch is not None
    breaker.record_failure(epoch)
    assert breaker.admit() is not None
