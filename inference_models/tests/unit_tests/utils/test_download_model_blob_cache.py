from pathlib import Path
from typing import Optional
from unittest import mock

from inference_models.utils import download
from inference_models.utils.content_addressed_artifact_cache import (
    ContentAddressedArtifactCache,
    NullContentAddressedArtifactCache,
)


def _download(
    target_path: Path,
    md5_hash: Optional[str] = "c770e3485f6f6cd5bf2f78504bd56c50",
    content_addressed_artifact_cache: Optional[ContentAddressedArtifactCache] = None,
) -> None:
    download.safe_download_file(
        target_file_path=str(target_path),
        download_url="https://models.example.com/weights.onnx",
        download_id="download-id",
        md5_hash=md5_hash,
        verify_hash_while_download=True,
        progress=mock.MagicMock(),
        response_codes_to_retry=set(),
        request_timeout=5,
        max_threads_per_download=1,
        file_lock_acquire_timeout=1,
        content_addressed_artifact_cache=content_addressed_artifact_cache,
    )


def test_safe_download_uses_blob_cache_hit_without_source_download(
    tmp_path,
) -> None:
    target_path = tmp_path / "weights.onnx"
    blob_cache = mock.MagicMock()

    def restore(*, content_hash: str, target_path: str) -> bool:
        Path(target_path).write_bytes(b"cached")
        return True

    blob_cache.restore.side_effect = restore
    with mock.patch.object(download, "safe_execute_download") as source_download:
        _download(target_path, content_addressed_artifact_cache=blob_cache)

    assert target_path.read_bytes() == b"cached"
    source_download.assert_not_called()
    blob_cache.schedule_store.assert_not_called()


def test_safe_download_falls_back_on_cache_miss_and_schedules_upload(
    tmp_path,
) -> None:
    target_path = tmp_path / "weights.onnx"
    blob_cache = mock.MagicMock()
    blob_cache.restore.return_value = False

    def source_download(**kwargs) -> None:
        Path(kwargs["target_file_path"]).write_bytes(b"source")

    with mock.patch.object(
        download, "safe_execute_download", side_effect=source_download
    ) as source_download_mock:
        _download(target_path, content_addressed_artifact_cache=blob_cache)

    assert target_path.read_bytes() == b"source"
    source_download_mock.assert_called_once()
    blob_cache.schedule_store.assert_called_once_with(
        content_hash="c770e3485f6f6cd5bf2f78504bd56c50",
        source_path=str(target_path),
    )


def test_safe_download_routes_hashless_files_through_the_null_cache(
    tmp_path,
) -> None:
    target_path = tmp_path / "weights.onnx"
    null_cache = mock.create_autospec(NullContentAddressedArtifactCache, instance=True)
    null_cache.restore.return_value = False

    def source_download(**kwargs) -> None:
        Path(kwargs["target_file_path"]).write_bytes(b"source")

    with mock.patch.object(
        download, "NullContentAddressedArtifactCache", return_value=null_cache
    ) as null_cache_factory, mock.patch.object(
        download, "safe_execute_download", side_effect=source_download
    ):
        _download(
            target_path,
            md5_hash=None,
            content_addressed_artifact_cache=mock.MagicMock(),
        )

    assert target_path.read_bytes() == b"source"
    null_cache_factory.assert_called_once_with()
    # The null object absorbs the hashless case, so the call sites stay unguarded.
    null_cache.restore.assert_called_once_with(content_hash=None, target_path=mock.ANY)
    null_cache.schedule_store.assert_called_once_with(
        content_hash=None, source_path=str(target_path)
    )


def test_safe_download_falls_back_when_blob_cache_lookup_raises(tmp_path) -> None:
    target_path = tmp_path / "weights.onnx"
    blob_cache = mock.MagicMock()
    blob_cache.restore.side_effect = RuntimeError("read failed")

    def source_download(**kwargs) -> None:
        Path(kwargs["target_file_path"]).write_bytes(b"source")

    with mock.patch.object(
        download, "safe_execute_download", side_effect=source_download
    ):
        _download(target_path, content_addressed_artifact_cache=blob_cache)

    assert target_path.read_bytes() == b"source"


def test_safe_download_ignores_upload_scheduling_failure(tmp_path) -> None:
    target_path = tmp_path / "weights.onnx"
    blob_cache = mock.MagicMock()
    blob_cache.restore.return_value = False
    blob_cache.schedule_store.side_effect = RuntimeError("queue failed")

    def source_download(**kwargs) -> None:
        Path(kwargs["target_file_path"]).write_bytes(b"source")

    with mock.patch.object(
        download, "safe_execute_download", side_effect=source_download
    ):
        _download(target_path, content_addressed_artifact_cache=blob_cache)

    assert target_path.read_bytes() == b"source"


def test_safe_download_defaults_to_null_cache(tmp_path) -> None:
    target_path = tmp_path / "weights.onnx"
    null_cache = mock.create_autospec(NullContentAddressedArtifactCache, instance=True)
    null_cache.restore.return_value = False

    def source_download(**kwargs) -> None:
        Path(kwargs["target_file_path"]).write_bytes(b"source")

    with mock.patch.object(
        download, "NullContentAddressedArtifactCache", return_value=null_cache
    ), mock.patch.object(
        download, "safe_execute_download", side_effect=source_download
    ):
        _download(target_path)

    assert target_path.read_bytes() == b"source"
    null_cache.restore.assert_called_once_with(
        content_hash="c770e3485f6f6cd5bf2f78504bd56c50",
        target_path=mock.ANY,
    )
