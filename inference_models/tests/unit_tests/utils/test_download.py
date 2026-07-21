from pathlib import Path
from unittest import mock

import pytest

from inference_models.utils import download


def test_download_files_to_directory_returns_cached_file_in_offline_mode(
    tmp_path: Path,
) -> None:
    """Cached downloads remain idempotent when network access is disabled."""
    # given
    target_file = tmp_path / "model.bin"
    target_file.write_bytes(b"cached model")
    files_specs = [("model.bin", "https://example.test/model.bin", None)]

    # when
    with mock.patch.object(download, "OFFLINE_MODE", True), mock.patch.object(
        download, "safe_download_file"
    ) as safe_download_file_mock:
        result = download.download_files_to_directory(
            target_dir=str(tmp_path), files_specs=files_specs
        )

    # then
    assert result == {"model.bin": str(target_file)}
    safe_download_file_mock.assert_not_called()


def test_download_files_to_directory_rejects_missing_file_in_offline_mode(
    tmp_path: Path,
) -> None:
    """An uncached file still fails before any download attempt in offline mode."""
    # given
    files_specs = [("model.bin", "https://example.test/model.bin", None)]

    # when
    with mock.patch.object(download, "OFFLINE_MODE", True), mock.patch.object(
        download, "safe_download_file"
    ) as safe_download_file_mock, pytest.raises(RuntimeError, match="OFFLINE_MODE"):
        download.download_files_to_directory(
            target_dir=str(tmp_path), files_specs=files_specs
        )

    # then
    safe_download_file_mock.assert_not_called()
