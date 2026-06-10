"""Regression tests for the file-name source in `pull_batch_element_to_directory`.

The function used to derive the on-disk file name from `urlparse(download_url).path`,
which broke for signed URLs whose path basename differs from the logical file name
(e.g. GCS V4-signed URLs that rewrite path components, or URLs ending in a UUID
without an extension). The fix uses `file_metadata.file_name` directly. These
tests pin the corrected behavior for both the loose-file and archive branches.
"""

from unittest.mock import MagicMock

from inference_cli.lib.roboflow_cloud.data_staging import api_operations
from inference_cli.lib.roboflow_cloud.data_staging.entities import FileMetadata


def _file_metadata(file_name: str, download_url: str) -> FileMetadata:
    return FileMetadata(
        downloadURL=download_url,
        fileName=file_name,
        contentType="application/octet-stream",
    )


def _export_log_allowing_export() -> MagicMock:
    log = MagicMock()
    log.is_already_exported.return_value = False
    return log


def test_pull_uses_metadata_file_name_not_url_basename_for_loose_files(
    monkeypatch, tmp_path
) -> None:
    # Signed URL path ends in a UUID with no relation to the logical name;
    # the fix uses file_metadata.file_name verbatim.
    file_metadata = _file_metadata(
        file_name="my-image.jpg",
        download_url="https://storage.googleapis.com/b/01HX-some-uuid?X-Goog-Signature=abc",
    )
    pull_file = MagicMock(return_value=str(tmp_path / "my-image.jpg"))
    monkeypatch.setattr(api_operations, "pull_file_to_directory", pull_file)

    api_operations.pull_batch_element_to_directory(
        file_metadata=file_metadata,
        target_directory=str(tmp_path),
        export_log=_export_log_allowing_export(),
        override_existing=False,
    )

    assert pull_file.call_count == 1
    assert pull_file.call_args.kwargs["file_name"] == "my-image.jpg"


def test_pull_uses_metadata_file_name_for_tar_gz_archive_branch(
    monkeypatch, tmp_path
) -> None:
    # Archive branch must use the metadata file_name too — otherwise
    # download_and_unpack_archive can't find / write to the right file.
    file_metadata = _file_metadata(
        file_name="images-shard-7.tar.gz",
        download_url="https://storage.googleapis.com/b/blob-uuid?X-Goog-Signature=xyz",
    )
    unpack = MagicMock()
    monkeypatch.setattr(api_operations, "download_and_unpack_archive", unpack)

    api_operations.pull_batch_element_to_directory(
        file_metadata=file_metadata,
        target_directory=str(tmp_path),
        export_log=_export_log_allowing_export(),
        override_existing=False,
    )

    assert unpack.call_count == 1
    assert unpack.call_args.kwargs["file_name"] == "images-shard-7.tar.gz"


def test_pull_uses_metadata_file_name_for_tar_archive_branch(
    monkeypatch, tmp_path
) -> None:
    file_metadata = _file_metadata(
        file_name="images-shard-7.tar",
        download_url="https://storage.googleapis.com/b/blob-uuid?X-Goog-Signature=xyz",
    )
    unpack = MagicMock()
    monkeypatch.setattr(api_operations, "download_and_unpack_archive", unpack)

    api_operations.pull_batch_element_to_directory(
        file_metadata=file_metadata,
        target_directory=str(tmp_path),
        export_log=_export_log_allowing_export(),
        override_existing=False,
    )

    assert unpack.call_args.kwargs["file_name"] == "images-shard-7.tar"
