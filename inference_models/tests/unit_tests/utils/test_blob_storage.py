from unittest import mock

import pytest

from inference_models.utils.blob_storage import BlobStorage, BlobTooLarge, S3BlobStorage


class _StreamingBody:
    def __init__(self, chunks) -> None:
        self._chunks = iter(chunks)
        self.closed = False
        self.reads = 0

    def read(self, _: int) -> bytes:
        self.reads += 1
        chunk = next(self._chunks)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def close(self) -> None:
        self.closed = True


def test_blob_storage_cannot_be_constructed_without_download_and_upload() -> None:
    class IncompleteBlobStorage(BlobStorage):
        pass

    try:
        IncompleteBlobStorage()
    except TypeError:
        return
    raise AssertionError("BlobStorage must enforce its transfer contract")


def test_s3_storage_translates_missing_object_to_false(tmp_path) -> None:
    class MissingObjectError(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    client = mock.MagicMock()
    client.get_object.side_effect = MissingObjectError()
    storage = S3BlobStorage(client=client, bucket="models")

    assert storage.download("prefix/hash", str(tmp_path / "blob")) is False


def test_s3_storage_streams_get_object_body_and_closes_it(tmp_path) -> None:
    body = _StreamingBody([b"cached ", b"weights", b""])
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": body}
    storage = S3BlobStorage(client=client, bucket="models")
    target_path = tmp_path / "blob"

    assert storage.download("prefix/hash", str(target_path)) is True

    client.get_object.assert_called_once_with(Bucket="models", Key="prefix/hash")
    assert target_path.read_bytes() == b"cached weights"
    assert body.closed is True


def test_s3_storage_closes_response_body_when_streaming_fails(tmp_path) -> None:
    body = _StreamingBody([b"partial", RuntimeError("read failed")])
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": body}
    storage = S3BlobStorage(client=client, bucket="models")

    try:
        storage.download("prefix/hash", str(tmp_path / "blob"))
    except RuntimeError as error:
        assert str(error) == "read failed"
    else:
        raise AssertionError("streaming failure must propagate")

    assert body.closed is True


def test_s3_storage_rejects_an_oversized_declared_content_length(tmp_path) -> None:
    body = _StreamingBody([b"chunk"] * 1000)
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": body, "ContentLength": 10_000_000}
    storage = S3BlobStorage(client=client, bucket="models")

    with pytest.raises(BlobTooLarge):
        storage.download("prefix/hash", str(tmp_path / "blob"), max_bytes=1_000_000)

    # Rejected on the declared size alone - nothing was ever read or written.
    assert body.reads == 0
    assert body.closed is True


def test_s3_storage_rejects_actual_bytes_exceeding_the_cap_even_when_undeclared(
    tmp_path,
) -> None:
    # No (or an understated) Content-Length must not be a way around the cap:
    # the actual streamed byte count is tracked independently.
    body = _StreamingBody([b"x" * 700_000, b"x" * 700_000, b""])
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": body}  # no ContentLength at all
    storage = S3BlobStorage(client=client, bucket="models")
    target_path = tmp_path / "blob"

    with pytest.raises(BlobTooLarge):
        storage.download("prefix/hash", str(target_path), max_bytes=1_000_000)

    assert body.closed is True
    # The second (over-budget) chunk must not have landed on disk.
    assert target_path.read_bytes() == b"x" * 700_000


def test_s3_storage_completes_a_transfer_within_its_byte_cap(tmp_path) -> None:
    body = _StreamingBody([b"cached ", b"weights", b""])
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": body, "ContentLength": 14}
    storage = S3BlobStorage(client=client, bucket="models")
    target_path = tmp_path / "blob"

    assert (
        storage.download("prefix/hash", str(target_path), max_bytes=1_000_000) is True
    )
    assert target_path.read_bytes() == b"cached weights"


def test_s3_storage_upload_disables_transfer_threads(tmp_path) -> None:
    client = mock.MagicMock()
    storage = S3BlobStorage(client=client, bucket="models")
    source_path = tmp_path / "blob"
    source_path.write_bytes(b"weights")

    storage.upload("prefix/hash", str(source_path))

    transfer_config = client.upload_file.call_args.kwargs["Config"]
    assert transfer_config.use_threads is False
