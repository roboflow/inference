from unittest import mock

import pytest

from inference_models.utils import blob_storage
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


class _MissingObjectError(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


def _configure_missing_object(client: mock.MagicMock) -> None:
    client.head_object.side_effect = _MissingObjectError()


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


def test_s3_storage_conditionally_creates_small_object(tmp_path) -> None:
    client = mock.MagicMock()
    _configure_missing_object(client)
    storage = S3BlobStorage(client=client, bucket="models")
    source_path = tmp_path / "blob"
    source_path.write_bytes(b"weights")
    uploaded = {}

    def put_object(**kwargs) -> None:
        uploaded.update(kwargs)
        uploaded["Body"] = kwargs["Body"].read()

    client.put_object.side_effect = put_object

    storage.upload("prefix/hash", str(source_path))

    assert uploaded == {
        "Bucket": "models",
        "Key": "prefix/hash",
        "Body": b"weights",
        "IfNoneMatch": "*",
    }


def test_s3_storage_skips_upload_when_object_already_exists(tmp_path) -> None:
    client = mock.MagicMock()
    storage = S3BlobStorage(client=client, bucket="models")
    source_path = tmp_path / "blob"
    source_path.write_bytes(b"weights")

    storage.upload("prefix/hash", str(source_path))

    client.head_object.assert_called_once_with(
        Bucket="models",
        Key="prefix/hash",
    )
    client.put_object.assert_not_called()
    client.create_multipart_upload.assert_not_called()


def test_s3_storage_treats_existing_small_object_as_success(tmp_path) -> None:
    class PreconditionFailedError(Exception):
        response = {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        }

    client = mock.MagicMock()
    _configure_missing_object(client)
    client.put_object.side_effect = PreconditionFailedError()
    storage = S3BlobStorage(client=client, bucket="models")
    source_path = tmp_path / "blob"
    source_path.write_bytes(b"weights")

    storage.upload("prefix/hash", str(source_path))

    client.put_object.assert_called_once()


def test_s3_storage_conditionally_completes_multipart_upload(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(blob_storage, "_MULTIPART_UPLOAD_THRESHOLD", 5)
    monkeypatch.setattr(blob_storage, "_MULTIPART_UPLOAD_CHUNK_SIZE", 5)
    client = mock.MagicMock()
    _configure_missing_object(client)
    client.create_multipart_upload.return_value = {"UploadId": "upload-id"}
    client.upload_part.side_effect = [{"ETag": "etag-1"}, {"ETag": "etag-2"}]
    storage = S3BlobStorage(client=client, bucket="models")
    source_path = tmp_path / "blob"
    source_path.write_bytes(b"0123456789")

    storage.upload("prefix/hash", str(source_path))

    assert client.upload_part.call_args_list == [
        mock.call(
            Bucket="models",
            Key="prefix/hash",
            UploadId="upload-id",
            PartNumber=1,
            Body=b"01234",
        ),
        mock.call(
            Bucket="models",
            Key="prefix/hash",
            UploadId="upload-id",
            PartNumber=2,
            Body=b"56789",
        ),
    ]
    client.complete_multipart_upload.assert_called_once_with(
        Bucket="models",
        Key="prefix/hash",
        UploadId="upload-id",
        MultipartUpload={
            "Parts": [
                {"ETag": "etag-1", "PartNumber": 1},
                {"ETag": "etag-2", "PartNumber": 2},
            ]
        },
        IfNoneMatch="*",
    )
    client.abort_multipart_upload.assert_not_called()


def test_s3_storage_aborts_losing_conditional_multipart_upload(
    tmp_path, monkeypatch
) -> None:
    class PreconditionFailedError(Exception):
        response = {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        }

    monkeypatch.setattr(blob_storage, "_MULTIPART_UPLOAD_THRESHOLD", 5)
    monkeypatch.setattr(blob_storage, "_MULTIPART_UPLOAD_CHUNK_SIZE", 5)
    client = mock.MagicMock()
    _configure_missing_object(client)
    client.create_multipart_upload.return_value = {"UploadId": "upload-id"}
    client.upload_part.side_effect = [{"ETag": "etag-1"}, {"ETag": "etag-2"}]
    client.complete_multipart_upload.side_effect = PreconditionFailedError()
    storage = S3BlobStorage(client=client, bucket="models")
    source_path = tmp_path / "blob"
    source_path.write_bytes(b"0123456789")

    storage.upload("prefix/hash", str(source_path))

    client.abort_multipart_upload.assert_called_once_with(
        Bucket="models",
        Key="prefix/hash",
        UploadId="upload-id",
    )


def test_s3_storage_propagates_non_precondition_upload_failure(tmp_path) -> None:
    client = mock.MagicMock()
    _configure_missing_object(client)
    client.put_object.side_effect = RuntimeError("upload failed")
    storage = S3BlobStorage(client=client, bucket="models")
    source_path = tmp_path / "blob"
    source_path.write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="upload failed"):
        storage.upload("prefix/hash", str(source_path))
