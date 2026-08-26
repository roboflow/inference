import math
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

_STREAM_CHUNK_SIZE = 1024 * 1024
_MULTIPART_UPLOAD_THRESHOLD = 8 * 1024 * 1024
_MULTIPART_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024
_MAX_MULTIPART_PARTS = 10_000


class BlobTooLarge(Exception):
    """Raised when a blob's declared or actual size exceeds `max_bytes`."""


class BlobStorage(ABC):
    @abstractmethod
    def download(
        self, blob_key: str, target_path: str, max_bytes: Optional[int] = None
    ) -> bool:
        """Download a blob, returning False only when it does not exist.

        Bounding a stalled or hung connection is the client's own job (its
        connect/read timeouts) - this method does not add a second timeout
        layer on top of that, since a fixed or re-armed app-level deadline
        either caps large-but-healthy transfers by their size or can be
        strung along indefinitely by a connection that trickles just enough
        data to keep renewing its own budget.

        `max_bytes`, if given, bounds how much this method will write to
        `target_path` before raising `BlobTooLarge`. This is a sanity cap on
        a network-backed cache accepting arbitrary bytes under a caller-
        supplied key, not a content-integrity check - the MD5 verification
        the cache performs on the completed file is what actually decides
        whether the content is trustworthy.
        """
        pass

    @abstractmethod
    def upload(self, blob_key: str, source_path: str) -> None:
        pass


class S3BlobStorage(BlobStorage):
    """Thin file-transfer adapter for an S3-compatible client."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def download(
        self, blob_key: str, target_path: str, max_bytes: Optional[int] = None
    ) -> bool:
        body = None
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=blob_key,
            )
            # `body` must be assigned before any early return/raise below, or
            # the `finally` block has nothing to close and the connection
            # leaks. Reject an oversized declared size before reading
            # anything, but don't stop there: a server can declare a small
            # Content-Length and then send more anyway (a misbehaving proxy,
            # chunked-encoding weirdness, a bug on the write side), so the
            # actual byte count streamed is tracked independently below too.
            body = response["Body"]
            content_length = response.get("ContentLength")
            if (
                max_bytes is not None
                and content_length is not None
                and content_length > max_bytes
            ):
                raise BlobTooLarge(
                    f"{blob_key} declares {content_length} bytes, over the "
                    f"{max_bytes} byte cache limit"
                )
            bytes_written = 0
            with open(target_path, "wb") as target_file:
                while True:
                    chunk = body.read(_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if max_bytes is not None and bytes_written > max_bytes:
                        raise BlobTooLarge(
                            f"{blob_key} exceeded the {max_bytes} byte cache "
                            "limit while streaming"
                        )
                    target_file.write(chunk)
            return True
        except Exception as error:
            if _is_missing_object_error(error):
                return False
            raise
        finally:
            if body is not None:
                try:
                    body.close()
                except Exception:
                    pass

    def upload(self, blob_key: str, source_path: str) -> None:
        if self._object_exists(blob_key):
            return
        source_size = os.path.getsize(source_path)
        try:
            if source_size < _MULTIPART_UPLOAD_THRESHOLD:
                self._put_object_if_absent(blob_key, source_path)
            else:
                self._multipart_upload_if_absent(
                    blob_key=blob_key,
                    source_path=source_path,
                    source_size=source_size,
                )
        except Exception as error:
            if _is_precondition_failed_error(error):
                # Concurrent processes running inference can observe the same
                # cache miss and write the same content-addressed object at the
                # same time. This is acceptable: the first conditional write
                # wins and the others are no-ops.
                return
            raise

    def _object_exists(self, blob_key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=blob_key)
        except Exception as error:
            if _is_missing_object_error(error):
                return False
            raise
        return True

    def _put_object_if_absent(self, blob_key: str, source_path: str) -> None:
        with open(source_path, "rb") as source_file:
            self._client.put_object(
                Bucket=self._bucket,
                Key=blob_key,
                Body=source_file,
                IfNoneMatch="*",
            )

    def _multipart_upload_if_absent(
        self, blob_key: str, source_path: str, source_size: int
    ) -> None:
        multipart_upload = self._client.create_multipart_upload(
            Bucket=self._bucket,
            Key=blob_key,
        )
        upload_id = multipart_upload["UploadId"]
        chunk_size = max(
            _MULTIPART_UPLOAD_CHUNK_SIZE,
            math.ceil(source_size / _MAX_MULTIPART_PARTS),
        )
        parts = []
        try:
            with open(source_path, "rb") as source_file:
                part_number = 1
                while chunk := source_file.read(chunk_size):
                    response = self._client.upload_part(
                        Bucket=self._bucket,
                        Key=blob_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=chunk,
                    )
                    parts.append({"ETag": response["ETag"], "PartNumber": part_number})
                    part_number += 1
            self._client.complete_multipart_upload(
                Bucket=self._bucket,
                Key=blob_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
        except Exception:
            try:
                self._client.abort_multipart_upload(
                    Bucket=self._bucket,
                    Key=blob_key,
                    UploadId=upload_id,
                )
            except Exception:
                pass
            raise


def _is_missing_object_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    error_details = response.get("Error", {})
    return str(error_details.get("Code")) in {"404", "NoSuchKey", "NotFound"}


def _is_precondition_failed_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    error_details = response.get("Error", {})
    response_metadata = response.get("ResponseMetadata", {})
    return str(error_details.get("Code")) in {"412", "PreconditionFailed"} or (
        response_metadata.get("HTTPStatusCode") == 412
    )
