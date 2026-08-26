from abc import ABC, abstractmethod
from typing import Any, Optional

_STREAM_CHUNK_SIZE = 1024 * 1024


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
    def exists(self, blob_key: str) -> bool:
        """Return whether `blob_key` is already present."""
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

    def exists(self, blob_key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=blob_key)
        except Exception as error:
            if _is_missing_object_error(error):
                return False
            raise
        return True

    def upload(self, blob_key: str, source_path: str) -> None:
        from boto3.s3.transfer import TransferConfig

        self._client.upload_file(
            source_path,
            self._bucket,
            blob_key,
            Config=TransferConfig(use_threads=False),
        )


def _is_missing_object_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    error_details = response.get("Error", {})
    return str(error_details.get("Code")) in {"404", "NoSuchKey", "NotFound"}
