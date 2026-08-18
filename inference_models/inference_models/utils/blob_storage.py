from abc import ABC, abstractmethod
from time import monotonic
from typing import Any, Optional

_STREAM_CHUNK_SIZE = 1024 * 1024


class TransferDeadlineExceeded(Exception):
    """Raised when a transfer is abandoned because its deadline passed."""


class BlobStorage(ABC):
    @abstractmethod
    def download(
        self, blob_key: str, target_path: str, timeout_seconds: Optional[float] = None
    ) -> bool:
        """Download a blob, returning False only when it does not exist.

        `timeout_seconds` is a no-progress budget, not a whole-transfer cap:
        implementations should re-arm it for the same duration each time a
        chunk arrives, so a slow-but-steady multi-gigabyte transfer isn't
        penalised for its size while a stalled or hung endpoint still raises
        `TransferDeadlineExceeded` instead of running to completion.
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
        self, blob_key: str, target_path: str, timeout_seconds: Optional[float] = None
    ) -> bool:
        body = None
        # A single fixed deadline would cap total transfer time rather than
        # bound stalls - re-arm it to `timeout_seconds` from now every time a
        # chunk arrives, so only a genuine no-progress gap raises.
        deadline = (
            monotonic() + timeout_seconds if timeout_seconds is not None else None
        )
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=blob_key,
            )
            body = response["Body"]
            with open(target_path, "wb") as target_file:
                while True:
                    if deadline is not None and monotonic() > deadline:
                        raise TransferDeadlineExceeded(
                            f"abandoned download of {blob_key}: no progress "
                            f"within {timeout_seconds:.2f}s"
                        )
                    chunk = body.read(_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    target_file.write(chunk)
                    if timeout_seconds is not None:
                        deadline = monotonic() + timeout_seconds
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
