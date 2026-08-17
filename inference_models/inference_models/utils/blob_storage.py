from abc import ABC, abstractmethod
from typing import Any

_STREAM_CHUNK_SIZE = 1024 * 1024


class BlobStorage(ABC):
    @abstractmethod
    def download(self, blob_key: str, target_path: str) -> bool:
        """Download a blob, returning False only when it does not exist."""
        pass

    @abstractmethod
    def upload(self, blob_key: str, source_path: str) -> None:
        pass


class S3BlobStorage(BlobStorage):
    """Thin file-transfer adapter for an S3-compatible client."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def download(self, blob_key: str, target_path: str) -> bool:
        body = None
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=blob_key,
            )
            body = response["Body"]
            with open(target_path, "wb") as target_file:
                while True:
                    chunk = body.read(_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
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
