import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

from inference_models.configuration import (
    MODEL_BLOB_CACHE_ACCESS_KEY_ID,
    MODEL_BLOB_CACHE_ADDRESSING_STYLE,
    MODEL_BLOB_CACHE_BUCKET,
    MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS,
    MODEL_BLOB_CACHE_COOLDOWN_SECONDS,
    MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS,
    MODEL_BLOB_CACHE_ENABLED,
    MODEL_BLOB_CACHE_ENDPOINT_URL,
    MODEL_BLOB_CACHE_FAILURE_THRESHOLD,
    MODEL_BLOB_CACHE_PREFIX,
    MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS,
    MODEL_BLOB_CACHE_REGION,
    MODEL_BLOB_CACHE_SECRET_ACCESS_KEY,
)
from inference_models.logger import LOGGER
from inference_models.utils.blob_storage import S3BlobStorage
from inference_models.utils.content_addressed_artifact_cache import (
    ContentAddressedArtifactCache,
    NullContentAddressedArtifactCache,
    VerifiedContentAddressedArtifactCache,
)

_SUPPORTED_ADDRESSING_STYLES = {"auto", "path", "virtual"}


@dataclass(frozen=True)
class ModelBlobCacheConfig:
    bucket: str
    prefix: str = "model-blobs"
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    addressing_style: str = "auto"
    connect_timeout_seconds: float = 1.0
    read_timeout_seconds: float = 2.0
    download_timeout_seconds: float = 30.0
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError(
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_BUCKET must be set when the "
                "cache is enabled"
            )
        if self.addressing_style not in _SUPPORTED_ADDRESSING_STYLES:
            raise ValueError(
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_ADDRESSING_STYLE must be one "
                f"of: {', '.join(sorted(_SUPPORTED_ADDRESSING_STYLES))}"
            )
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise ValueError(
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_ACCESS_KEY_ID and "
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_SECRET_ACCESS_KEY must be set "
                "together"
            )
        if self.failure_threshold < 1:
            raise ValueError(
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_FAILURE_THRESHOLD must be at "
                "least 1"
            )
        for name, value in (
            (
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS",
                self.connect_timeout_seconds,
            ),
            (
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS",
                self.read_timeout_seconds,
            ),
            (
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS",
                self.download_timeout_seconds,
            ),
            (
                "INFERENCE_MODELS_MODEL_BLOB_CACHE_COOLDOWN_SECONDS",
                self.cooldown_seconds,
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


def _current_configuration() -> ModelBlobCacheConfig:
    """Compose the cache configuration from `inference_models.configuration`.

    Construction validates the result, so an invalid combination raises here and
    is turned into a fail-open cache by `_initialize_model_blob_cache`.
    """
    return ModelBlobCacheConfig(
        bucket=MODEL_BLOB_CACHE_BUCKET,
        prefix=MODEL_BLOB_CACHE_PREFIX,
        endpoint_url=MODEL_BLOB_CACHE_ENDPOINT_URL,
        region=MODEL_BLOB_CACHE_REGION,
        access_key_id=MODEL_BLOB_CACHE_ACCESS_KEY_ID,
        secret_access_key=MODEL_BLOB_CACHE_SECRET_ACCESS_KEY,
        addressing_style=MODEL_BLOB_CACHE_ADDRESSING_STYLE,
        connect_timeout_seconds=MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds=MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS,
        download_timeout_seconds=MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS,
        failure_threshold=MODEL_BLOB_CACHE_FAILURE_THRESHOLD,
        cooldown_seconds=MODEL_BLOB_CACHE_COOLDOWN_SECONDS,
    )


def _build_s3_client(config: ModelBlobCacheConfig) -> Any:
    import boto3
    from botocore.config import Config

    client_config = Config(
        connect_timeout=config.connect_timeout_seconds,
        read_timeout=config.read_timeout_seconds,
        retries={"total_max_attempts": 1, "mode": "standard"},
        s3={"addressing_style": config.addressing_style},
    )
    client_kwargs: Dict[str, Any] = {"config": client_config}
    if config.endpoint_url:
        client_kwargs["endpoint_url"] = config.endpoint_url
    if config.region:
        client_kwargs["region_name"] = config.region
    if config.access_key_id and config.secret_access_key:
        client_kwargs["aws_access_key_id"] = config.access_key_id
        client_kwargs["aws_secret_access_key"] = config.secret_access_key
    return boto3.client("s3", **client_kwargs)


def _initialize_model_blob_cache() -> ContentAddressedArtifactCache:
    try:
        if not MODEL_BLOB_CACHE_ENABLED:
            return NullContentAddressedArtifactCache()
        config = _current_configuration()
        storage = S3BlobStorage(client=_build_s3_client(config), bucket=config.bucket)
        return VerifiedContentAddressedArtifactCache(
            storage=storage,
            prefix=config.prefix,
            read_deadline_seconds=config.download_timeout_seconds,
            failure_threshold=config.failure_threshold,
            cooldown_seconds=config.cooldown_seconds,
        )
    except Exception as error:
        LOGGER.warning(
            "Could not initialize model blob cache; using original model sources: %s",
            error,
        )
        return NullContentAddressedArtifactCache()


_MODEL_BLOB_CACHE_UNINITIALIZED = object()
_model_blob_cache_instance: object = _MODEL_BLOB_CACHE_UNINITIALIZED
_model_blob_cache_initialization_lock = threading.Lock()


def get_content_addressed_artifact_cache() -> ContentAddressedArtifactCache:
    global _model_blob_cache_instance

    if _model_blob_cache_instance is _MODEL_BLOB_CACHE_UNINITIALIZED:
        with _model_blob_cache_initialization_lock:
            if _model_blob_cache_instance is _MODEL_BLOB_CACHE_UNINITIALIZED:
                _model_blob_cache_instance = _initialize_model_blob_cache()
    return cast(ContentAddressedArtifactCache, _model_blob_cache_instance)
