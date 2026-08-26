from unittest import mock

import pytest

from inference_models.utils import model_blob_cache
from inference_models.utils.content_addressed_artifact_cache import (
    NullContentAddressedArtifactCache,
    VerifiedContentAddressedArtifactCache,
)
from inference_models.utils.model_blob_cache import (
    ModelBlobCacheConfig,
    _build_s3_client,
)


def _configure(monkeypatch, **overrides) -> None:
    """Point the cache module at a configuration, as `configuration` would."""
    settings = {"MODEL_BLOB_CACHE_ENABLED": True, "MODEL_BLOB_CACHE_BUCKET": "models"}
    settings.update(overrides)
    for name, value in settings.items():
        monkeypatch.setattr(model_blob_cache, name, value)


def test_disabled_factory_returns_null_cache(monkeypatch) -> None:
    _configure(monkeypatch, MODEL_BLOB_CACHE_ENABLED=False)

    cache = model_blob_cache.create_model_blob_cache()

    assert isinstance(cache, NullContentAddressedArtifactCache)


def test_factory_does_not_cache_instances(monkeypatch) -> None:
    _configure(monkeypatch, MODEL_BLOB_CACHE_ENABLED=False)

    first = model_blob_cache.create_model_blob_cache()
    second = model_blob_cache.create_model_blob_cache()

    assert first is not second


def test_factory_reads_current_configuration(monkeypatch) -> None:
    _configure(
        monkeypatch,
        MODEL_BLOB_CACHE_BUCKET="current-bucket",
        MODEL_BLOB_CACHE_PREFIX="/current-prefix/",
    )
    client = mock.sentinel.client

    with mock.patch.object(
        model_blob_cache, "_build_s3_client", return_value=client
    ) as build_client:
        cache = model_blob_cache.create_model_blob_cache()

    assert isinstance(cache, VerifiedContentAddressedArtifactCache)
    assert cache._prefix == "/current-prefix/"
    assert cache._storage._bucket == "current-bucket"
    build_client.assert_called_once()


def test_factory_passes_complete_configuration(monkeypatch) -> None:
    _configure(
        monkeypatch,
        MODEL_BLOB_CACHE_PREFIX="/artifacts/",
        MODEL_BLOB_CACHE_ENDPOINT_URL="https://objects.example.com",
        MODEL_BLOB_CACHE_REGION="region-1",
        MODEL_BLOB_CACHE_ACCESS_KEY_ID="access",
        MODEL_BLOB_CACHE_SECRET_ACCESS_KEY="secret",
        MODEL_BLOB_CACHE_ADDRESSING_STYLE="path",
        MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS=2.5,
        MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS=7.5,
        MODEL_BLOB_CACHE_FAILURE_THRESHOLD=4,
        MODEL_BLOB_CACHE_COOLDOWN_SECONDS=19.5,
        MODEL_BLOB_CACHE_MAX_OBJECT_BYTES=123_456,
    )

    with mock.patch.object(
        model_blob_cache, "_build_s3_client", return_value=mock.sentinel.client
    ) as build_client:
        cache = model_blob_cache.create_model_blob_cache()

    config = build_client.call_args.args[0]
    assert config == ModelBlobCacheConfig(
        bucket="models",
        prefix="/artifacts/",
        endpoint_url="https://objects.example.com",
        region="region-1",
        access_key_id="access",
        secret_access_key="secret",
        addressing_style="path",
        connect_timeout_seconds=2.5,
        read_timeout_seconds=7.5,
        failure_threshold=4,
        cooldown_seconds=19.5,
        max_object_bytes=123_456,
    )
    assert cache._max_object_bytes == 123_456


def test_missing_bucket_returns_null_cache(monkeypatch) -> None:
    _configure(monkeypatch, MODEL_BLOB_CACHE_BUCKET=None)

    assert isinstance(
        model_blob_cache.create_model_blob_cache(),
        NullContentAddressedArtifactCache,
    )


def test_half_configured_credentials_return_null_cache(monkeypatch) -> None:
    _configure(
        monkeypatch,
        MODEL_BLOB_CACHE_ACCESS_KEY_ID="access",
        MODEL_BLOB_CACHE_SECRET_ACCESS_KEY=None,
    )

    assert isinstance(
        model_blob_cache.create_model_blob_cache(),
        NullContentAddressedArtifactCache,
    )


def test_missing_boto3_returns_null_cache(monkeypatch) -> None:
    _configure(monkeypatch)

    with mock.patch.object(
        model_blob_cache, "_build_s3_client", side_effect=ImportError("no boto3")
    ):
        cache = model_blob_cache.create_model_blob_cache()

    assert isinstance(cache, NullContentAddressedArtifactCache)


def test_client_construction_failure_returns_null_cache(monkeypatch) -> None:
    _configure(monkeypatch)

    with mock.patch.object(
        model_blob_cache, "_build_s3_client", side_effect=RuntimeError("bad client")
    ):
        cache = model_blob_cache.create_model_blob_cache()

    assert isinstance(cache, NullContentAddressedArtifactCache)


def test_model_blob_cache_config_preserves_timeout_defaults() -> None:
    config = ModelBlobCacheConfig(bucket="models")

    assert config.connect_timeout_seconds == 1.0
    assert config.read_timeout_seconds == 2.0
    assert config.max_object_bytes == 20 * 1024**3


@pytest.mark.parametrize(
    "overrides",
    [
        {"bucket": ""},
        {"addressing_style": "dns"},
        {"access_key_id": "access"},
        {"secret_access_key": "secret"},
        {"failure_threshold": 0},
        {"connect_timeout_seconds": 0},
        {"read_timeout_seconds": -1},
        {"cooldown_seconds": 0},
        {"max_object_bytes": 0},
    ],
)
def test_model_blob_cache_config_rejects_invalid_values(overrides) -> None:
    settings = {"bucket": "models", **overrides}

    with pytest.raises(ValueError):
        ModelBlobCacheConfig(**settings)


def test_model_blob_cache_config_accepts_paired_credentials() -> None:
    config = ModelBlobCacheConfig(
        bucket="models", access_key_id="access", secret_access_key="secret"
    )

    assert config.access_key_id == "access"
    assert config.secret_access_key == "secret"


def test_s3_client_disables_retries_and_applies_provider_options() -> None:
    config = ModelBlobCacheConfig(
        bucket="models",
        endpoint_url="https://objects.example.com",
        region="region-1",
        access_key_id="access",
        secret_access_key="secret",
        addressing_style="path",
        connect_timeout_seconds=2,
        read_timeout_seconds=10,
    )

    with mock.patch("boto3.client") as client_factory:
        _build_s3_client(config)

    kwargs = client_factory.call_args.kwargs
    assert kwargs["endpoint_url"] == "https://objects.example.com"
    assert kwargs["region_name"] == "region-1"
    assert kwargs["aws_access_key_id"] == "access"
    assert kwargs["aws_secret_access_key"] == "secret"
    assert kwargs["config"].connect_timeout == 2
    assert kwargs["config"].read_timeout == 10
    assert kwargs["config"].retries["total_max_attempts"] == 1
    assert kwargs["config"].s3["addressing_style"] == "path"
