import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

    cache = model_blob_cache._initialize_model_blob_cache()

    assert isinstance(cache, NullContentAddressedArtifactCache)


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
        cache = model_blob_cache._initialize_model_blob_cache()

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
        MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS=11.5,
        MODEL_BLOB_CACHE_FAILURE_THRESHOLD=4,
        MODEL_BLOB_CACHE_COOLDOWN_SECONDS=19.5,
    )

    with mock.patch.object(
        model_blob_cache, "_build_s3_client", return_value=mock.sentinel.client
    ) as build_client:
        cache = model_blob_cache._initialize_model_blob_cache()

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
        download_timeout_seconds=11.5,
        failure_threshold=4,
        cooldown_seconds=19.5,
    )
    assert cache._read_deadline_seconds == 11.5


def test_missing_bucket_returns_null_cache(monkeypatch) -> None:
    _configure(monkeypatch, MODEL_BLOB_CACHE_BUCKET=None)

    assert isinstance(
        model_blob_cache._initialize_model_blob_cache(),
        NullContentAddressedArtifactCache,
    )


def test_half_configured_credentials_return_null_cache(monkeypatch) -> None:
    _configure(
        monkeypatch,
        MODEL_BLOB_CACHE_ACCESS_KEY_ID="access",
        MODEL_BLOB_CACHE_SECRET_ACCESS_KEY=None,
    )

    assert isinstance(
        model_blob_cache._initialize_model_blob_cache(),
        NullContentAddressedArtifactCache,
    )


def test_missing_boto3_returns_null_cache(monkeypatch) -> None:
    _configure(monkeypatch)

    with mock.patch.object(
        model_blob_cache, "_build_s3_client", side_effect=ImportError("no boto3")
    ):
        cache = model_blob_cache._initialize_model_blob_cache()

    assert isinstance(cache, NullContentAddressedArtifactCache)


def test_client_construction_failure_returns_null_cache(monkeypatch) -> None:
    _configure(monkeypatch)

    with mock.patch.object(
        model_blob_cache, "_build_s3_client", side_effect=RuntimeError("bad client")
    ):
        cache = model_blob_cache._initialize_model_blob_cache()

    assert isinstance(cache, NullContentAddressedArtifactCache)


def test_get_model_blob_cache_initializes_once_under_concurrent_load() -> None:
    caller_count = 8
    callers_ready = threading.Barrier(caller_count)
    initialization_started = threading.Event()
    release_initialization = threading.Event()
    initialized_cache = mock.sentinel.initialized_cache

    def initialize():
        initialization_started.set()
        release_initialization.wait()
        return initialized_cache

    def get_cache():
        callers_ready.wait()
        return model_blob_cache.get_content_addressed_artifact_cache()

    with mock.patch.object(
        model_blob_cache,
        "_model_blob_cache_instance",
        model_blob_cache._MODEL_BLOB_CACHE_UNINITIALIZED,
    ), mock.patch.object(
        model_blob_cache, "_initialize_model_blob_cache", side_effect=initialize
    ) as initialize_mock:
        with ThreadPoolExecutor(max_workers=caller_count) as executor:
            futures = [executor.submit(get_cache) for _ in range(caller_count)]
            assert initialization_started.wait(timeout=1)
            time.sleep(0.02)
            assert initialize_mock.call_count == 1
            release_initialization.set()
            results = [future.result(timeout=1) for future in futures]

    assert all(result is initialized_cache for result in results)


def test_model_blob_cache_config_preserves_timeout_defaults() -> None:
    config = ModelBlobCacheConfig(bucket="models")

    assert config.connect_timeout_seconds == 1.0
    assert config.read_timeout_seconds == 2.0
    assert config.download_timeout_seconds == 30.0


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
        {"download_timeout_seconds": 0},
        {"cooldown_seconds": 0},
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
