import importlib
import os
from typing import Callable

import pytest

import inference_models.configuration
from inference_models.configuration import (
    DEFAULT_RFDETR_PIPELINE_DEPTH,
    MAX_RFDETR_PIPELINE_DEPTH,
    get_rfdetr_pipeline_depth,
    parse_rfdetr_pipeline_depth,
)
from inference_models.errors import InvalidEnvVariable

CONFIGURATION_ENVIRONMENT_KEYS = [
    "ROBOFLOW_REGION",
    "ROBOFLOW_ENVIRONMENT",
    "ROBOFLOW_API_HOST",
    "MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS",
    "MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS",
    "MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS",
    "MODEL_BLOB_CACHE_COOLDOWN_SECONDS",
    "MODEL_BLOB_CACHE_FAILURE_THRESHOLD",
    "MODEL_BLOB_CACHE_ADDRESSING_STYLE",
]


@pytest.fixture
def reload_configuration() -> Callable[..., object]:
    saved_environment = {
        key: os.environ.pop(key)
        for key in CONFIGURATION_ENVIRONMENT_KEYS
        if key in os.environ
    }

    def _reload(**environment: str) -> object:
        for key in CONFIGURATION_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(environment)
        return importlib.reload(inference_models.configuration)

    try:
        yield _reload
    finally:
        for key in CONFIGURATION_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(saved_environment)
        importlib.reload(inference_models.configuration)


def test_parse_rfdetr_pipeline_depth_uses_default_when_env_missing() -> None:
    assert parse_rfdetr_pipeline_depth(None) == DEFAULT_RFDETR_PIPELINE_DEPTH


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", 1),
        ("2", 2),
        (" 3 ", MAX_RFDETR_PIPELINE_DEPTH),
        ("99", MAX_RFDETR_PIPELINE_DEPTH),
    ],
)
def test_parse_rfdetr_pipeline_depth_accepts_positive_integers(
    value: str,
    expected: int,
) -> None:
    assert parse_rfdetr_pipeline_depth(value) == expected


@pytest.mark.parametrize("value", ["invalid", "1.5", "", "0", "-1"])
def test_parse_rfdetr_pipeline_depth_rejects_invalid_values(value: str) -> None:
    with pytest.raises(InvalidEnvVariable):
        parse_rfdetr_pipeline_depth(value)


def test_get_rfdetr_pipeline_depth_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "3")
    assert get_rfdetr_pipeline_depth() == MAX_RFDETR_PIPELINE_DEPTH


@pytest.mark.parametrize("value", ["0", "-4", "invalid"])
def test_get_rfdetr_pipeline_depth_rejects_invalid_environment(
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", value)
    with pytest.raises(InvalidEnvVariable):
        get_rfdetr_pipeline_depth()


def test_roboflow_api_host_defaults_to_us_production(reload_configuration) -> None:
    configuration = reload_configuration()
    assert configuration.ROBOFLOW_REGION == "us"
    assert configuration.ROBOFLOW_API_HOST == "https://api.roboflow.com"


def test_model_blob_cache_uses_fail_fast_default_timeouts(
    reload_configuration,
) -> None:
    configuration = reload_configuration()

    assert configuration.MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS == 1.0
    assert configuration.MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS == 2.0
    assert configuration.MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS == 30.0


def test_model_blob_cache_timeout_defaults_can_be_overridden(
    reload_configuration,
) -> None:
    configuration = reload_configuration(
        MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS="4.5",
        MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS="8.5",
        MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS="45",
    )

    assert configuration.MODEL_BLOB_CACHE_CONNECT_TIMEOUT_SECONDS == 4.5
    assert configuration.MODEL_BLOB_CACHE_READ_TIMEOUT_SECONDS == 8.5
    assert configuration.MODEL_BLOB_CACHE_DOWNLOAD_TIMEOUT_SECONDS == 45.0


def test_model_blob_cache_leaves_range_checks_to_its_config(
    reload_configuration,
) -> None:
    # Parsing lives here; `ModelBlobCacheConfig` rejects out-of-range values so
    # a misconfigured optional cache cannot break the library import.
    configuration = reload_configuration(
        MODEL_BLOB_CACHE_COOLDOWN_SECONDS="0",
        MODEL_BLOB_CACHE_FAILURE_THRESHOLD="0",
        MODEL_BLOB_CACHE_ADDRESSING_STYLE="dns",
    )

    assert configuration.MODEL_BLOB_CACHE_COOLDOWN_SECONDS == 0.0
    assert configuration.MODEL_BLOB_CACHE_FAILURE_THRESHOLD == 0
    assert configuration.MODEL_BLOB_CACHE_ADDRESSING_STYLE == "dns"


def test_model_blob_cache_still_rejects_unparsable_values(reload_configuration) -> None:
    with pytest.raises(InvalidEnvVariable):
        reload_configuration(MODEL_BLOB_CACHE_COOLDOWN_SECONDS="not-a-number")


@pytest.mark.parametrize(
    "region, environment, expected_api_host",
    [
        ("us", "prod", "https://api.roboflow.com"),
        ("us", "staging", "https://api.roboflow.one"),
        ("eu", "prod", "https://api.roboflow.eu"),
        ("eu", "staging", "https://api.roboflow-eu.one"),
    ],
)
def test_roboflow_api_host_follows_region_and_environment_matrix(
    reload_configuration,
    region: str,
    environment: str,
    expected_api_host: str,
) -> None:
    configuration = reload_configuration(
        ROBOFLOW_REGION=region, ROBOFLOW_ENVIRONMENT=environment
    )
    assert configuration.ROBOFLOW_API_HOST == expected_api_host


def test_explicit_roboflow_api_host_beats_region_selection(
    reload_configuration,
) -> None:
    configuration = reload_configuration(
        ROBOFLOW_REGION="eu", ROBOFLOW_API_HOST="https://api.example.com"
    )
    assert configuration.ROBOFLOW_API_HOST == "https://api.example.com"


def test_unknown_roboflow_region_warns_and_falls_back_to_us(
    reload_configuration,
) -> None:
    with pytest.warns(UserWarning, match="Unknown ROBOFLOW_REGION"):
        configuration = reload_configuration(ROBOFLOW_REGION="mars")
    assert configuration.ROBOFLOW_REGION == "us"
    assert configuration.ROBOFLOW_API_HOST == "https://api.roboflow.com"
