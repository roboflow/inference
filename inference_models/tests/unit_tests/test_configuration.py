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

REGION_ENVIRONMENT_KEYS = [
    "ROBOFLOW_REGION",
    "ROBOFLOW_ENVIRONMENT",
    "ROBOFLOW_API_HOST",
]


@pytest.fixture
def reload_configuration() -> Callable[..., object]:
    saved_environment = {
        key: os.environ.pop(key) for key in REGION_ENVIRONMENT_KEYS if key in os.environ
    }

    def _reload(**environment: str) -> object:
        for key in REGION_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(environment)
        return importlib.reload(inference_models.configuration)

    try:
        yield _reload
    finally:
        for key in REGION_ENVIRONMENT_KEYS:
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


VLLM_ENVIRONMENT_KEYS = ["VLLM_PROXY_ENABLED", "VLLM_REQUEST_TIMEOUT_S"]


@pytest.fixture
def reload_configuration_with_vllm_environment() -> Callable[..., object]:
    saved_environment = {
        key: os.environ.pop(key) for key in VLLM_ENVIRONMENT_KEYS if key in os.environ
    }

    def _reload(**environment: str) -> object:
        for key in VLLM_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(environment)
        return importlib.reload(inference_models.configuration)

    try:
        yield _reload
    finally:
        for key in VLLM_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(saved_environment)
        importlib.reload(inference_models.configuration)


def test_invalid_vllm_setting_does_not_break_importing_the_package(
    reload_configuration_with_vllm_environment: Callable[..., object],
) -> None:
    configuration = reload_configuration_with_vllm_environment(
        VLLM_REQUEST_TIMEOUT_S="invalid"
    )

    assert configuration.VLLM_PROXY_ENABLED is False
    assert configuration.VLLM_REQUEST_TIMEOUT_S == "invalid"
    with pytest.raises(InvalidEnvVariable, match="VLLM_REQUEST_TIMEOUT_S"):
        configuration.validate_vllm_proxy_settings()


def test_vllm_settings_are_module_constants_parsed_at_import(
    reload_configuration_with_vllm_environment: Callable[..., object],
) -> None:
    configuration = reload_configuration_with_vllm_environment(
        VLLM_PROXY_ENABLED="true", VLLM_REQUEST_TIMEOUT_S="7.5"
    )

    assert configuration.VLLM_PROXY_ENABLED is True
    assert configuration.VLLM_REQUEST_TIMEOUT_S == 7.5
    assert configuration.VLLM_BASE_URL == "http://127.0.0.1:8000"
    assert "VLLM_MAX_LORA_RANK" in dir(configuration)
    configuration.validate_vllm_proxy_settings()


def test_invalid_vllm_flag_is_kept_raw_and_rejected_on_validation(
    reload_configuration_with_vllm_environment: Callable[..., object],
) -> None:
    configuration = reload_configuration_with_vllm_environment(
        VLLM_PROXY_ENABLED="maybe"
    )

    assert configuration.VLLM_PROXY_ENABLED == "maybe"
    with pytest.raises(InvalidEnvVariable, match="VLLM_PROXY_ENABLED"):
        configuration.validate_vllm_proxy_settings()


def test_disabled_backends_environment_is_parsed_but_not_applied_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DISABLED_INFERENCE_MODELS_BACKENDS", "trt,onnx")
    try:
        configuration = importlib.reload(inference_models.configuration)
        assert configuration.DISABLED_INFERENCE_MODELS_BACKENDS == {"trt", "onnx"}
    finally:
        monkeypatch.delenv("DISABLED_INFERENCE_MODELS_BACKENDS")
        importlib.reload(inference_models.configuration)
