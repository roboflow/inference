import importlib
import os
from typing import Callable, Tuple

import pytest

import inference_cli.lib.enterprise.inference_compiler.constants
import inference_cli.lib.env

ENVIRONMENT_KEYS = [
    "ROBOFLOW_REGION",
    "PROJECT",
    "API_BASE_URL",
    "ROBOFLOW_ENVIRONMENT",
    "ROBOFLOW_API_HOST",
]


@pytest.fixture
def reload_env_modules() -> Callable[..., Tuple[object, object]]:
    saved_environment = {
        key: os.environ.pop(key) for key in ENVIRONMENT_KEYS if key in os.environ
    }

    def _reload(**environment: str) -> Tuple[object, object]:
        for key in ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(environment)
        env_module = importlib.reload(inference_cli.lib.env)
        constants_module = importlib.reload(
            inference_cli.lib.enterprise.inference_compiler.constants
        )
        return env_module, constants_module

    try:
        yield _reload
    finally:
        for key in ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(saved_environment)
        importlib.reload(inference_cli.lib.env)
        importlib.reload(inference_cli.lib.enterprise.inference_compiler.constants)


def test_api_urls_default_to_us_production(reload_env_modules) -> None:
    # when
    env_module, constants_module = reload_env_modules()

    # then
    assert env_module.ROBOFLOW_REGION == "us"
    assert env_module.API_BASE_URL == "https://api.roboflow.com"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.com"


@pytest.mark.parametrize(
    "region, environment, expected_api_url",
    [
        ("us", "prod", "https://api.roboflow.com"),
        ("us", "staging", "https://api.roboflow.one"),
        ("eu", "prod", "https://api.roboflow.eu"),
        ("eu", "staging", "https://api.roboflow-eu.one"),
    ],
)
def test_api_urls_follow_region_and_environment_matrix(
    reload_env_modules,
    region: str,
    environment: str,
    expected_api_url: str,
) -> None:
    # when
    env_module, constants_module = reload_env_modules(
        ROBOFLOW_REGION=region, ROBOFLOW_ENVIRONMENT=environment
    )

    # then
    assert env_module.API_BASE_URL == expected_api_url
    assert constants_module.ROBOFLOW_API_HOST == expected_api_url


def test_region_value_is_normalized(reload_env_modules) -> None:
    # when
    env_module, _ = reload_env_modules(ROBOFLOW_REGION=" EU ")

    # then
    assert env_module.ROBOFLOW_REGION == "eu"
    assert env_module.API_BASE_URL == "https://api.roboflow.eu"


def test_explicit_url_overrides_beat_region_and_environment(
    reload_env_modules,
) -> None:
    # when
    env_module, constants_module = reload_env_modules(
        ROBOFLOW_REGION="eu",
        ROBOFLOW_ENVIRONMENT="staging",
        API_BASE_URL="https://api.example.com",
        ROBOFLOW_API_HOST="https://api-host.example.com",
    )

    # then
    assert env_module.API_BASE_URL == "https://api.example.com"
    assert constants_module.ROBOFLOW_API_HOST == "https://api-host.example.com"


def test_unknown_region_warns_and_falls_back_to_us(reload_env_modules) -> None:
    # when
    with pytest.warns(UserWarning, match="Unknown ROBOFLOW_REGION"):
        env_module, constants_module = reload_env_modules(ROBOFLOW_REGION="mars")

    # then
    assert env_module.ROBOFLOW_REGION == "us"
    assert env_module.API_BASE_URL == "https://api.roboflow.com"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.com"


def test_legacy_project_variable_still_selects_staging(reload_env_modules) -> None:
    # when
    env_module, constants_module = reload_env_modules(PROJECT="roboflow-staging")

    # then
    assert env_module.API_BASE_URL == "https://api.roboflow.one"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.one"


def test_legacy_project_variable_selects_staging_within_eu_region(
    reload_env_modules,
) -> None:
    # when
    env_module, constants_module = reload_env_modules(
        ROBOFLOW_REGION="eu", PROJECT="roboflow-staging"
    )

    # then
    assert env_module.API_BASE_URL == "https://api.roboflow-eu.one"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow-eu.one"


def test_roboflow_environment_beats_legacy_project_variable(
    reload_env_modules,
) -> None:
    # when
    env_module, constants_module = reload_env_modules(
        ROBOFLOW_ENVIRONMENT="prod", PROJECT="roboflow-staging"
    )

    # then
    assert env_module.API_BASE_URL == "https://api.roboflow.com"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.com"
