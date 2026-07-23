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


def test_api_urls_honor_eu_region(reload_env_modules) -> None:
    # when
    env_module, constants_module = reload_env_modules(ROBOFLOW_REGION="eu")

    # then
    assert env_module.ROBOFLOW_REGION == "eu"
    assert env_module.API_BASE_URL == "https://api.roboflow.eu"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.eu"


def test_region_value_is_normalized(reload_env_modules) -> None:
    # when
    env_module, _ = reload_env_modules(ROBOFLOW_REGION=" EU ")

    # then
    assert env_module.ROBOFLOW_REGION == "eu"
    assert env_module.API_BASE_URL == "https://api.roboflow.eu"


def test_explicit_url_overrides_beat_region(reload_env_modules) -> None:
    # when
    env_module, constants_module = reload_env_modules(
        ROBOFLOW_REGION="eu",
        API_BASE_URL="https://api.example.com",
        ROBOFLOW_API_HOST="https://api-host.example.com",
    )

    # then
    assert env_module.API_BASE_URL == "https://api.example.com"
    assert constants_module.ROBOFLOW_API_HOST == "https://api-host.example.com"


def test_unknown_region_warns_and_falls_back_to_us(reload_env_modules, capsys) -> None:
    # when
    env_module, constants_module = reload_env_modules(ROBOFLOW_REGION="mars")

    # then
    assert env_module.ROBOFLOW_REGION == "us"
    assert env_module.API_BASE_URL == "https://api.roboflow.com"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.com"
    assert "unknown Roboflow region" in capsys.readouterr().err


def test_non_platform_project_still_selects_staging_api(reload_env_modules) -> None:
    # when
    env_module, constants_module = reload_env_modules(
        PROJECT="roboflow-staging", ROBOFLOW_ENVIRONMENT="staging"
    )

    # then
    assert env_module.API_BASE_URL == "https://api.roboflow.one"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.one"


def test_eu_region_beats_project_and_environment_defaults(reload_env_modules) -> None:
    # when
    env_module, constants_module = reload_env_modules(
        ROBOFLOW_REGION="eu", PROJECT="roboflow-staging", ROBOFLOW_ENVIRONMENT="staging"
    )

    # then
    assert env_module.API_BASE_URL == "https://api.roboflow.eu"
    assert constants_module.ROBOFLOW_API_HOST == "https://api.roboflow.eu"
