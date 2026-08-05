import pytest

from inference_sdk.regions import (
    ROBOFLOW_SERVICE_URLS,
    get_roboflow_environment,
    get_roboflow_region,
    resolve_roboflow_service_url,
)

REGION_ENVIRONMENT_KEYS = [
    "ROBOFLOW_REGION",
    "ROBOFLOW_ENVIRONMENT",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch) -> None:
    for key in REGION_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_region_defaults_to_us() -> None:
    # when
    result = get_roboflow_region()

    # then
    assert result == "us"


def test_region_value_is_normalized(monkeypatch) -> None:
    # given
    monkeypatch.setenv("ROBOFLOW_REGION", " EU ")

    # when
    result = get_roboflow_region()

    # then
    assert result == "eu"


def test_unknown_region_warns_and_falls_back_to_us(monkeypatch) -> None:
    # given
    monkeypatch.setenv("ROBOFLOW_REGION", "mars")

    # when
    with pytest.warns(UserWarning, match="Unknown ROBOFLOW_REGION"):
        result = get_roboflow_region()

    # then
    assert result == "us"


def test_environment_defaults_to_prod() -> None:
    # when
    result = get_roboflow_environment()

    # then
    assert result == "prod"


def test_environment_honors_roboflow_environment_variable(monkeypatch) -> None:
    # given
    monkeypatch.setenv("ROBOFLOW_ENVIRONMENT", "staging")

    # when
    result = get_roboflow_environment()

    # then
    assert result == "staging"


def test_environment_treats_any_non_prod_value_as_staging(monkeypatch) -> None:
    # given
    monkeypatch.setenv("ROBOFLOW_ENVIRONMENT", "dev")

    # when
    result = get_roboflow_environment()

    # then
    assert result == "staging"


def test_environment_falls_back_to_legacy_project_signal() -> None:
    # when
    result = get_roboflow_environment(project="roboflow-staging")

    # then
    assert result == "staging"


def test_roboflow_environment_variable_beats_legacy_project_signal(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setenv("ROBOFLOW_ENVIRONMENT", "prod")

    # when
    result = get_roboflow_environment(project="roboflow-staging")

    # then
    assert result == "prod"


@pytest.mark.parametrize(
    "region, environment, expected_api_url",
    [
        ("us", "prod", "https://api.roboflow.com"),
        ("us", "staging", "https://api.roboflow.one"),
        ("eu", "prod", "https://api.roboflow.eu"),
        ("eu", "staging", "https://api.roboflow-eu.one"),
    ],
)
def test_api_url_matrix(
    monkeypatch,
    region: str,
    environment: str,
    expected_api_url: str,
) -> None:
    # given
    monkeypatch.setenv("ROBOFLOW_REGION", region)
    monkeypatch.setenv("ROBOFLOW_ENVIRONMENT", environment)

    # when
    result = resolve_roboflow_service_url("api")

    # then
    assert result == expected_api_url


def test_every_region_environment_pair_defines_the_same_services() -> None:
    # given
    expected_services = {"api", "app", "serverless"}

    # then
    for services in ROBOFLOW_SERVICE_URLS.values():
        assert set(services) == expected_services
