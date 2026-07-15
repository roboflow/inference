import importlib
from unittest import mock
from urllib.parse import parse_qs, urlparse

import inference.core.utils.url_utils as url_utils
from inference.usage_tracking import config as telemetry_config


def test_telemetry_endpoints_are_not_wrapped_without_secure_gateway():
    # when
    with mock.patch.object(url_utils, "SECURE_GATEWAY", None):
        settings = telemetry_config.TelemetrySettings()

    # then
    assert settings.api_usage_endpoint_url.endswith("/usage/inference")
    assert settings.api_plan_endpoint_url.endswith("/usage/plan")
    assert settings.webrtc_plans_endpoint_url.endswith("/webrtc_plans")
    assert "/proxy?url=" not in settings.api_usage_endpoint_url


def test_telemetry_endpoints_are_wrapped_with_secure_gateway():
    # when
    with mock.patch.object(url_utils, "SECURE_GATEWAY", "gateway.local:8080"):
        settings = telemetry_config.TelemetrySettings()

    # then
    assert settings.api_usage_endpoint_url.startswith(
        "http://gateway.local:8080/proxy?url="
    )
    assert settings.api_plan_endpoint_url.startswith(
        "http://gateway.local:8080/proxy?url="
    )
    assert settings.webrtc_plans_endpoint_url.startswith(
        "http://gateway.local:8080/proxy?url="
    )


def test_telemetry_endpoint_env_overrides_are_wrapped_with_secure_gateway(monkeypatch):
    # given
    monkeypatch.setenv(
        "TELEMETRY_API_USAGE_ENDPOINT_URL", "https://custom.example.com/usage"
    )

    # when
    with mock.patch.object(url_utils, "SECURE_GATEWAY", "https://gateway.local"):
        settings = telemetry_config.TelemetrySettings()

    # then - env override is routed through the gateway too
    parsed = urlparse(settings.api_usage_endpoint_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://gateway.local/proxy"
    )
    proxied_url = parse_qs(parsed.query)["url"][0]
    assert proxied_url == "https://custom.example.com/usage"
