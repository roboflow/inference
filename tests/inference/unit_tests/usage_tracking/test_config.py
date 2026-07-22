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


def test_ssl_verify_for_endpoint_judges_the_request_host_not_the_embedded_target():
    from inference.usage_tracking.utils import ssl_verify_for_endpoint

    # local development endpoints skip verification
    assert ssl_verify_for_endpoint("http://localhost:8080/usage") is False
    assert ssl_verify_for_endpoint("https://127.0.0.1/usage") is False
    # real hosts verify
    assert ssl_verify_for_endpoint("https://api.roboflow.com/usage") is True
    # a gateway-wrapped localhost target is judged by the gateway host
    assert (
        ssl_verify_for_endpoint(
            "https://gateway.local/proxy?url=http%3A%2F%2Flocalhost%3A9000%2Fusage"
        )
        is True
    )
