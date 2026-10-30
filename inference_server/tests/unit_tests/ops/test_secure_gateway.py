from types import SimpleNamespace

import pytest
import requests

from inference_server import configuration
from inference_server.ops import secure_gateway
from tests.unit_tests.ops.conftest import running_on_event_loop

GATEWAY = "https://gateway.internal.example:8443"
RESPONSE_KEYS = ["status", "reason", "gateway_status_code", "latency_ms"]


@pytest.fixture
def gateway_client(ops_client, monkeypatch):
    def _build(outcome, gateway=GATEWAY, **overrides):
        calls = []

        def _fake_get(url, **kwargs):
            calls.append((url, kwargs, running_on_event_loop()))
            if isinstance(outcome, Exception):
                raise outcome

            return outcome

        monkeypatch.setattr(secure_gateway.requests, "get", _fake_get)
        monkeypatch.setattr(
            secure_gateway.models_configuration, "SECURE_GATEWAY", gateway
        )
        client = ops_client(SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED=True, **overrides)

        return client, calls

    return _build


def _answer(status_code: int, headers=None) -> SimpleNamespace:
    return SimpleNamespace(status_code=status_code, headers=headers or {})


def test_route_is_not_registered_when_flag_is_off(ops_client):
    response = ops_client().get("/secure-gateway/health")

    assert response.status_code == 404
    assert response.content == b'{"detail":"Not Found"}'


def test_healthy_gateway_answers_200(gateway_client):
    client, calls = gateway_client(_answer(204))

    response = client.get("/secure-gateway/health")

    body = response.json()
    assert response.status_code == 200
    assert list(body) == RESPONSE_KEYS
    assert body["status"] == "healthy"
    assert body["reason"] is None
    assert body["gateway_status_code"] == 204
    assert isinstance(body["latency_ms"], float)
    assert calls == [
        (
            f"{GATEWAY}/health",
            {
                "timeout": 5.0,
                "verify": True,
                "allow_redirects": False,
                "headers": {
                    "User-Agent": f"roboflow-inference/{configuration.SERVER_VERSION}"
                },
            },
            False,
        )
    ]


def test_probe_honours_timeout_and_verify_settings(gateway_client):
    client, calls = gateway_client(
        _answer(200),
        SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT=1.5,
        ROBOFLOW_API_VERIFY_SSL=False,
    )

    client.get("/secure-gateway/health")

    assert calls[0][1]["timeout"] == 1.5
    assert calls[0][1]["verify"] is False


def test_unconfigured_gateway_answers_404_without_probing(gateway_client):
    client, calls = gateway_client(_answer(200), gateway=None)

    response = client.get("/secure-gateway/health")

    assert response.status_code == 404
    assert response.content == (
        b'{"status":"not_configured","reason":null,'
        b'"gateway_status_code":null,"latency_ms":null}'
    )
    assert calls == []


@pytest.mark.parametrize(
    "outcome,status_code,reason,gateway_status_code",
    [
        (_answer(500), 502, "gateway_error", 500),
        (_answer(404), 502, "gateway_error", 404),
        (
            _answer(301, {"Location": "https://elsewhere.example/health"}),
            502,
            "unexpected_redirect",
            301,
        ),
        (
            requests.exceptions.SSLError(f"bad cert for {GATEWAY}"),
            503,
            "tls_error",
            None,
        ),
        (
            requests.exceptions.ConnectionError(f"refused by {GATEWAY}"),
            503,
            "connection_error",
            None,
        ),
        (requests.exceptions.ConnectTimeout(f"slow {GATEWAY}"), 504, "timeout", None),
        (requests.exceptions.ReadTimeout(f"slow {GATEWAY}"), 504, "timeout", None),
        (
            requests.exceptions.InvalidURL(f"bad url {GATEWAY}"),
            503,
            "request_error",
            None,
        ),
    ],
)
def test_unhealthy_outcomes_map_to_legacy_statuses(
    gateway_client, outcome, status_code, reason, gateway_status_code
):
    client, _ = gateway_client(outcome)

    response = client.get("/secure-gateway/health")

    body = response.json()
    assert response.status_code == status_code
    assert list(body) == RESPONSE_KEYS
    assert body["status"] == "unhealthy"
    assert body["reason"] == reason
    assert body["gateway_status_code"] == gateway_status_code
    assert (body["latency_ms"] is None) is (gateway_status_code is None)
    assert "gateway.internal.example" not in response.text
    assert "elsewhere.example" not in response.text


def test_failed_probe_leaves_no_gateway_address_in_the_log_buffer(
    gateway_client, log_buffer
):
    client, _ = gateway_client(requests.exceptions.SSLError(f"bad cert for {GATEWAY}"))

    client.get("/secure-gateway/health")

    buffered = log_buffer()
    assert "reason=tls_error" in buffered
    assert "SSLError" in buffered
    assert "gateway.internal.example" not in buffered
    assert "8443" not in buffered


def test_gateway_error_log_line_carries_the_upstream_status(gateway_client, log_buffer):
    client, _ = gateway_client(_answer(500))

    client.get("/secure-gateway/health")

    buffered = log_buffer()
    assert "reason=gateway_error" in buffered
    assert "500" in buffered


def test_gateway_base_is_the_normalised_value_used_for_proxying(monkeypatch):
    monkeypatch.setattr(
        secure_gateway.models_configuration, "SECURE_GATEWAY", f"{GATEWAY}/"
    )

    assert secure_gateway.get_secure_gateway_base_url() == GATEWAY

    monkeypatch.setattr(secure_gateway.models_configuration, "SECURE_GATEWAY", None)

    assert secure_gateway.get_secure_gateway_base_url() is None
