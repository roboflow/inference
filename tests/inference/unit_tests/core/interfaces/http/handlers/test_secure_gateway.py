import pytest
import requests
from requests_mock import Mocker

from inference.core.entities.responses.secure_gateway import SecureGatewayHealthResponse
from inference.core.interfaces.http.handlers.secure_gateway import (
    probe_secure_gateway_health,
)

GATEWAY_HOST = "gateway.local"
GATEWAY = f"http://{GATEWAY_HOST}:8080"
HEALTH_URL = f"{GATEWAY}/health"
GATEWAY_HEALTH_BODY = {"status": "healthy", "timestamp": "2026-09-02T10:00:00.000Z"}
RESPONSE_KEYS = {"status", "reason", "gateway_status_code", "latency_ms"}


def test_probe_when_gateway_is_not_configured() -> None:
    # when
    status_code, payload = probe_secure_gateway_health(
        gateway_base_url=None, timeout=1.0, verify_ssl=True
    )

    # then
    assert status_code == 404
    assert payload.status == "not_configured"
    assert payload.reason is None
    assert payload.gateway_status_code is None
    assert payload.latency_ms is None


def test_probe_when_gateway_is_healthy(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(HEALTH_URL, json=GATEWAY_HEALTH_BODY)

    # when
    status_code, payload = probe_secure_gateway_health(
        gateway_base_url=GATEWAY, timeout=2.5, verify_ssl=False
    )

    # then
    assert status_code == 200
    assert payload.status == "healthy"
    assert payload.reason is None
    assert payload.gateway_status_code == 200
    assert payload.latency_ms is not None and payload.latency_ms >= 0


def test_probe_requests_gateway_health_directly_with_configured_options(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(HEALTH_URL, json=GATEWAY_HEALTH_BODY)

    # when
    probe_secure_gateway_health(gateway_base_url=GATEWAY, timeout=2.5, verify_ssl=False)

    # then - the probe must hit the gateway itself, never the /proxy?url= route
    # (both proxies unwrap a self-referential proxy URL to api.roboflow.com)
    request = requests_mock.last_request
    assert request.url == HEALTH_URL
    assert "/proxy?url=" not in request.url
    assert request.timeout == 2.5
    assert request.verify is False
    # allow_redirects is consumed by requests.Session.send() before the adapter
    # sees it, so it is pinned behaviourally in test_probe_when_gateway_redirects
    assert request.headers["User-Agent"].startswith("roboflow-inference/")


def test_probe_when_gateway_answers_with_non_json_body(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(HEALTH_URL, text="OK")

    # when
    status_code, payload = probe_secure_gateway_health(
        gateway_base_url=GATEWAY, timeout=1.0, verify_ssl=True
    )

    # then - any 2xx is healthy; the body is neither parsed nor relayed
    assert status_code == 200
    assert payload.status == "healthy"


@pytest.mark.parametrize("gateway_status", [400, 401, 404, 500, 503])
def test_probe_when_gateway_answers_with_error(
    requests_mock: Mocker, gateway_status: int
) -> None:
    # given
    requests_mock.get(HEALTH_URL, status_code=gateway_status, text="nope")

    # when
    status_code, payload = probe_secure_gateway_health(
        gateway_base_url=GATEWAY, timeout=1.0, verify_ssl=True
    )

    # then
    assert status_code == 502
    assert payload.status == "unhealthy"
    assert payload.reason == "gateway_error"
    assert payload.gateway_status_code == gateway_status
    assert payload.latency_ms is not None


def test_probe_when_gateway_redirects(requests_mock: Mocker) -> None:
    # given - TLS-enabled secure gateway with HTTP_REDIRECT_PORT answers 301 on http
    requests_mock.get(
        HEALTH_URL,
        status_code=301,
        headers={"Location": f"https://{GATEWAY_HOST}/health"},
    )

    # when
    status_code, payload = probe_secure_gateway_health(
        gateway_base_url=GATEWAY, timeout=1.0, verify_ssl=True
    )

    # then - redirect is reported, not followed
    assert status_code == 502
    assert payload.status == "unhealthy"
    assert payload.reason == "unexpected_redirect"
    assert payload.gateway_status_code == 301
    assert requests_mock.call_count == 1


@pytest.mark.parametrize(
    "error, expected_status_code, expected_reason",
    [
        # SSLError subclasses ConnectionError - pins the except-arm order
        (requests.exceptions.SSLError("certificate verify failed"), 503, "tls_error"),
        # ConnectTimeout subclasses BOTH ConnectionError and Timeout
        (requests.exceptions.ConnectTimeout("connect timed out"), 504, "timeout"),
        (requests.exceptions.ReadTimeout("read timed out"), 504, "timeout"),
        (
            requests.exceptions.ConnectionError("connection refused"),
            503,
            "connection_error",
        ),
        (requests.exceptions.TooManyRedirects("loop"), 503, "request_error"),
        (requests.exceptions.InvalidURL("bad url"), 503, "request_error"),
    ],
)
def test_probe_when_request_fails(
    requests_mock: Mocker,
    error: Exception,
    expected_status_code: int,
    expected_reason: str,
) -> None:
    # given
    requests_mock.get(HEALTH_URL, exc=error)

    # when
    status_code, payload = probe_secure_gateway_health(
        gateway_base_url=GATEWAY, timeout=1.0, verify_ssl=True
    )

    # then
    assert status_code == expected_status_code
    assert payload.status == "unhealthy"
    assert payload.reason == expected_reason
    assert payload.gateway_status_code is None
    assert payload.latency_ms is None


@pytest.mark.parametrize(
    "mock_kwargs",
    [
        {"json": GATEWAY_HEALTH_BODY},
        {"status_code": 503, "text": f"upstream {GATEWAY_HOST} down"},
        {"status_code": 301, "headers": {"Location": f"https://{GATEWAY_HOST}/health"}},
        {
            "exc": requests.exceptions.ConnectionError(
                f"Failed to establish a new connection to {HEALTH_URL}"
            )
        },
        {"exc": requests.exceptions.SSLError(f"hostname {GATEWAY_HOST} mismatch")},
        {"exc": requests.exceptions.ConnectTimeout(f"{HEALTH_URL} timed out")},
    ],
)
def test_probe_response_never_identifies_the_gateway(
    requests_mock: Mocker, mock_kwargs: dict
) -> None:
    """The gateway address is operator configuration and must not leak to
    callers - not via a URL field, not via an upstream body, not via
    exception text."""
    # given
    requests_mock.get(HEALTH_URL, **mock_kwargs)

    # when
    _, payload = probe_secure_gateway_health(
        gateway_base_url=GATEWAY, timeout=1.0, verify_ssl=True
    )

    # then
    serialized = payload.model_dump_json()
    assert GATEWAY_HOST not in serialized
    assert "8080" not in serialized
    assert "timestamp" not in serialized
    assert set(payload.model_dump().keys()) == RESPONSE_KEYS


def test_response_model_has_only_the_contract_fields() -> None:
    assert set(SecureGatewayHealthResponse.model_fields.keys()) == RESPONSE_KEYS
