from unittest.mock import Mock, call

import pytest
import requests
import requests_mock as rm
from roboflow_workflows.prototypes.platform_errors import (
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)

import inference_server.workflows.host as host

OCR = "apiproxy/google_vision_ocr"


@pytest.fixture
def sleep(monkeypatch):
    sleeper = Mock()
    clock = Mock(wraps=host.time)
    clock.sleep = sleeper
    monkeypatch.setattr(host, "time", clock)
    return sleeper


@pytest.mark.parametrize("status", [502, 503, 504])
def test_ocr_retries_gateway_errors(requests_mock, sleep, status):
    requests_mock.post(rm.ANY, [{"status_code": status}, {"json": {"ok": True}}])
    assert host.PLATFORM_CLIENT.post(OCR, "key") == {"ok": True}
    assert requests_mock.call_count == 2
    assert sleep.call_args_list == [call(1)]


def test_exhaustion_and_handler(requests_mock, sleep):
    requests_mock.post(rm.ANY, status_code=503)
    handler = Mock()
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError) as exc:
        host.PLATFORM_CLIENT.post(
            OCR, "secret-key", http_errors_handlers={503: handler}
        )
    assert requests_mock.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    handler.assert_called_once()
    assert "secret-key" not in str(exc.value)


@pytest.mark.parametrize(
    "endpoint,status,expected",
    [
        (OCR, 400, RoboflowAPIUnsuccessfulRequestError),
        (OCR, 401, RoboflowAPINotAuthorizedError),
        (OCR, 403, RoboflowAPIForbiddenError),
        (OCR, 429, RoboflowAPIUnsuccessfulRequestError),
        (OCR, 500, RoboflowAPIUnsuccessfulRequestError),
        ("apiproxy/other", 503, RoboflowAPIUnsuccessfulRequestError),
        ("apiproxy/email", 503, RoboflowAPIUnsuccessfulRequestError),
        ("apiproxy/twilio", 503, RoboflowAPIUnsuccessfulRequestError),
    ],
)
def test_other_errors_and_endpoints_are_not_retried(
    requests_mock, sleep, endpoint, status, expected
):
    requests_mock.post(rm.ANY, status_code=status)
    with pytest.raises(expected):
        host.PLATFORM_CLIENT.post(endpoint, "key")
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "error,expected",
    [
        (requests.Timeout, RoboflowAPITimeoutError),
        (requests.ConnectionError, RoboflowAPIConnectionError),
    ],
)
def test_transport_errors_are_not_retried(requests_mock, sleep, error, expected):
    requests_mock.post(rm.ANY, exc=error)
    with pytest.raises(expected):
        host.PLATFORM_CLIENT.post(OCR, "key")
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_request_is_preserved(requests_mock, sleep, monkeypatch):
    from inference_models.weights_providers import roboflow
    from inference_server import configuration

    monkeypatch.setattr(configuration, "API_PROXY_BASE_URL", "https://proxy.example")
    monkeypatch.setattr(configuration, "ROBOFLOW_API_VERIFY_SSL", False)
    monkeypatch.setattr(
        configuration, "ROBOFLOW_API_EXTRA_HEADERS", '{"X-Test": "value"}'
    )
    monkeypatch.setattr(roboflow, "SECURE_GATEWAY", "https://gateway.example")
    requests_mock.post(
        rm.ANY, [{"status_code": 503}, {"status_code": 504}, {"json": {"ok": True}}]
    )
    assert host.PLATFORM_CLIENT.post(
        "/" + OCR + "/", "key", payload={"a": 1}, params=[("x", "y")]
    ) == {"ok": True}
    assert requests_mock.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    for request in requests_mock.request_history:
        assert request.url == host.PLATFORM_CLIENT.wrap_url(
            "https://proxy.example/apiproxy/google_vision_ocr?api_key=key&x=y"
        )
        assert request.json() == {"a": 1}
        assert request.headers["X-Allow-Chunked"] == "true"
        assert request.headers["X-Test"] == "value"
        assert request.timeout == host.API_REQUEST_TIMEOUT_S
        assert request.verify is False


def test_invalid_json_is_not_retried(requests_mock, sleep):
    requests_mock.post(rm.ANY, text="not json")
    with pytest.raises(requests.exceptions.JSONDecodeError):
        host.PLATFORM_CLIENT.post(OCR, "key")
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_success_does_not_retry(requests_mock, sleep):
    requests_mock.post(rm.ANY, json={"ok": True})
    assert host.PLATFORM_CLIENT.post(OCR, "key") == {"ok": True}
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("module_name", ["v1", "v1_tensor"])
def test_invalid_ocr_response_structure_does_not_retry(
    requests_mock, sleep, module_name
):
    from importlib import import_module

    module = import_module(
        "roboflow_workflows.core_steps.models.foundation.google_vision_ocr."
        + module_name
    )
    requests_mock.post(rm.ANY, json={"responses": []})
    with pytest.raises(RuntimeError, match="Invalid response structure"):
        module._execute_proxied_google_vision_request(
            roboflow_api_key="key",
            platform_client=host.PLATFORM_CLIENT,
            google_vision_api_key="google-key",
            request_json={},
        )
    assert requests_mock.call_count == 1
    sleep.assert_not_called()
