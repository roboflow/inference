from unittest.mock import Mock, call

import pytest
import requests
import requests_mock as requests_mock_lib

from inference.core import roboflow_api
from inference.core.exceptions import (
    MalformedRoboflowAPIResponseError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)
from inference.core.utils import url_utils

OCR = "apiproxy/google_vision_ocr"


@pytest.fixture
def sleep(monkeypatch):
    sleeper = Mock()
    clock = Mock(wraps=roboflow_api.time)
    clock.sleep = sleeper
    monkeypatch.setattr(roboflow_api, "time", clock)
    return sleeper


@pytest.mark.parametrize("status", [502, 503, 504])
def test_ocr_retries_gateway_errors(requests_mock, sleep, status):
    requests_mock.post(
        requests_mock_lib.ANY, [{"status_code": status}, {"json": {"ok": True}}]
    )
    result = roboflow_api.post_to_roboflow_api(OCR, "key", payload={"image": "data"})
    assert result == {"ok": True}
    assert requests_mock.call_count == 2
    sleep.assert_has_calls([call(1)])


def test_exhaustion_redacts_key_and_waits(requests_mock, sleep):
    requests_mock.post(requests_mock_lib.ANY, status_code=503)
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError) as exc:
        roboflow_api.post_to_roboflow_api(OCR, "secret-key")
    assert requests_mock.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    error = exc.value
    while error is not None:
        assert "secret-key" not in str(error)
        error = error.__cause__


def test_success_does_not_retry(requests_mock, sleep):
    requests_mock.post(requests_mock_lib.ANY, json={"ok": True})
    assert roboflow_api.post_to_roboflow_api(OCR, "key") == {"ok": True}
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "status,expected",
    [
        (400, RoboflowAPIUnsuccessfulRequestError),
        (401, RoboflowAPINotAuthorizedError),
        (403, RoboflowAPIForbiddenError),
        (429, RoboflowAPIUnsuccessfulRequestError),
        (500, RoboflowAPIUnsuccessfulRequestError),
    ],
)
def test_permanent_errors_do_not_retry(requests_mock, sleep, status, expected):
    requests_mock.post(requests_mock_lib.ANY, status_code=status)
    with pytest.raises(expected):
        roboflow_api.post_to_roboflow_api(OCR, "key")
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_invalid_json_does_not_retry(requests_mock, sleep):
    requests_mock.post(requests_mock_lib.ANY, text="not json")
    with pytest.raises(MalformedRoboflowAPIResponseError):
        roboflow_api.post_to_roboflow_api(OCR, "key")
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "error, expected",
    [
        (requests.Timeout, RoboflowAPITimeoutError),
        (requests.ConnectionError, RoboflowAPIConnectionError),
    ],
)
def test_connection_errors_do_not_retry(
    requests_mock, sleep, monkeypatch, error, expected
):
    monkeypatch.setattr(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
    requests_mock.post(requests_mock_lib.ANY, exc=error)
    with pytest.raises(expected):
        roboflow_api.post_to_roboflow_api(OCR, "key")
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "endpoint",
    [
        "apiproxy/other",
        "apiproxy/email",
        "apiproxy/twilio",
        "apiproxy/google_vision_ocr/extra",
    ],
)
def test_other_endpoints_do_not_retry(requests_mock, sleep, endpoint):
    requests_mock.post(requests_mock_lib.ANY, status_code=503)
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        roboflow_api.post_to_roboflow_api(endpoint, "key")
    assert requests_mock.call_count == 1
    sleep.assert_not_called()


def test_custom_handler_runs_after_exhaustion(requests_mock, sleep):
    requests_mock.post(requests_mock_lib.ANY, status_code=503)
    handler = Mock()
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        roboflow_api.post_to_roboflow_api(
            OCR, "key", http_errors_handlers={503: handler}
        )
    assert requests_mock.call_count == 3
    handler.assert_called_once()


def test_retries_preserve_request(requests_mock, sleep, monkeypatch):
    monkeypatch.setattr(roboflow_api, "API_PROXY_BASE_URL", "https://proxy.example")
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_REQUEST_TIMEOUT", 37)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_VERIFY_SSL", False)
    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", "gateway.example")
    requests_mock.post(
        requests_mock_lib.ANY,
        [{"status_code": 503}, {"status_code": 504}, {"json": {"ok": True}}],
    )
    result = roboflow_api.post_to_roboflow_api(
        "/" + OCR + "/", "key", payload={"a": 1}, params=[("x", "y")]
    )
    assert result == {"ok": True}
    assert requests_mock.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]
    for request in requests_mock.request_history:
        assert request.url == url_utils.wrap_url(
            "https://proxy.example/apiproxy/google_vision_ocr?api_key=key&x=y"
        )
        assert request.json() == {"a": 1}
        assert request.headers["X-Allow-Chunked"] == "true"
        assert request.timeout == 37
        assert request.verify is False


@pytest.mark.parametrize("module_name", ["v1", "v1_tensor"])
def test_ocr_helpers_use_server_adapter(requests_mock, sleep, module_name):
    from importlib import import_module

    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )

    module = import_module(
        "roboflow_workflows.core_steps.models.foundation.google_vision_ocr."
        + module_name
    )
    requests_mock.post(
        requests_mock_lib.ANY,
        [{"status_code": 503}, {"json": {"responses": [{"text": "found"}]}}],
    )
    result = module._execute_proxied_google_vision_request(
        roboflow_api_key="key",
        platform_client=ServerRoboflowPlatformClient(),
        google_vision_api_key="google-key",
        request_json={"requests": []},
    )
    assert result == {"text": "found"}
    assert requests_mock.call_count == 2
    assert requests_mock.request_history[0].json() == {
        "google_vision_api_key": "google-key",
        "request_json": {"requests": []},
    }


@pytest.mark.parametrize("module_name", ["v1", "v1_tensor"])
def test_invalid_ocr_response_structure_does_not_retry(
    requests_mock, sleep, module_name
):
    from importlib import import_module

    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )

    module = import_module(
        "roboflow_workflows.core_steps.models.foundation.google_vision_ocr."
        + module_name
    )
    requests_mock.post(requests_mock_lib.ANY, json={"responses": []})
    with pytest.raises(RuntimeError, match="Invalid response structure"):
        module._execute_proxied_google_vision_request(
            roboflow_api_key="key",
            platform_client=ServerRoboflowPlatformClient(),
            google_vision_api_key="google-key",
            request_json={},
        )
    assert requests_mock.call_count == 1
    sleep.assert_not_called()
