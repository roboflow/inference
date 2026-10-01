import logging

import pytest
import requests
from requests_mock import ANY, Mocker

from inference.core import roboflow_api
from inference.core.exceptions import RoboflowAPIRequestError
from inference.core.utils import url_utils

_API_URL = "https://retry-api.example.test"
_URLS = {
    "get": f"{_API_URL}/example",
    "upload": f"{_API_URL}/dataset/project/upload",
    "annotation": f"{_API_URL}/dataset/project/annotate/image-id",
}
_SUCCESS = {"success": True, "id": "image-id"}
_FAILURES = [
    pytest.param({"status_code": 503}, id="http-503"),
    pytest.param({"exc": requests.exceptions.ConnectionError}, id="connection"),
    pytest.param({"exc": requests.exceptions.Timeout}, id="timeout"),
]


@pytest.fixture(autouse=True)
def configure_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(roboflow_api, "API_BASE_URL", _API_URL)
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", "")


def _call_api(operation: str, *, enable_retries: bool = False) -> dict:
    retry_options = {"enable_retries": True} if enable_retries else {}
    if operation == "get":
        response = roboflow_api.get_from_url(f"{_URLS['get']}?api_key=test-key")
    elif operation == "upload":
        response = roboflow_api.register_image_at_roboflow(
            api_key="test-key",
            dataset_id="project",
            local_image_id="local-id",
            image_bytes=b"jpeg image",
            batch_name="batch",
            **retry_options,
        )
    else:
        response = roboflow_api.annotate_image_at_roboflow(
            api_key="test-key",
            dataset_id="project",
            local_image_id="local-id",
            roboflow_image_id="image-id",
            annotation_content="cat",
            annotation_file_type="txt",
            **retry_options,
        )

    return response


@pytest.mark.parametrize("operation", ["upload", "annotation"])
@pytest.mark.parametrize("failure", _FAILURES)
def test_upload_retries_require_opt_in(
    requests_mock: Mocker,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure: dict,
) -> None:
    monkeypatch.setattr(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
    monkeypatch.setattr(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", {503})
    request = requests_mock.post(_URLS[operation], [failure, {"json": _SUCCESS}])

    with pytest.raises(RoboflowAPIRequestError):
        _call_api(operation)

    assert request.call_count == 1


@pytest.mark.parametrize("enable_retries", [False, True])
@pytest.mark.parametrize("failure", _FAILURES)
def test_get_retry_configuration_is_preserved(
    requests_mock: Mocker,
    monkeypatch: pytest.MonkeyPatch,
    enable_retries: bool,
    failure: dict,
) -> None:
    monkeypatch.setattr(
        roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", enable_retries
    )
    monkeypatch.setattr(
        roboflow_api,
        "TRANSIENT_ROBOFLOW_API_ERRORS",
        {503} if enable_retries else set(),
    )
    request = requests_mock.get(_URLS["get"], [failure, {"json": _SUCCESS}])

    if enable_retries:
        assert _call_api("get") == _SUCCESS
    else:
        with pytest.raises(RoboflowAPIRequestError):
            _call_api("get")

    assert request.call_count == (2 if enable_retries else 1)


@pytest.mark.parametrize("operation", ["get", "upload", "annotation"])
def test_retries_preserve_gateway_and_transport_settings(
    requests_mock: Mocker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operation: str,
) -> None:
    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", "https://gateway.example.test")
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_REQUEST_TIMEOUT", 7)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_VERIFY_SSL", "/test/ca.pem")
    monkeypatch.setattr(
        roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", '{"X-Test": "value"}'
    )
    monkeypatch.setattr(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", {503})
    caplog.set_level(logging.INFO, logger="backoff")
    method = "GET" if operation == "get" else "POST"
    request = requests_mock.register_uri(
        method, ANY, [{"status_code": 503}, {"json": _SUCCESS}]
    )

    result = _call_api(operation, enable_retries=True)

    assert result == _SUCCESS
    assert request.call_count == 2
    assert "test-key" not in caplog.text
    for attempt in request.request_history:
        assert attempt.url.startswith("https://gateway.example.test/proxy?")
        assert attempt.qs["url"][0].startswith(_URLS[operation])
        assert attempt.headers["X-Test"] == "value"
        assert attempt.timeout == 7
        assert attempt.verify == "/test/ca.pem"
