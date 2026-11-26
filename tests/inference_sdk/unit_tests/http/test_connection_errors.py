from unittest import mock

import aiohttp
import numpy as np
import pytest
import requests
from aioresponses import aioresponses

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.errors import (
    HTTPCallErrorError,
    HTTPClientError,
    HTTPServerConnectionError,
    RetryError,
)


@pytest.mark.parametrize(
    "connection_error",
    [ConnectionError, requests.exceptions.ConnectionError],
)
def test_server_connection_error_preserves_cause(requests_mock, connection_error):
    http_client = InferenceHTTPClient(api_url="http://localhost:9001")
    original = connection_error("Connection refused")
    requests_mock.get("http://localhost:9001/model/registry", exc=original)

    with pytest.raises(HTTPServerConnectionError) as error:
        http_client.list_loaded_models()

    assert error.value.__cause__ is original
    assert "Connection refused" in str(error.value)


@pytest.mark.parametrize(
    "original, expected_type",
    [
        (ConnectionError("Connection refused"), HTTPServerConnectionError),
        (
            requests.exceptions.ConnectionError("Connection refused"),
            HTTPServerConnectionError,
        ),
        (
            requests.exceptions.ConnectTimeout("Connect timed out"),
            HTTPServerConnectionError,
        ),
        (requests.exceptions.ReadTimeout("Read timed out"), HTTPClientError),
        (requests.exceptions.Timeout("Request timed out"), HTTPClientError),
    ],
)
def test_workflow_retry_exhaustion_preserves_error_category(
    requests_mock, monkeypatch, original, expected_type
):
    monkeypatch.setattr("backoff._sync.time.sleep", lambda _: None)
    http_client = InferenceHTTPClient(api_url="http://localhost:9001")
    requests_mock.post(
        "http://localhost:9001/workspace/workflows/workflow", exc=original
    )

    with pytest.raises(HTTPClientError) as error:
        http_client.run_workflow(workspace_name="workspace", workflow_id="workflow")

    assert type(error.value) is expected_type
    assert requests_mock.call_count == 3
    assert isinstance(error.value.__cause__, RetryError)
    assert error.value.__cause__.inner_error is original
    assert str(original) in str(error.value)


def test_workflow_retry_exhaustion_keeps_http_status(requests_mock, monkeypatch):
    monkeypatch.setattr("backoff._sync.time.sleep", lambda _: None)
    http_client = InferenceHTTPClient(api_url="http://localhost:9001")
    requests_mock.post(
        "http://localhost:9001/workspace/workflows/workflow", status_code=503
    )

    with pytest.raises(HTTPCallErrorError) as error:
        http_client.run_workflow(workspace_name="workspace", workflow_id="workflow")

    assert error.value.status_code == 503
    assert requests_mock.call_count == 3


def test_workflow_connection_retry_error_redacts_api_key(requests_mock, monkeypatch):
    monkeypatch.setattr("backoff._sync.time.sleep", lambda _: None)
    http_client = InferenceHTTPClient(api_url="http://localhost:9001")
    requests_mock.post(
        "http://localhost:9001/workspace/workflows/workflow",
        exc=requests.exceptions.ConnectionError(
            "Connection refused: http://localhost:9001/?api_key=fake12345678"
        ),
    )

    with pytest.raises(HTTPServerConnectionError) as error:
        http_client.run_workflow(workspace_name="workspace", workflow_id="workflow")

    assert "fake12345678" not in str(error.value)
    assert "api_key=fa***78" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original, expected_type",
    [
        (
            aiohttp.ClientConnectionError("Connection refused"),
            HTTPServerConnectionError,
        ),
        (aiohttp.ServerTimeoutError("Read timed out"), HTTPClientError),
    ],
)
async def test_async_server_error_category(original, expected_type):
    http_client = InferenceHTTPClient(api_url="http://localhost:9001").configure(
        InferenceConfiguration(api_key_transport="header")
    )
    with aioresponses() as responses:
        responses.get("http://localhost:9001/model/registry", exception=original)

        with pytest.raises(HTTPClientError) as error:
            await http_client.list_loaded_models_async()

    assert type(error.value) is expected_type
    assert error.value.__cause__ is original
    assert str(original) in str(error.value)


def test_infer_image_download_failure_is_not_a_server_connection_error(requests_mock):
    http_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", api_key="key"
    )
    image_url = "https://images.example.com/image.jpg"
    requests_mock.get(image_url, exc=requests.exceptions.ConnectionError("refused"))

    with pytest.raises(HTTPClientError) as error:
        http_client.infer(inference_input=image_url, model_id="some/1")

    assert type(error.value) is HTTPClientError
    assert "Could not load image from URL" in str(error.value)
    assert "refused" in str(error.value)
    assert requests_mock.call_count == 1
    assert requests_mock.last_request.url == image_url


@pytest.mark.asyncio
async def test_infer_async_image_download_failure_is_not_a_server_connection_error():
    http_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", api_key="key"
    )
    image_url = "https://images.example.com/image.jpg"
    with aioresponses() as responses:
        responses.get(image_url, exception=aiohttp.ClientConnectionError("refused"))

        with pytest.raises(HTTPClientError) as error:
            await http_client.infer_async(inference_input=image_url, model_id="some/1")

        responses.assert_called_once()

    assert type(error.value) is HTTPClientError
    assert "Could not load image from URL" in str(error.value)
    assert "refused" in str(error.value)


def test_get_server_info_forwards_request_timeout(requests_mock):
    http_client = InferenceHTTPClient(api_url="http://localhost:9001")
    requests_mock.get(
        "http://localhost:9001/info",
        json={"name": "inference", "version": "1.0.0", "uuid": "server-uuid"},
    )

    http_client.get_server_info()
    assert requests_mock.last_request.timeout == 300.0

    with http_client.use_configuration(InferenceConfiguration(request_timeout=12.5)):
        http_client.get_server_info()
    assert requests_mock.last_request.timeout == 12.5

    with http_client.use_configuration(InferenceConfiguration(request_timeout=None)):
        http_client.get_server_info()
    assert requests_mock.last_request.timeout is None


def test_get_server_info_read_timeout_raises_client_error(requests_mock):
    http_client = InferenceHTTPClient(api_url="http://localhost:9001")
    original = requests.exceptions.ReadTimeout("Read timed out")
    requests_mock.get("http://localhost:9001/info", exc=original)

    with pytest.raises(HTTPClientError) as error:
        http_client.get_server_info()

    assert type(error.value) is HTTPClientError
    assert str(error.value).startswith("Server request timed out")
    assert error.value.__cause__ is original


def test_get_server_info_connect_timeout_stays_connection_error(requests_mock):
    http_client = InferenceHTTPClient(api_url="http://localhost:9001")
    original = requests.exceptions.ConnectTimeout("Connect timed out")
    requests_mock.get("http://localhost:9001/info", exc=original)

    with pytest.raises(HTTPServerConnectionError) as error:
        http_client.get_server_info()

    assert type(error.value) is HTTPServerConnectionError
    assert error.value.__cause__ is original


def test_run_workflow_and_infer_forward_request_timeout(requests_mock):
    workflow_client = InferenceHTTPClient(api_url="http://localhost:9001")
    requests_mock.post(
        "http://localhost:9001/workspace/workflows/workflow",
        json={"outputs": [{"ok": True}]},
    )

    with workflow_client.use_configuration(
        InferenceConfiguration(request_timeout=12.5)
    ):
        workflow_result = workflow_client.run_workflow(
            workspace_name="workspace", workflow_id="workflow"
        )

    assert workflow_result == [{"ok": True}]
    assert requests_mock.last_request.timeout == 12.5

    infer_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", api_key="key"
    )
    requests_mock.post(
        "https://serverless.roboflow.com/some/1",
        json={"predictions": []},
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with infer_client.use_configuration(InferenceConfiguration(request_timeout=12.5)):
        infer_result = infer_client.infer(inference_input=image, model_id="some/1")

    assert infer_result == {"predictions": []}
    assert requests_mock.last_request.timeout == 12.5


def test_infer_read_timeout_does_not_retry(requests_mock):
    http_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", api_key="key"
    )
    original = requests.exceptions.ReadTimeout("Read timed out")
    requests_mock.post("https://serverless.roboflow.com/some/1", exc=original)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(HTTPClientError) as error:
        http_client.infer(inference_input=image, model_id="some/1")

    assert type(error.value) is HTTPClientError
    assert str(error.value).startswith("Server request timed out")
    assert error.value.__cause__ is original
    assert requests_mock.call_count == 1


def test_infer_connect_timeout_retries_and_raises_connection_error(
    requests_mock, monkeypatch
):
    monkeypatch.setattr("backoff._sync.time.sleep", lambda _: None)
    http_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", api_key="key"
    )
    original = requests.exceptions.ConnectTimeout("Connect timed out")
    requests_mock.post("https://serverless.roboflow.com/some/1", exc=original)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(HTTPServerConnectionError) as error:
        http_client.infer(inference_input=image, model_id="some/1")

    assert type(error.value) is HTTPServerConnectionError
    assert error.value.__cause__ is original
    assert requests_mock.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configuration, expected_timeout",
    [
        (None, aiohttp.ClientTimeout(total=300)),
        (
            InferenceConfiguration(request_timeout=12.5),
            aiohttp.ClientTimeout(total=12.5),
        ),
        (
            InferenceConfiguration(request_timeout=None),
            aiohttp.ClientTimeout(total=None),
        ),
    ],
)
async def test_async_client_session_uses_configured_timeout(
    configuration, expected_timeout
):
    captured = {}
    real_session = aiohttp.ClientSession

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return real_session(*args, **kwargs)

    http_client = InferenceHTTPClient(api_url="http://localhost:9001", api_key="key")
    if configuration is not None:
        http_client.configure(configuration)

    with aioresponses() as responses:
        responses.get(
            "http://localhost:9001/model/registry?api_key=key",
            payload={"models": []},
        )
        with mock.patch.object(aiohttp, "ClientSession", side_effect=_capture):
            await http_client.list_loaded_models_async()

    assert captured["timeout"] == expected_timeout
