import json
from typing import Type
from unittest import mock
from unittest.mock import MagicMock

import pytest
import requests.exceptions
from aioresponses import aioresponses
from requests_mock import Mocker
from yarl import URL

from inference.core import roboflow_api
from inference.core.cache import MemoryCache
from inference.core.env import API_BASE_URL
from inference.core.exceptions import (
    MalformedRoboflowAPIResponseError,
    MalformedWorkflowResponseError,
    MissingDefaultModelError,
    RetryRequestError,
    RoboflowAPIConnectionError,
    RoboflowAPIIAlreadyAnnotatedError,
    RoboflowAPIIAnnotationRejectionError,
    RoboflowAPIImageUploadRejectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
    WorkspaceLoadError,
)
from inference.core.roboflow_api import (
    ModelEndpointType,
    annotate_image_at_roboflow,
    build_roboflow_api_headers,
    delete_cached_workflow_response_if_exists,
    get_from_url,
    get_roboflow_active_learning_configuration,
    get_roboflow_dataset_type,
    get_roboflow_labeling_batches,
    get_roboflow_labeling_jobs,
    get_roboflow_model_data,
    get_roboflow_model_type,
    get_roboflow_workspace,
    get_roboflow_workspace_async,
    get_workflow_specification,
    raise_from_lambda,
    register_image_at_roboflow,
    wrap_roboflow_api_errors,
)
from inference.core.version import __version__
from inference.core.utils.url_utils import wrap_url


class TestException(Exception):
    pass


def test_wrap_roboflow_api_errors_when_no_error_occurs() -> None:
    # given

    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        return a + b

    # when
    result = my_fun(2, 3)

    # then
    assert result == 5


def test_wrap_roboflow_api_errors_when_given_up_on_timeout_error() -> None:
    # given

    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        raise RetryRequestError("some", inner_error=requests.Timeout())

    # when
    with pytest.raises(RoboflowAPITimeoutError):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_given_up_on_http_error() -> None:
    # given

    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        response = requests.Response()
        response.status_code = 404
        raise RetryRequestError(
            "some", inner_error=requests.exceptions.HTTPError(response=response)
        )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = my_fun(2, 3)


@pytest.mark.parametrize(
    "exception_class", [ConnectionError, requests.exceptions.ConnectionError]
)
def test_wrap_roboflow_api_errors_when_connection_error_occurs(
    exception_class: Type[Exception],
) -> None:
    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        raise exception_class("some")

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_timeout_error_is_raised() -> None:
    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        raise requests.Timeout("some")

    # when
    with pytest.raises(RoboflowAPITimeoutError):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_http_401_error_occurs_and_default_handlers_used() -> (
    None
):
    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        response = requests.Response()
        response.status_code = 401
        raise requests.exceptions.HTTPError("some", response=response)

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_http_404_error_occurs_and_default_handlers_used() -> (
    None
):
    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError("some", response=response)

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_http_error_occurs_and_custom_handlers_used() -> (
    None
):
    @wrap_roboflow_api_errors(
        http_errors_handlers={
            500: lambda e: raise_from_lambda(e, TestException, "Some")
        }
    )
    def my_fun(a: int, b: int) -> int:
        response = requests.Response()
        response.status_code = 500
        raise requests.exceptions.HTTPError("some", response=response)

    # when
    with pytest.raises(TestException):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_http_error_occurs_and_custom_handlers_override_defaults() -> (
    None
):
    @wrap_roboflow_api_errors(
        http_errors_handlers={
            401: lambda e: raise_from_lambda(e, TestException, "Some")
        }
    )
    def my_fun(a: int, b: int) -> int:
        response = requests.Response()
        response.status_code = 401
        raise requests.exceptions.HTTPError("some", response=response)

    # when
    with pytest.raises(TestException):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_http_error_occurs_and_default_handlers_used() -> (
    None
):
    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        response = requests.Response()
        response.status_code = 401
        raise requests.exceptions.HTTPError("some", response=response)

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_connection_json_parsing_error_occurs() -> None:
    @wrap_roboflow_api_errors()
    def my_fun(a: int, b: int) -> int:
        raise requests.exceptions.InvalidJSONError("some")

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = my_fun(2, 3)


def test_get_roboflow_workspace_when_wrong_api_key_used(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_roboflow_workspace(api_key="my_api_key")

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


@pytest.mark.asyncio
async def test_get_roboflow_workspace_async_when_wrong_api_key_used() -> None:
    # given
    with aioresponses() as request_mock:
        request_mock.get(
            f"{API_BASE_URL}/?api_key=my_api_key&nocache=true",
            payload={
                "error": {
                    "message": "This API key does not exist (or has been revoked).",
                    "status": 401,
                    "type": "OAuthException",
                    "hint": "You may retrieve your API key via the Roboflow Dashboard.",
                    "key": "my_api_key",
                }
            },
            status=401,
        )

        # when
        with pytest.raises(RoboflowAPINotAuthorizedError):
            _ = await get_roboflow_workspace_async(api_key="my_api_key")


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_workspace_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_workspace(api_key="my_api_key")


@pytest.mark.asyncio
async def test_get_roboflow_workspace_async_when_connection_error_occurs() -> None:
    # given
    with aioresponses():
        # when
        with pytest.raises(RoboflowAPIConnectionError):
            _ = await get_roboflow_workspace_async(api_key="my_api_key")


@mock.patch.object(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
@pytest.mark.asyncio
async def test_get_roboflow_workspace_async_when_connection_error_occurs_and_retries_enforced() -> (
    None
):
    # given
    with aioresponses():
        # when
        with pytest.raises(RoboflowAPIConnectionError):
            _ = await get_roboflow_workspace_async(api_key="my_api_key")


@mock.patch.object(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", {503})
@pytest.mark.asyncio
async def test_get_roboflow_workspace_async_when_transient_errors_occur_and_finally_request_succeed() -> (
    None
):
    # given
    with aioresponses() as request_mock:
        request_mock.get(
            f"{API_BASE_URL}/?api_key=my_api_key&nocache=true",
            status=503,
        )
        request_mock.get(
            f"{API_BASE_URL}/?api_key=my_api_key&nocache=true",
            status=503,
        )
        request_mock.get(
            f"{API_BASE_URL}/?api_key=my_api_key&nocache=true",
            payload={"workspace": "my-workspace"},
        )

        # when
        result = await get_roboflow_workspace_async(api_key="my_api_key")

        # then
        assert result == "my-workspace"


def test_get_roboflow_workspace_when_response_parsing_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/"),
        content=b"For sure not a JSON payload",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_roboflow_workspace(api_key="my_api_key")

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


@pytest.mark.asyncio
async def test_get_roboflow_workspace_async_when_response_parsing_error_occurs() -> (
    None
):
    # given
    with aioresponses() as request_mock:
        request_mock.get(
            f"{API_BASE_URL}/?api_key=my_api_key&nocache=true",
            body=b"For sure not a JSON payload",
        )

        # when
        with pytest.raises(MalformedRoboflowAPIResponseError):
            _ = await get_roboflow_workspace_async(api_key="my_api_key")


def test_get_roboflow_workspace_when_workspace_id_is_empty(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/"),
        json={"some": "payload"},
    )

    # when
    with pytest.raises(WorkspaceLoadError):
        _ = get_roboflow_workspace(api_key="my_api_key")

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


@pytest.mark.asyncio
async def test_get_roboflow_workspace_async_when_workspace_id_is_empty() -> None:
    # given
    with aioresponses() as request_mock:
        request_mock.get(
            f"{API_BASE_URL}/?api_key=my_api_key&nocache=true",
            payload={"some": "payload"},
        )

        # when
        with pytest.raises(WorkspaceLoadError):
            _ = await get_roboflow_workspace_async(api_key="my_api_key")


@mock.patch.object(
    roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", json.dumps({"extra": "header"})
)
def test_get_roboflow_workspace_when_response_is_valid(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/"),
        json={"workspace": "my_workspace"},
    )

    # when
    result = get_roboflow_workspace(api_key="my_api_key")

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"
    assert result == "my_workspace"


@mock.patch.object(
    roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", json.dumps({"extra": "header"})
)
@pytest.mark.asyncio
async def test_get_roboflow_workspace_async_when_response_is_valid() -> None:
    # given
    with aioresponses() as request_mock:
        request_mock.get(
            f"{API_BASE_URL}/?api_key=my_api_key&nocache=true",
            payload={"workspace": "my_workspace"},
        )

        # when
        result = await get_roboflow_workspace_async(api_key="my_api_key")

        # then
        assert result == "my_workspace"
        registered_requests = request_mock.requests[
            ("GET", URL(f"{API_BASE_URL}/?api_key=my_api_key&nocache=true"))
        ]
        assert registered_requests[0].kwargs["headers"] == {
            "extra": "header",
            roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
            roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER: "true",
        }


def test_get_roboflow_dataset_type_when_wrong_key_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_roboflow_dataset_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


def test_get_roboflow_dataset_type_when_project_not_found_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_dataset_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_dataset_type_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_dataset_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )


def test_get_roboflow_dataset_type_when_response_parsing_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection"),
        content=b"For sure not a JSON payload",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_roboflow_dataset_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


def test_get_roboflow_dataset_type_when_project_is_empty(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection"),
        json={"project": {}},
    )

    # when
    result = get_roboflow_dataset_type(
        api_key="my_api_key", workspace_id="my_workspace", dataset_id="coins_detection"
    )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"
    assert result == "object-detection"


def test_get_roboflow_dataset_type_when_response_is_valid(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection"),
        json={"project": {"type": "classification"}},
    )

    # when
    result = get_roboflow_dataset_type(
        api_key="my_api_key", workspace_id="my_workspace", dataset_id="coins_detection"
    )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"
    assert result == "classification"


def test_get_roboflow_model_type_when_wrong_api_key_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/1"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_roboflow_model_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
            version_id="1",
            project_task_type="object-detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


def test_get_roboflow_model_type_when_wrong_project_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/1"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_model_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
            version_id="1",
            project_task_type="object-detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


def test_get_roboflow_model_type_when_wrong_version_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/invalid"),
        status_code=500,  # This is to compensate bug in Roboflow API
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_model_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
            version_id="invalid",
            project_task_type="object-detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_model_type_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_model_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
            version_id="1",
            project_task_type="object-detection",
        )


def test_get_roboflow_model_type_when_response_parsing_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/1"),
        content=b"For sure not a JSON payload",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_roboflow_model_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
            version_id="1",
            project_task_type="object-detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


def test_get_roboflow_model_type_when_default_model_can_be_chosen(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/1"),
        json={"version": {}},
    )

    # when
    result = get_roboflow_model_type(
        api_key="my_api_key",
        workspace_id="my_workspace",
        dataset_id="coins_detection",
        version_id="1",
        project_task_type="object-detection",
    )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"
    assert result == "yolov5v2s"


def test_get_roboflow_model_type_when_default_model_cannot_be_chosen(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/1"),
        json={"version": {}},
    )

    # when
    with pytest.raises(MissingDefaultModelError):
        _ = get_roboflow_model_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
            version_id="1",
            project_task_type="unknown",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


def test_get_roboflow_model_type_when_response_is_valid(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/1"),
        json={"version": {"modelType": "yolov8n"}},
    )

    # when
    result = get_roboflow_model_type(
        api_key="my_api_key",
        workspace_id="my_workspace",
        dataset_id="coins_detection",
        version_id="1",
        project_task_type="object-detection",
    )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"
    assert result == "yolov8n"


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_model_data_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_model_data(
            api_key="my_api_key",
            model_id="coins_detection/1",
            endpoint_type=ModelEndpointType.ORT,
            device_id="some",
        )


def test_get_roboflow_model_data_when_wrong_api_key_used(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/ort/coins_detection/1"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_roboflow_model_data(
            api_key="my_api_key",
            model_id="coins_detection/1",
            endpoint_type=ModelEndpointType.ORT,
            device_id="some",
        )

    # then
    params = ["api_key=my_api_key", "nocache=true", "device=some", "dynamic=true"]
    for param in params:
        assert param in requests_mock.last_request.query


def test_get_roboflow_model_data_when_wrong_model_used(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/ort/coins_detection/1"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_model_data(
            api_key="my_api_key",
            model_id="coins_detection/1",
            endpoint_type=ModelEndpointType.ORT,
            device_id="some",
        )

    # then
    params = ["api_key=my_api_key", "nocache=true", "device=some", "dynamic=true"]
    for param in params:
        assert param in requests_mock.last_request.query


def test_get_roboflow_model_data_when_http_error_occurs(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/ort/coins_detection/1"),
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_roboflow_model_data(
            api_key="my_api_key",
            model_id="coins_detection/1",
            endpoint_type=ModelEndpointType.ORT,
            device_id="some",
        )

    # then
    params = ["api_key=my_api_key", "nocache=true", "device=some", "dynamic=true"]
    for param in params:
        assert param in requests_mock.last_request.query


def test_get_roboflow_model_data_when_response_parsing_error_occurs(
    requests_mock: Mocker,
) -> None:
    expected_response = b"For sure not a JSON payload"
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/ort/coins_detection/1"),
        content=expected_response,
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_roboflow_model_data(
            api_key="my_api_key",
            model_id="coins_detection/1",
            endpoint_type=ModelEndpointType.ORT,
            device_id="some",
        )

    # then
    params = ["api_key=my_api_key", "nocache=true", "device=some", "dynamic=true"]
    for param in params:
        assert param in requests_mock.last_request.query


def test_get_roboflow_model_data_when_valid_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    expected_response = {
        "ort": {
            "name": "barbel-detection",
            "type": "object-detection",
            "icon": "XXX",
            "iconHasAnnotation": False,
            "annotation": "person-barbell-edge",
            "environment": "some-link",
            "classes": ["barbell-edge", "person", "plate"],
            "colors": {"coin": "#C7FC00"},
            "model": "some-link",
            "modelType": "yolov8s",
        }
    }
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/ort/coins_detection/1"),
        json=expected_response,
    )

    # when
    result = get_roboflow_model_data(
        api_key="my_api_key",
        model_id="coins_detection/1",
        endpoint_type=ModelEndpointType.ORT,
        device_id="some",
    )

    # then
    params = ["api_key=my_api_key", "nocache=true", "device=some", "dynamic=true"]
    for param in params:
        assert param in requests_mock.last_request.query

    assert result == expected_response


@mock.patch.object(roboflow_api.requests, "post")
def test_register_image_at_roboflow_when_connection_error_occurs(
    post_mock: MagicMock,
) -> None:
    # given
    post_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = register_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            image_bytes=b"SOME_IMAGE_BYTES",
            batch_name="my-batch",
            tags=["a", "b", "c/d"],
        )


def test_register_image_at_roboflow_when_wrong_api_key_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = register_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            image_bytes=b"SOME_IMAGE_BYTES",
            batch_name="my-batch",
            tags=["a", "b", "c/d"],
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&batch=my-batch&tag=a&tag=b&tag=c%2fd"
    )
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )


def test_register_image_at_roboflow_when_wrong_dataset_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/invalid/upload"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = register_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="invalid",
            local_image_id="local_id",
            image_bytes=b"SOME_IMAGE_BYTES",
            batch_name="my-batch",
            tags=["a", "b", "c/d"],
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&batch=my-batch&tag=a&tag=b&tag=c%2fd"
    )
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )


def test_register_image_at_roboflow_when_http_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = register_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            image_bytes=b"SOME_IMAGE_BYTES",
            batch_name="my-batch",
            tags=["a", "b", "c/d"],
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&batch=my-batch&tag=a&tag=b&tag=c%2fd"
    )
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )


def test_register_image_at_roboflow_when_response_parsing_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        content=b"For sure not a JSON payload",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = register_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            image_bytes=b"SOME_IMAGE_BYTES",
            batch_name="my-batch",
            tags=["a", "b", "c/d"],
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&batch=my-batch&tag=a&tag=b&tag=c%2fd"
    )
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )


def test_register_image_at_roboflow_when_valid_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        json={"success": True, "id": "xxx"},
    )

    # when
    response = register_image_at_roboflow(
        api_key="my_api_key",
        dataset_id="coins_detection",
        local_image_id="local_id",
        image_bytes=b"SOME_IMAGE_BYTES",
        batch_name="my-batch",
        tags=["a", "b", "c/d"],
    )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&batch=my-batch&tag=a&tag=b&tag=c%2fd"
    )
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )
    assert response == {"success": True, "id": "xxx"}


def test_register_image_at_roboflow_when_valid_response_returned_and_no_tags_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        json={"success": True, "id": "xxx"},
    )

    # when
    response = register_image_at_roboflow(
        api_key="my_api_key",
        dataset_id="coins_detection",
        local_image_id="local_id",
        image_bytes=b"SOME_IMAGE_BYTES",
        batch_name="my-batch",
    )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&batch=my-batch"
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )
    assert response == {"success": True, "id": "xxx"}


def test_register_image_at_roboflow_when_duplicate_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        json={"duplicate": True, "id": "xxx"},
    )

    # when
    response = register_image_at_roboflow(
        api_key="my_api_key",
        dataset_id="coins_detection",
        local_image_id="local_id",
        image_bytes=b"SOME_IMAGE_BYTES",
        batch_name="my-batch",
    )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&batch=my-batch"
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )
    assert response == {"duplicate": True, "id": "xxx"}


def test_register_image_at_roboflow_when_error_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        json={"error": "some"},
    )

    # when
    with pytest.raises(RoboflowAPIImageUploadRejectionError) as e:
        _ = register_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            image_bytes=b"SOME_IMAGE_BYTES",
            batch_name="my-batch",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&batch=my-batch"
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )
    assert str({"error": "some"}) in str(e.value)


def test_register_image_at_roboflow_when_lack_of_success_reported(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/upload"),
        json={"success": False},
    )

    # when
    with pytest.raises(RoboflowAPIImageUploadRejectionError) as e:
        _ = register_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            image_bytes=b"SOME_IMAGE_BYTES",
            batch_name="my-batch",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&batch=my-batch"
    assert requests_mock.last_request.text.fields["name"] == "local_id.jpg"
    assert requests_mock.last_request.text.fields["file"] == (
        "imageToUpload",
        b"SOME_IMAGE_BYTES",
        "image/jpeg",
    )
    assert str({"success": False}) in str(e.value)


@mock.patch.object(roboflow_api.requests, "post")
def test_annotate_image_at_roboflow_when_connection_error_occurs(
    post_mock: MagicMock,
) -> None:
    # given
    post_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )


def test_annotate_image_at_roboflow_when_wrong_api_key_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/roboflow_id"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )


def test_annotate_image_at_roboflow_when_wrong_dataset_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/invalid/annotate/roboflow_id"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="invalid",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )


def test_annotate_image_at_roboflow_when_wrong_image_id_selected_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/invalid"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="invalid",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=False,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=false"
    )


def test_annotate_image_at_roboflow_when_image_already_annotated(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/roboflow_id"),
        status_code=409,
    )

    # when
    with pytest.raises(RoboflowAPIIAlreadyAnnotatedError):
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )


def test_annotate_image_at_roboflow_when_http_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/roboflow_id"),
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )


def test_annotate_image_at_roboflow_when_response_parsing_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/roboflow_id"),
        content=b"For sure not a JSON payload",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )


def test_annotate_image_at_roboflow_when_error_description_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/roboflow_id"),
        json={"error": "some"},
    )

    # when
    with pytest.raises(RoboflowAPIIAnnotationRejectionError) as error:
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )
    assert str({"error": "some"}) in str(error.value)


def test_annotate_image_at_roboflow_when_lack_of_success_reported(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/roboflow_id"),
        json={"success": False},
    )

    # when
    with pytest.raises(RoboflowAPIIAnnotationRejectionError) as error:
        _ = annotate_image_at_roboflow(
            api_key="my_api_key",
            dataset_id="coins_detection",
            local_image_id="local_id",
            roboflow_image_id="roboflow_id",
            annotation_content="some",
            annotation_file_type="txt",
            is_prediction=True,
        )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )
    assert str({"success": False}) in str(error.value)


def test_annotate_image_at_roboflow_when_successful_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.post(
        url=wrap_url(f"{API_BASE_URL}/dataset/coins_detection/annotate/roboflow_id"),
        json={"success": True},
    )

    # when
    result = annotate_image_at_roboflow(
        api_key="my_api_key",
        dataset_id="coins_detection",
        local_image_id="local_id",
        roboflow_image_id="roboflow_id",
        annotation_content="some",
        annotation_file_type="txt",
        is_prediction=True,
    )

    # then
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&name=local_id.txt&prediction=true"
    )
    assert result == {"success": True}


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_labeling_batches_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_labeling_batches(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )


def test_get_roboflow_labeling_batches_when_wrong_api_key_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/batches"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_roboflow_labeling_batches(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_batches_when_wrong_dataset_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/invalid/batches"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_labeling_batches(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="invalid",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_batches_when_wrong_workspace_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/invalid/coins_detection/batches"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_labeling_batches(
            api_key="my_api_key",
            workspace_id="invalid",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_batches_when_http_error_occurred(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/invalid/coins_detection/batches"),
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_roboflow_labeling_batches(
            api_key="my_api_key",
            workspace_id="invalid",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_batches_when_malformed_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/invalid/coins_detection/batches"),
        content=b"Not a JSON :)",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_roboflow_labeling_batches(
            api_key="my_api_key",
            workspace_id="invalid",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_batches_when_valid_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    expected_result = {
        "batches": [
            {
                "name": "Pip Package Upload",
                "numJobs": 0,
                "uploaded": {"_seconds": 1698060510, "_nanoseconds": 403000000},
                "images": 1,
                "id": "XXX",
            },
            {
                "name": "active-learning-23-10-2023",
                "uploaded": {"_seconds": 1698060589, "_nanoseconds": 416000000},
                "numJobs": 1,
                "images": 2,
                "id": "XXX",
            },
        ]
    }
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/batches"),
        json=expected_result,
    )

    # when
    result = get_roboflow_labeling_batches(
        api_key="my_api_key",
        workspace_id="my_workspace",
        dataset_id="coins_detection",
    )

    # then
    assert result == expected_result
    assert requests_mock.last_request.query == "api_key=my_api_key"


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_labeling_jobs_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_labeling_jobs(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )


def test_get_roboflow_labeling_jobs_when_wrong_api_key_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/jobs"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_roboflow_labeling_jobs(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_jobs_when_wrong_dataset_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/invalid/jobs"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_labeling_jobs(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="invalid",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_jobs_when_wrong_workspace_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/invalid/coins_detection/jobs"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_labeling_jobs(
            api_key="my_api_key",
            workspace_id="invalid",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_jobs_when_http_error_occurred(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/invalid/coins_detection/jobs"),
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_roboflow_labeling_jobs(
            api_key="my_api_key",
            workspace_id="invalid",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_jobs_when_malformed_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/invalid/coins_detection/jobs"),
        content=b"Not a JSON :)",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_roboflow_labeling_jobs(
            api_key="my_api_key",
            workspace_id="invalid",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_labeling_jobs_when_valid_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    expected_result = {
        "jobs": [
            {
                "owner": "XXX",
                "rejected": 0,
                "project": "XXX",
                "reviewer": "XXX",
                "sourceBatch": "XXX/YYY",
                "approved": 0,
                "instructionsText": "",
                "createdBy": "XXX",
                "name": "Uploaded on 10/11/23 at 7:20 pm",
                "labeler": "XXX",
                "created": {"_seconds": 1697044995, "_nanoseconds": 375000000},
                "numImages": 161,
                "status": "complete",
                "unannotated": 0,
                "annotated": 161,
                "id": "XXX",
            }
        ],
    }
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/invalid/coins_detection/jobs"),
        json=expected_result,
    )

    # when
    result = get_roboflow_labeling_jobs(
        api_key="my_api_key",
        workspace_id="invalid",
        dataset_id="coins_detection",
    )

    # then
    assert result == expected_result
    assert requests_mock.last_request.query == "api_key=my_api_key"


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_active_learning_configuration_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_active_learning_configuration(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )


def test_get_roboflow_active_learning_configuration_when_wrong_api_key_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/active_learning"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_roboflow_active_learning_configuration(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_roboflow_active_learning_configuration_when_not_found_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/active_learning"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_roboflow_active_learning_configuration(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_roboflow_active_learning_configuration_when_internal_error_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/active_learning"),
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_roboflow_active_learning_configuration(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_roboflow_active_learning_configuration_when_malformed_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/coins_detection/active_learning"),
        content=b"Not a JSON :)",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_roboflow_active_learning_configuration(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


@mock.patch.object(roboflow_api.requests, "get")
def test_get_workflow_specification_when_connection_error_occurs_and_no_cache_to_be_used(
    get_mock: MagicMock,
) -> None:
    # given
    delete_cached_workflow_response_if_exists(
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        api_key="my_api_key",
    )
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            use_cache=False,
        )


@mock.patch.object(roboflow_api.requests, "get")
def test_get_workflow_specification_when_connection_error_occurs_but_file_is_cached_in_file(
    get_mock: MagicMock,
) -> None:
    # given
    delete_cached_workflow_response_if_exists(
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        api_key="my_api_key",
    )
    get_mock.return_value = MagicMock(
        status_code=200,
        json=MagicMock(
            return_value={
                "workflow": {"config": json.dumps({"specification": {"some": "some"}})}
            }
        ),
    )
    _ = get_workflow_specification(
        api_key="my_api_key",
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        ephemeral_cache=MemoryCache(),
    )
    get_mock.side_effect = ConnectionError()

    # when
    result = get_workflow_specification(
        api_key="my_api_key",
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        ephemeral_cache=MemoryCache(),
    )

    # then
    assert result == {
        "some": "some",
        "id": None,
    }, "Expected workflow specification to be retrieved from file"


@mock.patch.object(roboflow_api.requests, "get")
def test_get_workflow_specification_when_consecutive_request_hits_ephemeral_cache(
    get_mock: MagicMock,
) -> None:
    # given
    delete_cached_workflow_response_if_exists(
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        api_key="my_api_key",
    )
    get_mock.return_value = MagicMock(
        status_code=200,
        json=MagicMock(
            return_value={
                "workflow": {"config": json.dumps({"specification": {"some": "some"}})}
            }
        ),
    )
    ephemeral_cache = MemoryCache()
    _ = get_workflow_specification(
        api_key="my_api_key",
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        ephemeral_cache=ephemeral_cache,
    )

    # when
    result = get_workflow_specification(
        api_key="my_api_key",
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        ephemeral_cache=ephemeral_cache,
    )

    # then
    assert result == {
        "some": "some",
        "id": None,
    }, "Expected workflow specification to be retrieved from file"
    assert get_mock.call_count == 1, "Expected remote API to be only called once"


def test_get_workflow_specification_when_wrong_api_key_used_and_no_cache_allowed_to_be_used(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            use_cache=False,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_wrong_api_key_used_and_ephemeral_cache_miss_detected(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        status_code=401,
    )

    # when
    with pytest.raises(RoboflowAPINotAuthorizedError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            ephemeral_cache=MemoryCache(),
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_not_found_returned_and_cache_disabled(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            use_cache=False,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_not_found_returned_and_ephemeral_cache_miss_detected(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        status_code=404,
    )

    # when
    with pytest.raises(RoboflowAPINotNotFoundError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            ephemeral_cache=MemoryCache(),
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_internal_error_returned_and_cache_disabled(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            use_cache=False,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_internal_error_returned_and_ephemeral_cache_miss_detected(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        status_code=500,
    )
    ephemeral_cache = MemoryCache()

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            ephemeral_cache=ephemeral_cache,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"
    assert len(ephemeral_cache.cache) == 0, "Expected nothing saved to cache"


def test_get_workflow_specification_when_malformed_response_returned_and_cache_disabled(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        content=b"Not a JSON :)",
    )

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            use_cache=False,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_malformed_response_returned_and_ephemeral_cache_miss_detected(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        content=b"Not a JSON :)",
    )
    ephemeral_cache = MemoryCache()

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            ephemeral_cache=ephemeral_cache,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"
    assert len(ephemeral_cache.cache) == 0, "Expected nothing saved to cache"


def test_get_workflow_specification_when_config_not_provided_and_cache_disabled(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        json={},
    )

    # when
    with pytest.raises(MalformedWorkflowResponseError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            use_cache=False,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_config_not_provided_and_ephemeral_cache_miss_detected(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        json={},
    )
    ephemeral_cache = MemoryCache()

    # when
    with pytest.raises(MalformedWorkflowResponseError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            ephemeral_cache=ephemeral_cache,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"
    assert len(ephemeral_cache.cache) == 0, "Expected nothing saved to cache"


def test_get_workflow_specification_when_config_not_parsable_and_cache_disabled(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        json={"config": "Not a JSON"},
    )

    # when
    with pytest.raises(MalformedWorkflowResponseError):
        _ = get_workflow_specification(
            api_key="my_api_key",
            workspace_id="my_workspace",
            workflow_id="some_workflow",
            use_cache=False,
        )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"


def test_get_workflow_specification_when_valid_response_given_and_cache_disabled(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        json={
            "workflow": {
                "owner": "50hbxrck9m8nKykOhCEq",
                "name": "Thermal",
                "url": "thermal",
                "config": '{"specification":{"version":"1.0","inputs":[{"type":"InferenceImage","name":"image"}],"steps":[{"type":"CVModel","name":"step_1","image":"$inputs.image","model_id":"thermal dogs and people/18"}],"outputs":[{"type":"JsonField","name":"a","selector":"$steps.step_1.predictions"}]},"preset":"single-model"}',
                "id": "Har3FW34j1Rjc4p8IX4B",
            },
            "status": "ok",
        },
    )

    # when
    result = get_workflow_specification(
        api_key="my_api_key",
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        use_cache=False,
    )

    # then
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"
    assert result == {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "CVModel",
                "name": "step_1",
                "image": "$inputs.image",
                "model_id": "thermal dogs and people/18",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "a",
                "selector": "$steps.step_1.predictions",
            }
        ],
        "id": "Har3FW34j1Rjc4p8IX4B",
    }


def test_get_workflow_specification_when_valid_response_given_on_consecutive_requests(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/my_workspace/workflows/some_workflow"),
        json={
            "workflow": {
                "owner": "50hbxrck9m8nKykOhCEq",
                "name": "Thermal",
                "url": "thermal",
                "config": '{"specification":{"version":"1.0","inputs":[{"type":"InferenceImage","name":"image"}],"steps":[{"type":"CVModel","name":"step_1","image":"$inputs.image","model_id":"thermal dogs and people/18"}],"outputs":[{"type":"JsonField","name":"a","selector":"$steps.step_1.predictions"}]},"preset":"single-model"}',
                "id": "Har3FW34j1Rjc4p8IX4B",
            },
            "status": "ok",
        },
    )
    ephemeral_cache = MemoryCache()

    # when
    result_1 = get_workflow_specification(
        api_key="my_api_key",
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        ephemeral_cache=ephemeral_cache,
    )
    result_2 = get_workflow_specification(
        api_key="my_api_key",
        workspace_id="my_workspace",
        workflow_id="some_workflow",
        ephemeral_cache=ephemeral_cache,
    )

    # then
    assert requests_mock.call_count == 1, "Expected remote API to be called only once"
    assert (
        requests_mock.last_request.query == "api_key=my_api_key"
    ), "API key must be given in query"
    assert (
        result_1
        == result_2
        == {
            "version": "1.0",
            "inputs": [{"type": "InferenceImage", "name": "image"}],
            "steps": [
                {
                    "type": "CVModel",
                    "name": "step_1",
                    "image": "$inputs.image",
                    "model_id": "thermal dogs and people/18",
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "a",
                    "selector": "$steps.step_1.predictions",
                }
            ],
            "id": "Har3FW34j1Rjc4p8IX4B",
        }
    )
    assert len(ephemeral_cache.cache) == 1, "Expected cache content to appear"


@mock.patch.object(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", None)
def test_build_roboflow_api_headers_when_no_extra_headers() -> None:
    # when
    result = build_roboflow_api_headers()

    # then
    assert result == {
        roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
        roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }


@mock.patch.object(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", None)
def test_build_roboflow_api_headers_when_no_extra_headers_but_explicit_headers_given() -> (
    None
):
    # when
    result = build_roboflow_api_headers(explicit_headers={"my": "header"})

    # then
    assert result == {
        "my": "header",
        roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
        roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }, "Expected to preserve explicit header and inject version"


@mock.patch.object(
    roboflow_api,
    "ROBOFLOW_API_EXTRA_HEADERS",
    json.dumps({"extra": "header", "another": "extra"}),
)
def test_build_roboflow_api_headers_when_extra_headers_given() -> None:
    # when
    result = build_roboflow_api_headers()

    # then
    assert result == {
        "extra": "header",
        "another": "extra",
        roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
        roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }, "Expected extra headers to be decoded"


@mock.patch.object(
    roboflow_api,
    "ROBOFLOW_API_EXTRA_HEADERS",
    json.dumps({"extra": "header", "another": "extra"}),
)
def test_build_roboflow_api_headers_when_extra_headers_given_and_explicit_headers_present() -> (
    None
):
    # when
    result = build_roboflow_api_headers(explicit_headers={"my": "header"})

    # then
    assert result == {
        "my": "header",
        "extra": "header",
        "another": "extra",
        roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
        roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }, "Expected extra headers to be decoded and shipped along with explicit headers"


@mock.patch.object(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", "For sure not a JSON :)")
def test_build_roboflow_api_headers_when_extra_headers_given_as_invalid_json() -> None:
    # when
    result = build_roboflow_api_headers(explicit_headers={"my": "header"})

    # then
    assert result == {
        "my": "header",
        roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
        roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }, "Expected extra headers to be decoded and shipped along with explicit headers"


@mock.patch.object(
    roboflow_api,
    "ROBOFLOW_API_EXTRA_HEADERS",
    json.dumps({"extra": "header", "another": "extra"}),
)
def test_build_roboflow_api_headers_when_extra_headers_given_and_explicit_headers_collide_with_extras() -> (
    None
):
    # when
    result = build_roboflow_api_headers(
        explicit_headers={
            "extra": "explicit-is-better",
            "my": "header",
        }
    )

    # then
    assert result == {
        "another": "extra",
        "extra": "explicit-is-better",
        "my": "header",
        roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
        roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }, "Expected extra headers to be decoded and explicit header to override implicit one while keeping version header"


def test_build_roboflow_api_headers_always_sets_version_header() -> None:
    # when
    result = build_roboflow_api_headers(
        explicit_headers={
            roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER: "should-be-overwritten",
            "custom": "value",
        }
    )

    # then
    assert result[roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER] == __version__
    assert result[roboflow_api.ALLOW_CHUNKED_RESPONSE_HEADER] == "true"
    assert result["custom"] == "value"


@mock.patch.object(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", False)
@mock.patch.object(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", set())
def test_get_from_url_when_no_retires_possible(
    requests_mock: Mocker,
) -> None:
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/some"),
        json={
            "status": "fail",
        },
        status_code=500,
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_from_url(url=wrap_url(f"{API_BASE_URL}/some"), json_response=True)


@mock.patch.object(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
@mock.patch.object(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", {503})
def test_get_from_url_when_retires_possible(
    requests_mock: Mocker,
) -> None:
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/some"),
        response_list=[
            {"json": {"status": "fail"}, "status_code": 503},
            {"json": {"status": "ok"}, "status_code": 200},
        ],
    )

    # when
    result = get_from_url(url=wrap_url(f"{API_BASE_URL}/some"), json_response=True)

    # then
    assert result == {"status": "ok"}


@mock.patch.object(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES", 1)
@mock.patch.object(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", True)
@mock.patch.object(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", {503})
def test_get_from_url_when_retires_possible_but_given_up(
    requests_mock: Mocker,
) -> None:
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/some"),
        response_list=[
            {"json": {"status": "fail"}, "status_code": 503},
            {"json": {"status": "fail"}, "status_code": 503},
        ],
    )

    # when
    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _ = get_from_url(url=wrap_url(f"{API_BASE_URL}/some"), json_response=True)


@mock.patch.object(roboflow_api, "MD5_VERIFICATION_ENABLED", True)
@mock.patch.object(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", False)
@mock.patch.object(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", set())
def test_get_from_url_when_md5_verification_enabled_but_x_goog_hash_header_missing(
    requests_mock: Mocker,
) -> None:
    request_url = wrap_url(f"{API_BASE_URL}/some")
    requests_mock.get(
        url=request_url,
        json={"status": "ok"},
        status_code=200,
        headers={},
    )

    with mock.patch.object(roboflow_api, "logger") as logger_mock:
        result = get_from_url(url=request_url, json_response=True)

    assert result == {"status": "ok"}
    logger_mock.warning.assert_called_once()
    call_args = logger_mock.warning.call_args[0][0]
    assert "x-goog-hash" in call_args
    assert request_url in call_args


@mock.patch.object(roboflow_api, "MD5_VERIFICATION_ENABLED", True)
@mock.patch.object(roboflow_api, "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", False)
@mock.patch.object(roboflow_api, "TRANSIENT_ROBOFLOW_API_ERRORS", set())
def test_get_from_url_when_md5_verification_enabled_but_x_goog_hash_missing_does_not_log_api_key(
    requests_mock: Mocker,
) -> None:
    secret_api_key = "my-secret-api-key-12345"
    request_url = wrap_url(f"{API_BASE_URL}/some?api_key={secret_api_key}")
    requests_mock.get(
        url=request_url,
        json={"status": "ok"},
        status_code=200,
        headers={},
    )

    with mock.patch.object(roboflow_api, "logger") as logger_mock:
        get_from_url(url=request_url, json_response=True)

    logged_message = logger_mock.warning.call_args[0][0]
    assert secret_api_key not in logged_message
    assert "x-goog-hash" in logged_message
