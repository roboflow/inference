from typing import Type
from unittest import mock
from unittest.mock import MagicMock

import pytest
import requests.exceptions
from requests_mock import Mocker

from inference.core import roboflow_api
from inference.core.env import API_BASE_URL
from inference.core.exceptions import (
    MalformedRoboflowAPIResponseError,
    MissingDefaultModelError,
    RoboflowAPIConnectionError,
    RoboflowAPIIAlreadyAnnotatedError,
    RoboflowAPIIAnnotationRejectionError,
    RoboflowAPIImageUploadRejectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIUnsuccessfulRequestError,
    WorkspaceLoadError,
)
from inference.core.roboflow_api import (
    ModelEndpointType,
    annotate_image_at_roboflow,
    get_roboflow_dataset_type,
    get_roboflow_labeling_batches,
    get_roboflow_labeling_jobs,
    get_roboflow_model_data,
    get_roboflow_model_type,
    get_roboflow_workspace,
    raise_from_lambda,
    register_image_at_roboflow,
    wrap_roboflow_api_errors,
    get_roboflow_active_learning_configuration,
)
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


@mock.patch.object(roboflow_api.requests, "get")
def test_get_roboflow_workspace_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        _ = get_roboflow_workspace(api_key="my_api_key")


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
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&nocache=true&device=some&dynamic=true"
    )


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
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&nocache=true&device=some&dynamic=true"
    )


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
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&nocache=true&device=some&dynamic=true"
    )


def test_get_roboflow_model_data_when_response_parsing_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/ort/coins_detection/1"),
        content=b"For sure not a JSON payload",
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
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&nocache=true&device=some&dynamic=true"
    )


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
    assert (
        requests_mock.last_request.query
        == "api_key=my_api_key&nocache=true&device=some&dynamic=true"
    )
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
