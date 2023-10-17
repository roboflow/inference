import json
import os.path
from typing import Any, Type, Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest
import requests.exceptions
from requests_mock import Mocker

from inference.core.entities.types import TaskType, ModelType
from inference.core.env import API_BASE_URL
from inference.core.exceptions import (
    InvalidModelIDError,
    MalformedRoboflowAPIResponseError,
    WorkspaceLoadError,
    DatasetLoadError,
    MissingDefaultModelError, ModelNotRecognisedError,
)
from inference.core.registries.roboflow import (
    get_model_id_chunks,
    get_model_metadata_from_cache,
    model_metadata_content_is_invalid,
    save_model_metadata_in_cache,
    wrap_roboflow_api_errors,
    raise_from_lambda,
    get_roboflow_workspace,
    get_roboflow_dataset_type,
    get_roboflow_model_type,
    get_model_type, RoboflowModelRegistry,
)
from inference.core.registries import roboflow
from inference.core.utils.url_utils import wrap_url


@pytest.mark.parametrize("value", ["some", "some/2/invalid", "another-2"])
def test_get_model_id_chunks_when_invalid_input_given(value: Any) -> None:
    # when
    with pytest.raises(InvalidModelIDError):
        _ = get_model_id_chunks(model_id=value)


def test_get_model_id_chunks_when_valid_input_given() -> None:
    # when
    result = get_model_id_chunks("some/1")

    # then
    assert result == ("some", "1")


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_does_not_exist(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    construct_model_type_cache_path_mock.return_value = os.path.join(
        empty_local_dir, "model_type.json"
    )

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_not_json(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("FOR SURE NOT JSON :)")

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_empty(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("")

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_is_invalid(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(json.dumps({"some": "key"}))

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_invalid(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "project_task_type": "object-detection",
                    "model_type": "yolov8n",
                }
            )
        )

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result == ("object-detection", "yolov8n")


def test_model_metadata_content_is_invalid_when_content_is_empty() -> None:
    # when
    result = model_metadata_content_is_invalid(content=None)

    # then
    assert result is True


def test_model_metadata_content_is_invalid_when_content_is_not_dict() -> None:
    # when
    result = model_metadata_content_is_invalid(content=[1, 2, 3])

    # then
    assert result is True


def test_model_metadata_content_is_invalid_when_model_type_is_missing() -> None:
    # when
    result = model_metadata_content_is_invalid(
        content={
            "project_task_type": "object-detection",
        }
    )

    # then
    assert result is True


def test_model_metadata_content_is_invalid_when_task_type_is_missing() -> None:
    # when
    result = model_metadata_content_is_invalid(
        content={
            "model_type": "yolov8n",
        }
    )

    # then
    assert result is True


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_save_model_metadata_in_cache(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path

    # when
    save_model_metadata_in_cache(
        dataset_id="some",
        version_id="1",
        project_task_type="instance-segmentation",
        model_type="yolov8l",
    )
    with open(metadata_path) as f:
        result = json.load(f)

    # then
    assert result["model_type"] == "yolov8l"
    assert result["project_task_type"] == "instance-segmentation"
    construct_model_type_cache_path_mock.assert_called_once_with(
        dataset_id="some", version_id="1"
    )


class TestException1(Exception):
    pass


class TestException2(Exception):
    pass


def test_wrap_roboflow_api_errors_when_no_error_occurs() -> None:
    # given

    @wrap_roboflow_api_errors(
        on_connection_error=lambda e: raise_from_lambda(e, TestException1, "some"),
        on_http_error=lambda e: raise_from_lambda(e, TestException2, "other"),
    )
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
    @wrap_roboflow_api_errors(
        on_connection_error=lambda e: raise_from_lambda(e, TestException1, "some"),
        on_http_error=lambda e: raise_from_lambda(e, TestException2, "other"),
    )
    def my_fun(a: int, b: int) -> int:
        raise exception_class("some")

    # when
    with pytest.raises(TestException1):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_connection_http_error_occurs() -> None:
    @wrap_roboflow_api_errors(
        on_connection_error=lambda e: raise_from_lambda(e, TestException1, "some"),
        on_http_error=lambda e: raise_from_lambda(e, TestException2, "other"),
    )
    def my_fun(a: int, b: int) -> int:
        raise requests.exceptions.HTTPError("some")

    # when
    with pytest.raises(TestException2):
        _ = my_fun(2, 3)


def test_wrap_roboflow_api_errors_when_connection_json_parsing_error_occurs() -> None:
    @wrap_roboflow_api_errors(
        on_connection_error=lambda e: raise_from_lambda(e, TestException1, "some"),
        on_http_error=lambda e: raise_from_lambda(e, TestException2, "other"),
    )
    def my_fun(a: int, b: int) -> int:
        raise requests.exceptions.InvalidJSONError("some")

    # when
    with pytest.raises(MalformedRoboflowAPIResponseError):
        _ = my_fun(2, 3)


def test_get_roboflow_workspace_when_http_error_occurs(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/"),
        status_code=403,
    )

    # when
    with pytest.raises(WorkspaceLoadError):
        _ = get_roboflow_workspace(api_key="my_api_key")

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"


@mock.patch.object(roboflow.requests, "get")
def test_get_roboflow_workspace_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(WorkspaceLoadError):
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
    assert requests_mock.last_request.query == "api_key=my_api_key"


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
    assert requests_mock.last_request.query == "api_key=my_api_key"


def test_get_roboflow_workspace_when_response_is_valid(requests_mock: Mocker) -> None:
    # given
    requests_mock.get(
        url=wrap_url(f"{API_BASE_URL}/"),
        json={"workspace": "my_workspace"},
    )

    # when
    result = get_roboflow_workspace(api_key="my_api_key")

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key"
    assert result == "my_workspace"


def test_get_roboflow_dataset_type_when_http_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/?api_key=my_api_key&nocache=true"
        ),
        status_code=403,
    )

    # when
    with pytest.raises(DatasetLoadError):
        _ = get_roboflow_dataset_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


@mock.patch.object(roboflow.requests, "get")
def test_get_roboflow_dataset_type_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(DatasetLoadError):
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
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/?api_key=my_api_key&nocache=true"
        ),
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
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/?api_key=my_api_key&nocache=true"
        ),
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
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/?api_key=my_api_key&nocache=true"
        ),
        json={"project": {"type": "classification"}},
    )

    # when
    result = get_roboflow_dataset_type(
        api_key="my_api_key", workspace_id="my_workspace", dataset_id="coins_detection"
    )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"
    assert result == "classification"


def test_get_roboflow_model_type_when_http_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/1/?api_key=my_api_key&nocache=true"
        ),
        status_code=403,
    )

    # when
    with pytest.raises(DatasetLoadError):
        _ = get_roboflow_model_type(
            api_key="my_api_key",
            workspace_id="my_workspace",
            dataset_id="coins_detection",
            version_id="1",
            project_task_type="object-detection",
        )

    # then
    assert requests_mock.last_request.query == "api_key=my_api_key&nocache=true"


@mock.patch.object(roboflow.requests, "get")
def test_get_roboflow_model_type_when_connection_error_occurs(
    get_mock: MagicMock,
) -> None:
    # given
    get_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(DatasetLoadError):
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
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/1/?api_key=my_api_key&nocache=true"
        ),
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
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/1/?api_key=my_api_key&nocache=true"
        ),
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


def test_get_roboflow_model_type_when_default_model_canot_be_chosen(
    requests_mock: Mocker,
) -> None:
    # given
    requests_mock.get(
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/1/?api_key=my_api_key&nocache=true"
        ),
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
        url=wrap_url(
            f"{API_BASE_URL}/my_workspace/coins_detection/1/?api_key=my_api_key&nocache=true"
        ),
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


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_cache_is_utilised(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "project_task_type": "object-detection",
                    "model_type": "yolov8n",
                }
            )
        )

    # when
    result = get_model_type(model_id="some/1", api_key="my_api_key")

    # then
    construct_model_type_cache_path_mock.assert_called_once_with(
        dataset_id="some", version_id="1"
    )
    assert result == ("object-detection", "yolov8n")


@pytest.mark.parametrize(
    "model_id, expected_result",
    [
        ("clip/1", ("embed", "clip")),
        ("sam/1", ("embed", "sam")),
        ("gaze/1", ("gaze", "l2cs")),
    ],
)
def test_get_model_type_when_generic_model_is_utilised(
    model_id: str,
    expected_result: Tuple[TaskType, ModelType],
) -> None:
    # when
    result = get_model_type(model_id=model_id, api_key="my_api_key")

    # then
    assert result == expected_result


@mock.patch.object(roboflow, "get_roboflow_workspace")
@mock.patch.object(roboflow, "get_roboflow_dataset_type")
@mock.patch.object(roboflow, "get_roboflow_model_type")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_type_mock: MagicMock,
    get_roboflow_dataset_type_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_type_mock.return_value = "yolov8n"
    get_roboflow_dataset_type_mock.return_value = "object-detection"
    get_roboflow_workspace_mock.return_value = "my_workspace"

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov8n")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov8n"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_type_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my_workspace",
        dataset_id="some",
        version_id="1",
        project_task_type="object-detection",
    )
    get_roboflow_dataset_type_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my_workspace",
        dataset_id="some",
    )
    get_roboflow_workspace_mock.assert_called_once_with(api_key="my_api_key")


@mock.patch.object(roboflow, "get_model_type")
def test_roboflow_model_registry_get_model_on_cache_miss(
    get_model_type_mock: MagicMock,
) -> None:
    # given
    get_model_type_mock.return_value = ("object-detection", "yolov8n")
    registry = RoboflowModelRegistry(registry_dict={})

    # when
    with pytest.raises(ModelNotRecognisedError):
        _ = registry.get_model(model_id="some/1", api_key="my_api_key")


@mock.patch.object(roboflow, "get_model_type")
def test_roboflow_model_registry_get_model_on_cache_ht(
    get_model_type_mock: MagicMock,
) -> None:
    # given
    get_model_type_mock.return_value = ("object-detection", "yolov8n")
    registry = RoboflowModelRegistry(registry_dict={
        ("object-detection", "yolov8n"): "some"
    })

    # when
    result = registry.get_model(model_id="some/1", api_key="my_api_key")

    # then
    assert result == "some"