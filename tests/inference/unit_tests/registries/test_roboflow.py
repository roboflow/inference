import json
import os.path
from typing import Any, Type
from unittest import mock
from unittest.mock import MagicMock

import pytest
import requests.exceptions

from inference.core.exceptions import (
    InvalidModelIDError,
    MalformedRoboflowAPIResponseError,
)
from inference.core.registries.roboflow import (
    get_model_id_chunks,
    get_model_metadata_from_cache,
    model_metadata_content_is_invalid,
    save_model_metadata_in_cache,
    wrap_roboflow_api_errors,
    raise_from_lambda,
)
from inference.core.registries import roboflow


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
    construct_model_type_cache_path: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    construct_model_type_cache_path.return_value = os.path.join(
        empty_local_dir, "model_type.json"
    )

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_not_json(
    construct_model_type_cache_path: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("FOR SURE NOT JSON :)")

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_empty(
    construct_model_type_cache_path: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("")

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_is_invalid(
    construct_model_type_cache_path: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(json.dumps({"some": "key"}))

    # when
    result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_invalid(
    construct_model_type_cache_path: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path.return_value = metadata_path
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
    construct_model_type_cache_path: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path.return_value = metadata_path

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
    construct_model_type_cache_path.assert_called_once_with(
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
