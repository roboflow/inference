import json
import os.path
from typing import Any, Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.entities.types import TaskType, ModelType
from inference.core.exceptions import (
    InvalidModelIDError,
    ModelNotRecognisedError,
)
from inference.core.registries.roboflow import (
    get_model_id_chunks,
    get_model_metadata_from_cache,
    model_metadata_content_is_invalid,
    save_model_metadata_in_cache,
    get_model_type,
    RoboflowModelRegistry,
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
    registry = RoboflowModelRegistry(
        registry_dict={("object-detection", "yolov8n"): "some"}
    )

    # when
    result = registry.get_model(model_id="some/1", api_key="my_api_key")

    # then
    assert result == "some"
