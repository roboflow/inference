import json
import os.path
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.cache import model_artifacts
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import ModelType, TaskType
from inference.core.exceptions import (
    MissingApiKeyError,
    ModelDeploymentNotSupportedError,
    ModelNotRecognisedError,
)
from inference.core.registries import roboflow
from inference.core.registries.roboflow import (
    FINE_TUNED_SAM3_DEPLOYMENT_ERROR,
    RoboflowModelRegistry,
    _in_process_metadata_cache,
    get_model_metadata_from_cache,
    get_model_type,
    model_metadata_content_is_invalid,
    save_model_metadata_in_cache,
)
from inference.core.roboflow_api import ModelEndpointType


@pytest.fixture(autouse=True)
def clear_in_process_metadata_cache():
    _in_process_metadata_cache.cache.clear()
    yield
    _in_process_metadata_cache.cache.clear()


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_does_not_exist(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    construct_model_type_cache_path_mock.return_value = os.path.join(
        empty_local_dir, "model_type.json"
    )

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_not_json(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("FOR SURE NOT JSON :)")

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_empty(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("")

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_is_invalid(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(json.dumps({"some": "key"}))

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_is_valid(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
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
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
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


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_save_model_metadata_in_cache(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
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


def test_save_and_load_model_metadata_in_cache_when_instant_model_slug_is_long(
    empty_local_dir: str,
) -> None:
    # given
    long_model_slug = "find-" + ("class-" * 60) + "instant-1"
    dataset_id = f"huizen/{long_model_slug}"

    # when
    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True):
        save_model_metadata_in_cache(
            dataset_id=dataset_id,
            version_id=None,
            project_task_type="object-detection",
            model_type="yolov8n",
        )
        _in_process_metadata_cache.cache.clear()
        result = get_model_metadata_from_cache(dataset_id=dataset_id, version_id=None)
        cache_path = roboflow.construct_model_type_cache_path(
            dataset_id=dataset_id, version_id=None
        )

    # then
    assert result == ("object-detection", "yolov8n")
    assert os.path.isfile(cache_path)
    assert all(
        len(os.fsencode(path_segment)) <= 255
        for path_segment in cache_path.split(os.sep)
        if path_segment
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


@mock.patch.object(roboflow, "SAM3_FINE_TUNED_MODELS_ENABLED", False)
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_fine_tuned_sam3_is_cached_but_disabled(
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
                    "project_task_type": "instance-segmentation",
                    "model_type": "sam3-large",
                }
            )
        )

    # when / then
    with pytest.raises(ModelDeploymentNotSupportedError) as error:
        get_model_type(model_id="workspace/123", api_key="my_api_key")

    assert str(error.value) == FINE_TUNED_SAM3_DEPLOYMENT_ERROR


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


@mock.patch.object(roboflow, "SAM3_FINE_TUNED_MODELS_ENABLED", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_fine_tuned_sam3_is_requested_and_disabled(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "instance-segmentation",
            "modelType": "sam3-large",
        }
    }

    # when / then
    with pytest.raises(ModelDeploymentNotSupportedError) as error:
        get_model_type(
            model_id="workspace/123",
            api_key="my_api_key",
        )

    assert str(error.value) == FINE_TUNED_SAM3_DEPLOYMENT_ERROR
    assert not os.path.exists(metadata_path)
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="workspace/123",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
            "modelType": "yolov8n",
        }
    }

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
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_and_model_type_specified_as_ort(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
            "modelType": "ort",
        }
    }

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov5v2s")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov5v2s"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_roboflow_api_is_called_for_model_from_new_model_registry(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "yolov8",
        "taskType": "object-detection",
    }

    # when
    result = get_model_type(
        model_id="dummy-model",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov8")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov8"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="dummy-model",
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_and_model_type_not_specified(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
        }
    }

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov5v2s")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov5v2s"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_and_project_type_not_specified(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {"ort": {}}

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov5v2s")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov5v2s"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_without_api_key_for_public_model(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
            "modelType": "yolov8n",
        }
    }

    # when
    result = get_model_type(
        model_id="some/1",
        api_key=None,
    )

    # then
    assert result == ("object-detection", "yolov8n")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov8n"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key=None,
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_roboflow_workspace")
@mock.patch.object(roboflow, "get_roboflow_dataset_type")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_mock(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_dataset_type_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_dataset_type_mock.return_value = "object-detection"
    get_roboflow_workspace_mock.return_value = "my_workspace"

    # when
    result = get_model_type(
        model_id="some/0",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "stub")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "stub"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_dataset_type_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my_workspace",
        dataset_id="some",
    )
    get_roboflow_workspace_mock.assert_called_once_with(api_key="my_api_key")


def test_get_model_type_when_roboflow_api_is_called_for_mock_without_api_key() -> None:
    with pytest.raises(MissingApiKeyError):
        _ = get_model_type(
            model_id="some/0",
            api_key=None,
        )


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
