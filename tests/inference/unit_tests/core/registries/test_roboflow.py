import json
import os.path
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import ModelType, TaskType
from inference.core.exceptions import (
    MissingApiKeyError,
    ModelNotRecognisedError,
    RoboflowAPINotAuthorizedError,
)
from inference.core.registries import roboflow
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
    _check_if_api_key_has_access_to_model,
    _construct_model_auth_cache_key,
    _in_process_model_auth_cache,
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
    _in_process_model_auth_cache.cache.clear()
    yield
    _in_process_metadata_cache.cache.clear()
    _in_process_model_auth_cache.cache.clear()


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


@mock.patch.object(roboflow, "cache")
@mock.patch.object(roboflow, "_shared_model_auth_cache_enabled", return_value=True)
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_check_if_api_key_has_access_to_model_when_shared_cache_hit_skips_api_call(
    get_roboflow_model_data_mock: MagicMock,
    _shared_model_auth_cache_enabled_mock: MagicMock,
    cache_mock: MagicMock,
) -> None:
    # given
    cache_key = _construct_model_auth_cache_key(
        api_key="my_api_key",
        model_id="some/1",
        endpoint_type=ModelEndpointType.ORT,
        countinference=None,
        service_secret=None,
    )
    cache_mock.get.return_value = True

    # when
    result = _check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="some/1",
    )

    # then
    assert result is True
    cache_mock.get.assert_called_once_with(cache_key)
    cache_mock.set.assert_not_called()
    get_roboflow_model_data_mock.assert_not_called()


@mock.patch.object(roboflow, "cache")
@mock.patch.object(roboflow, "_shared_model_auth_cache_enabled", return_value=True)
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_check_if_api_key_has_access_to_model_when_authorized_result_cached_in_shared_cache(
    get_roboflow_model_data_mock: MagicMock,
    _shared_model_auth_cache_enabled_mock: MagicMock,
    cache_mock: MagicMock,
) -> None:
    # given
    cache_mock.get.return_value = None
    get_roboflow_model_data_mock.return_value = {"ort": {}}

    # when
    result = _check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="some/1",
    )

    # then
    assert result is True
    cache_key = cache_mock.set.call_args.kwargs["key"]
    assert "my_api_key" not in cache_key
    assert "some/1" in cache_key
    assert "api_key=" in cache_key
    cache_mock.set.assert_called_once_with(
        key=cache_key,
        value=True,
        expire=roboflow.MODELS_CACHE_AUTH_SHARED_CACHE_TTL,
    )
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "cache")
@mock.patch.object(roboflow, "_shared_model_auth_cache_enabled", return_value=True)
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_check_if_api_key_has_access_to_model_when_unauthorized_result_cached_with_short_ttl(
    get_roboflow_model_data_mock: MagicMock,
    _shared_model_auth_cache_enabled_mock: MagicMock,
    cache_mock: MagicMock,
) -> None:
    # given
    cache_mock.get.return_value = None
    get_roboflow_model_data_mock.side_effect = RoboflowAPINotAuthorizedError(
        "not allowed"
    )

    # when
    result = _check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="some/1",
    )

    # then
    assert result is False
    cache_mock.set.assert_called_once()
    assert cache_mock.set.call_args.kwargs["expire"] == (
        roboflow.MODELS_CACHE_AUTH_FAILURE_CACHE_TTL
    )


@mock.patch.object(roboflow, "cache")
@mock.patch.object(roboflow, "_shared_model_auth_cache_enabled", return_value=True)
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_check_if_api_key_has_access_to_model_when_upstream_error_occurs_then_error_is_not_retried(
    get_roboflow_model_data_mock: MagicMock,
    _shared_model_auth_cache_enabled_mock: MagicMock,
    cache_mock: MagicMock,
) -> None:
    # given
    cache_mock.get.return_value = None
    get_roboflow_model_data_mock.side_effect = RuntimeError("upstream unavailable")

    # when
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        _check_if_api_key_has_access_to_model(
            api_key="my_api_key",
            model_id="some/1",
        )

    # then
    get_roboflow_model_data_mock.assert_called_once()
    cache_mock.set.assert_not_called()


@mock.patch.object(roboflow, "cache")
@mock.patch.object(roboflow, "_shared_model_auth_cache_enabled", return_value=False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_check_if_api_key_has_access_to_model_when_shared_cache_disabled_skips_shared_cache(
    get_roboflow_model_data_mock: MagicMock,
    _shared_model_auth_cache_enabled_mock: MagicMock,
    cache_mock: MagicMock,
) -> None:
    # given
    get_roboflow_model_data_mock.return_value = {"ort": {}}

    # when
    result = _check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="some/1",
    )

    # then
    assert result is True
    cache_mock.get.assert_not_called()
    cache_mock.set.assert_not_called()
    get_roboflow_model_data_mock.assert_called_once()


@mock.patch.object(roboflow, "INTERNAL_WEIGHTS_URL_SUFFIX", "serverless")
@mock.patch.object(roboflow, "ROBOFLOW_SERVICE_SECRET", "shared-secret")
def test_construct_model_auth_cache_key_when_credit_bypass_context_changes() -> None:
    # when
    regular_key = _construct_model_auth_cache_key(
        api_key="my_api_key",
        model_id="some/1",
        endpoint_type=ModelEndpointType.ORT,
        countinference=True,
        service_secret=None,
    )
    bypass_key = _construct_model_auth_cache_key(
        api_key="my_api_key",
        model_id="some/1",
        endpoint_type=ModelEndpointType.ORT,
        countinference=False,
        service_secret="shared-secret",
    )

    # then
    assert "my_api_key" not in regular_key
    assert "shared-secret" not in bypass_key
    assert regular_key != bypass_key


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
