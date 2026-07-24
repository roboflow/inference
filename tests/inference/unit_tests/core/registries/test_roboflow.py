import json
import os.path
from pathlib import Path
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
    roboflow._check_if_api_key_has_access_to_model.cache_clear()
    yield
    _in_process_metadata_cache.cache.clear()
    roboflow._check_if_api_key_has_access_to_model.cache_clear()


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


def test_model_metadata_cache_allows_mounted_symlink_cache_root(
    tmp_path: Path,
) -> None:
    real_cache_root = tmp_path / "real-cache"
    real_cache_root.mkdir()
    mounted_cache_root = tmp_path / "mounted-cache"
    mounted_cache_root.symlink_to(real_cache_root, target_is_directory=True)

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", str(mounted_cache_root)
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        save_model_metadata_in_cache(
            dataset_id="some",
            version_id="1",
            project_task_type="object-detection",
            model_type="yolov8n",
        )
        _in_process_metadata_cache.cache.clear()
        result = get_model_metadata_from_cache(
            dataset_id="some",
            version_id="1",
        )
        cache_path = roboflow.construct_model_type_cache_path(
            dataset_id="some",
            version_id="1",
        )

    assert result == ("object-detection", "yolov8n")
    assert cache_path == str(mounted_cache_root / "some" / "1" / "model_type.json")


def test_model_metadata_cache_returns_absolute_lexical_path_for_relative_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "relative-cache"):
        cache_path = roboflow.construct_model_type_cache_path(
            dataset_id="some",
            version_id="1",
        )

    assert cache_path == str(
        tmp_path / "relative-cache" / "some" / "1" / "model_type.json"
    )


def test_model_metadata_cache_rejects_cross_model_directory_symlink(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    owner_metadata_path = cache_root / "owner" / "1" / "model_type.json"
    owner_metadata_path.parent.mkdir(parents=True)
    owner_metadata = {
        "project_task_type": "classification",
        "model_type": "owner-model",
    }
    owner_metadata_path.write_text(json.dumps(owner_metadata))
    (cache_root / "requested").symlink_to(
        cache_root / "owner",
        target_is_directory=True,
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", str(cache_root)
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(
            dataset_id="requested",
            version_id="1",
        )
        with pytest.raises(ValueError, match="symbolic link"):
            save_model_metadata_in_cache(
                dataset_id="requested",
                version_id="1",
                project_task_type="object-detection",
                model_type="replacement-model",
            )

    assert result is None
    assert json.loads(owner_metadata_path.read_text()) == owner_metadata


def test_model_metadata_cache_rejects_final_symlink_and_preserves_outside_target(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    requested_model_dir = cache_root / "requested" / "1"
    requested_model_dir.mkdir(parents=True)
    outside_metadata_path = tmp_path / "outside-model-type.json"
    outside_metadata = {
        "project_task_type": "classification",
        "model_type": "outside-model",
    }
    outside_metadata_path.write_text(json.dumps(outside_metadata))
    (requested_model_dir / "model_type.json").symlink_to(outside_metadata_path)

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", str(cache_root)
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(
            dataset_id="requested",
            version_id="1",
        )
        with pytest.raises(ValueError, match="symbolic link"):
            save_model_metadata_in_cache(
                dataset_id="requested",
                version_id="1",
                project_task_type="object-detection",
                model_type="replacement-model",
            )

    assert result is None
    assert json.loads(outside_metadata_path.read_text()) == outside_metadata


def test_model_metadata_reader_closes_descriptor_when_fdopen_fails(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "model_type.json"
    metadata_path.write_text("{}")
    opened_descriptors = []
    real_open = os.open

    def record_open(*args, **kwargs) -> int:
        descriptor = real_open(*args, **kwargs)
        opened_descriptors.append(descriptor)
        return descriptor

    with (
        mock.patch.object(roboflow.os, "open", side_effect=record_open),
        mock.patch.object(
            roboflow.os,
            "fdopen",
            side_effect=OSError("fdopen failed"),
        ),
    ):
        with pytest.raises(OSError, match="fdopen failed"):
            roboflow._read_model_metadata_json(str(metadata_path))

    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is POSIX-only")
def test_model_metadata_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    metadata_path = tmp_path / "model_type.json"
    os.mkfifo(metadata_path)

    with pytest.raises(OSError, match="non-regular metadata"):
        roboflow._read_model_metadata_json(str(metadata_path))

    assert metadata_path.exists()


@pytest.mark.skipif(
    os.name != "posix",
    reason="Symlink-race simulation requires POSIX symlinks",
)
def test_model_metadata_reader_rejects_symlink_swap_without_no_follow(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "model_type.json"
    outside_metadata_path = tmp_path / "outside.json"
    metadata_path.write_text(json.dumps({"model_type": "inside"}))
    outside_metadata_path.write_text(json.dumps({"model_type": "outside"}))
    real_lstat = os.lstat
    path_swapped = False

    def lstat_then_swap(path: str) -> os.stat_result:
        nonlocal path_swapped
        file_status = real_lstat(path)
        if not path_swapped:
            Path(path).unlink()
            Path(path).symlink_to(outside_metadata_path)
            path_swapped = True
        return file_status

    with (
        mock.patch.object(
            roboflow.os,
            "lstat",
            side_effect=lstat_then_swap,
        ),
        mock.patch.object(
            roboflow.os,
            "O_NOFOLLOW",
            0,
            create=True,
        ),
    ):
        with pytest.raises(OSError, match="changed while it was being opened"):
            roboflow._read_model_metadata_json(str(metadata_path))

    assert metadata_path.is_symlink()
    assert json.loads(outside_metadata_path.read_text()) == {"model_type": "outside"}


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


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_offline_cache_auth_is_enabled_does_not_call_api(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
            },
            f,
        )

    # when
    with mock.patch.object(roboflow, "OFFLINE_MODE", True), mock.patch.object(
        roboflow, "MODELS_CACHE_AUTH_ENABLED", True
    ), mock.patch.object(
        roboflow, "_check_if_api_key_has_access_to_model"
    ) as auth_check_mock, mock.patch.object(
        roboflow, "get_model_metadata_from_inference_models_registry"
    ) as registry_api_mock, mock.patch.object(
        roboflow, "get_roboflow_model_data"
    ) as model_api_mock, mock.patch.object(
        roboflow, "get_roboflow_instant_model_data"
    ) as instant_model_api_mock:
        result = get_model_type(model_id="some/1", api_key="my_api_key")

    # then
    assert result == ("object-detection", "yolov8n")
    auth_check_mock.assert_not_called()
    registry_api_mock.assert_not_called()
    model_api_mock.assert_not_called()
    instant_model_api_mock.assert_not_called()


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_classification_subtype_is_cached(
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
                    "project_task_type": "multi-label-classification",
                    "model_type": "vit",
                }
            )
        )

    # when
    result = get_model_type(model_id="some/1", api_key="my_api_key")

    # then
    assert result == ("multi-label-classification", "vit")


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


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_check_api_key_for_yolo_world_core_model_uses_legacy_core_model_endpoint(
    get_roboflow_model_data_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
) -> None:
    # when
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="yolo_world/l",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is True
    get_model_metadata_from_inference_models_registry_mock.assert_not_called()
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="yolo_world/l",
        endpoint_type=ModelEndpointType.CORE_MODEL,
        device_id=GLOBAL_DEVICE_ID,
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "SAM3_FINE_TUNED_MODELS_ENABLED", False)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
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


@mock.patch.object(roboflow, "SAM3_FINE_TUNED_MODELS_ENABLED", False)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_sam3_from_new_model_registry_is_requested_and_disabled(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "sam3",
        "taskType": "instance-segmentation",
    }

    # when / then
    with pytest.raises(ModelDeploymentNotSupportedError) as error:
        get_model_type(
            model_id="workspace/123",
            api_key="my_api_key",
        )

    assert str(error.value) == FINE_TUNED_SAM3_DEPLOYMENT_ERROR
    assert not os.path.exists(metadata_path)
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="workspace/123",
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
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


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
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


@mock.patch.object(
    roboflow, "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", True
)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_for_local_inference_models_package_uses_declared_architecture(
    empty_local_dir: str,
) -> None:
    # given
    with open(os.path.join(empty_local_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "model_architecture": "depth-anything-v2",
                "task_type": "depth-estimation",
                "backend_type": "torch",
            },
            f,
        )

    # when
    result = get_model_type(model_id=empty_local_dir, api_key="my_api_key")

    # then
    assert result == ("depth-estimation", "depth-anything-v2")


@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_new_model_registry_returns_classification_subtype(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "vit",
        "taskType": "multi-label-classification",
    }

    # when
    result = get_model_type(
        model_id="animal-classification-9lufm/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("multi-label-classification", "vit")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "vit"
    assert persisted_metadata["project_task_type"] == "multi-label-classification"


@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_versioned_model_from_new_model_registry_is_requested(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "rfdetr",
        "taskType": "object-detection",
    }

    # when
    result = get_model_type(
        model_id="coco/38",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "rfdetr")
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="coco/38",
        countinference=None,
        service_secret=None,
    )
    get_roboflow_model_data_mock.assert_not_called()


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
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


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
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


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
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


# ---------------------------------------------------------------------------
# _get_model_metadata_from_inference_models_cache
# ---------------------------------------------------------------------------


def test_compat_cache_finder_supports_released_inference_models(
    tmp_path: Path,
) -> None:
    """The server can scan cache packages before the new helper is released."""
    from inference_models.models.auto_loaders import model_cache_paths

    model_id = "workspace/project/3"
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", str(tmp_path)):
        package_dir = Path(
            model_cache_paths.generate_model_package_cache_path(
                model_id=model_id, package_id="package1"
            )
        )
        package_dir.mkdir(parents=True)
        (package_dir / "model_config.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "task_type": "object-detection",
                    "model_architecture": "yolov8n",
                    "backend_type": "onnx",
                }
            )
        )

        result = roboflow._find_cached_model_package_dir_compat(model_id=model_id)

    assert result == str(package_dir.resolve())


@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            "model_id": "other/model/1",
            "task_type": "object-detection",
            "model_architecture": "yolov8n",
            "backend_type": "onnx",
        },
        {
            "model_id": "workspace/project/3",
            "task_type": ["object-detection"],
            "model_architecture": "yolov8n",
            "backend_type": "onnx",
        },
        {
            "model_id": "workspace/project/3",
            "task_type": "object-detection",
            "model_architecture": {"name": "yolov8n"},
            "backend_type": "onnx",
        },
    ],
)
def test_compat_cache_finder_rejects_malformed_metadata(
    tmp_path: Path,
    config: dict,
) -> None:
    from inference_models.models.auto_loaders import model_cache_paths

    model_id = "workspace/project/3"
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", str(tmp_path)):
        package_dir = Path(
            model_cache_paths.generate_model_package_cache_path(
                model_id=model_id,
                package_id="package1",
            )
        )
        package_dir.mkdir(parents=True)
        (package_dir / "model_config.json").write_text(json.dumps(config))

        result = roboflow._find_cached_model_package_dir_compat(model_id=model_id)

    assert result is None


def test_get_model_metadata_from_inference_models_cache_when_config_found(
    empty_local_dir: str,
) -> None:
    # given
    package_dir = os.path.join(empty_local_dir, "pkg001")
    os.makedirs(package_dir, exist_ok=True)
    with open(os.path.join(package_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "model_id": "coco/22",
                "task_type": "object-detection",
                "model_architecture": "yolov8",
                "backend_type": "onnx",
            },
            f,
        )
    # when
    with mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True), mock.patch.object(
        roboflow, "find_cached_model_package_dir", return_value=package_dir
    ):
        result = roboflow._get_model_metadata_from_inference_models_cache(
            model_id="coco/22"
        )

    # then
    assert result == ("object-detection", "yolov8")


def test_get_model_metadata_from_inference_models_cache_when_no_package_found() -> None:
    # when
    with mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True), mock.patch.object(
        roboflow, "find_cached_model_package_dir", return_value=None
    ):
        result = roboflow._get_model_metadata_from_inference_models_cache(
            model_id="coco/22"
        )

    # then
    assert result is None


def test_get_model_metadata_from_inference_models_cache_when_backend_disabled() -> None:
    # when
    with mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False):
        result = roboflow._get_model_metadata_from_inference_models_cache(
            model_id="coco/22"
        )

    # then
    assert result is None
