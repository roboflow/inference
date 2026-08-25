import json
import os.path
from pathlib import Path
from typing import Any, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.cache import model_artifacts
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import ModelType, TaskType
from inference.core.exceptions import (
    MissingApiKeyError,
    ModelArtefactError,
    ModelDeploymentNotSupportedError,
    ModelNotRecognisedError,
    RoboflowAPINotAuthorizedError,
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
from inference.usage_tracking.model_types import (
    ModelDescriptor,
    clear_recorded_model_descriptors,
    get_recorded_model_descriptor,
)


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


def test_in_process_model_metadata_cache_is_scoped_by_api_key() -> None:
    with mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow,
        "_get_model_metadata_from_cache",
        side_effect=[
            ("object-detection", "model-a"),
            ("classification", "model-b"),
        ],
    ) as load_metadata:
        first = get_model_metadata_from_cache(
            dataset_id="workspace/model",
            version_id="1",
            api_key="credential-a",
        )
        second = get_model_metadata_from_cache(
            dataset_id="workspace/model",
            version_id="1",
            api_key="credential-b",
        )

    assert first == ("object-detection", "model-a")
    assert second == ("classification", "model-b")
    assert load_metadata.call_count == 2


def test_save_model_metadata_populates_credential_scoped_in_process_cache() -> None:
    with mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "_save_model_metadata_in_cache"
    ), mock.patch.object(
        roboflow,
        "_get_model_metadata_from_cache",
        return_value=("classification", "model-b"),
    ) as load_metadata:
        save_model_metadata_in_cache(
            dataset_id="workspace/model",
            version_id="1",
            project_task_type="object-detection",
            model_type="model-a",
            api_key="credential-a",
        )
        same_credential = get_model_metadata_from_cache(
            dataset_id="workspace/model",
            version_id="1",
            api_key="credential-a",
        )
        different_credential = get_model_metadata_from_cache(
            dataset_id="workspace/model",
            version_id="1",
            api_key="credential-b",
        )

    assert same_credential == ("object-detection", "model-a")
    assert different_credential == ("classification", "model-b")
    load_metadata.assert_called_once_with(
        dataset_id="workspace/model",
        version_id="1",
        api_key="credential-b",
    )


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
    assert result["model_id"] == "some/1"
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
        with open(cache_path) as metadata_file:
            persisted_metadata = json.load(metadata_file)

    # then
    assert result == ("object-detection", "yolov8n")
    assert persisted_metadata["model_id"] == dataset_id
    assert os.path.isfile(cache_path)
    assert all(
        len(os.fsencode(path_segment)) <= 255
        for path_segment in cache_path.split(os.sep)
        if path_segment
    )


@pytest.mark.parametrize("existing_owner", [None, "", "workspace/different"])
def test_save_model_metadata_refuses_to_claim_unowned_generated_v2_path(
    empty_local_dir: str,
    existing_owner: Optional[str],
) -> None:
    model_id = f"workspace/{'x' * 300}"
    current_cache_key = model_artifacts.get_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=empty_local_dir,
    )
    metadata = {
        "project_task_type": "classification",
        "model_type": "old-raw-model",
    }
    if existing_owner is not None:
        metadata["model_id"] = existing_owner
    metadata_path = Path(empty_local_dir) / current_cache_key / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata))

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True):
        with pytest.raises(ModelArtefactError, match="Refusing to claim"):
            save_model_metadata_in_cache(
                dataset_id=model_id,
                version_id=None,
                project_task_type="object-detection",
                model_type="yolov8n",
            )

    assert json.loads(metadata_path.read_text()) == metadata


def test_save_model_metadata_refuses_nonempty_generated_v2_path_without_owner(
    empty_local_dir: str,
) -> None:
    model_id = f"workspace/{'x' * 300}"
    current_cache_key = model_artifacts.get_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=empty_local_dir,
    )
    model_cache_dir = Path(empty_local_dir) / current_cache_key
    model_cache_dir.mkdir(parents=True)
    (model_cache_dir / "weights.bin").write_bytes(b"old raw model")

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True):
        with pytest.raises(ModelArtefactError, match="non-empty generated"):
            save_model_metadata_in_cache(
                dataset_id=model_id,
                version_id=None,
                project_task_type="object-detection",
                model_type="yolov8n",
            )

    assert not (model_cache_dir / "model_type.json").exists()
    assert (model_cache_dir / "weights.bin").read_bytes() == b"old raw model"


def test_save_model_metadata_updates_exactly_owned_generated_v2_path(
    empty_local_dir: str,
) -> None:
    model_id = f"workspace/{'x' * 300}"
    current_cache_key = model_artifacts.get_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=empty_local_dir,
    )
    metadata_path = Path(empty_local_dir) / current_cache_key / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "project_task_type": "classification",
                "model_type": "vit",
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True):
        save_model_metadata_in_cache(
            dataset_id=model_id,
            version_id=None,
            project_task_type="object-detection",
            model_type="yolov8n",
        )

    assert json.loads(metadata_path.read_text()) == {
        "model_id": model_id,
        "project_task_type": "object-detection",
        "model_type": "yolov8n",
    }


def test_model_metadata_cache_reads_owned_legacy_slug(
    empty_local_dir: str,
) -> None:
    model_id = f"workspace/{'x' * 300}"
    legacy_cache_key = model_artifacts.get_legacy_model_id_cache_path(
        model_id=model_id, cache_dir_root=empty_local_dir
    )
    assert legacy_cache_key is not None
    metadata_path = Path(empty_local_dir) / legacy_cache_key / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
                "model_id": model_id,
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(dataset_id=model_id, version_id=None)

    assert result == ("object-detection", "yolov8n")


def test_model_metadata_cache_reads_owned_legacy_raw_path(
    empty_local_dir: str,
) -> None:
    model_id = "Workspace/Model/1"
    current_cache_key = model_artifacts.get_model_id_cache_path(
        model_id=model_id, cache_dir_root=empty_local_dir
    )
    legacy_cache_key = model_artifacts.get_legacy_model_id_cache_path(
        model_id=model_id, cache_dir_root=empty_local_dir
    )
    assert current_cache_key != model_id
    assert legacy_cache_key == model_id
    metadata_path = Path(empty_local_dir) / legacy_cache_key / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
                "model_id": model_id,
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(dataset_id=model_id, version_id=None)

    assert result == ("object-detection", "yolov8n")


def test_model_metadata_cache_rejects_unowned_legacy_raw_path(
    empty_local_dir: str,
) -> None:
    model_id = "Workspace/Model/1"
    metadata_path = Path(empty_local_dir) / model_id / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "project_task_type": "object-detection",
                "model_type": "attacker-controlled",
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(dataset_id=model_id, version_id=None)

    assert result is None


@pytest.mark.parametrize("cached_model_id", [None, "", "workspace/different"])
def test_model_metadata_cache_rejects_unowned_current_v2_slug(
    empty_local_dir: str, cached_model_id: Optional[str]
) -> None:
    model_id = f"workspace/{'x' * 300}"
    current_cache_key = model_artifacts.get_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=empty_local_dir,
    )
    assert current_cache_key.startswith(
        model_artifacts.MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX
    )
    metadata = {
        "project_task_type": "object-detection",
        "model_type": "attacker-controlled",
    }
    if cached_model_id is not None:
        metadata["model_id"] = cached_model_id
    metadata_path = Path(empty_local_dir) / current_cache_key / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata))

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(
            dataset_id=model_id,
            version_id=None,
        )

    assert result is None


def test_model_metadata_cache_does_not_confuse_old_raw_id_with_v2_slug(
    empty_local_dir: str,
) -> None:
    victim_model_id = f"workspace/{'x' * 300}"
    generated_v2_key = model_artifacts.get_model_id_cache_path(
        model_id=victim_model_id,
        cache_dir_root=empty_local_dir,
    )
    metadata_path = Path(empty_local_dir) / generated_v2_key / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "project_task_type": "object-detection",
                "model_type": "old-raw-model",
                # This is the exact identity an old writer could have stored
                # ownerlessly at the path now reserved for the victim's V2 key.
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        assert (
            get_model_metadata_from_cache(
                dataset_id=victim_model_id,
                version_id=None,
            )
            is None
        )


@pytest.mark.parametrize("cached_model_id", [None, "", "workspace/different"])
def test_model_metadata_cache_rejects_unowned_legacy_slug(
    empty_local_dir: str, cached_model_id: Optional[str]
) -> None:
    model_id = f"workspace/{'x' * 300}"
    legacy_cache_key = model_artifacts.get_legacy_model_id_cache_path(
        model_id=model_id, cache_dir_root=empty_local_dir
    )
    assert legacy_cache_key is not None
    metadata = {
        "project_task_type": "object-detection",
        "model_type": "attacker-controlled",
    }
    if cached_model_id is not None:
        metadata["model_id"] = cached_model_id
    metadata_path = Path(empty_local_dir) / legacy_cache_key / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata))

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(dataset_id=model_id, version_id=None)

    assert result is None


def test_model_metadata_cache_keeps_ownerless_safe_raw_path_compatible(
    empty_local_dir: str,
) -> None:
    metadata_path = Path(empty_local_dir) / "some" / "1" / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")
        constructed_path = roboflow.construct_model_type_cache_path(
            dataset_id="some", version_id="1"
        )

    assert result == ("object-detection", "yolov8n")
    assert constructed_path == str(metadata_path)


@pytest.mark.parametrize("cached_model_id", [None, "", "other/1"])
def test_model_metadata_cache_rejects_invalid_owner_on_safe_raw_path(
    empty_local_dir: str,
    cached_model_id: Optional[str],
) -> None:
    metadata_path = Path(empty_local_dir) / "some" / "1" / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "project_task_type": "classification",
                "model_type": "wrong-owner-type",
                "model_id": cached_model_id,
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True), mock.patch.object(
        roboflow, "USE_INFERENCE_MODELS", False
    ):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    assert result is None


@pytest.mark.parametrize("cached_model_id", [None, "", "other/1"])
def test_save_model_metadata_refuses_invalid_owner_on_safe_raw_path(
    empty_local_dir: str,
    cached_model_id: Optional[str],
) -> None:
    existing_metadata = {
        "project_task_type": "classification",
        "model_type": "wrong-owner-type",
        "model_id": cached_model_id,
    }
    metadata_path = Path(empty_local_dir) / "some" / "1" / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(existing_metadata))

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True):
        with pytest.raises(ModelArtefactError, match="Refusing to claim raw"):
            save_model_metadata_in_cache(
                dataset_id="some",
                version_id="1",
                project_task_type="object-detection",
                model_type="yolov8n",
            )

    assert json.loads(metadata_path.read_text()) == existing_metadata


def test_save_model_metadata_upgrades_ownerless_safe_raw_path(
    empty_local_dir: str,
) -> None:
    metadata_path = Path(empty_local_dir) / "some" / "1" / "model_type.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "project_task_type": "classification",
                "model_type": "vit",
            }
        )
    )

    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True):
        save_model_metadata_in_cache(
            dataset_id="some",
            version_id="1",
            project_task_type="object-detection",
            model_type="yolov8n",
        )

    assert json.loads(metadata_path.read_text()) == {
        "project_task_type": "object-detection",
        "model_type": "yolov8n",
        "model_id": "some/1",
    }


def test_model_metadata_cache_rejects_ambiguous_id_before_memory_lookup() -> None:
    with mock.patch.object(_in_process_metadata_cache, "get") as memory_get_mock:
        with pytest.raises(ValueError, match="unsafe or ambiguous path segment"):
            get_model_metadata_from_cache(dataset_id="victim", version_id=".")

    memory_get_mock.assert_not_called()


def test_model_metadata_cache_rejects_ambiguous_id_before_write_lock() -> None:
    with mock.patch.object(roboflow, "LAMBDA", False), mock.patch.object(
        roboflow.cache, "lock"
    ) as lock_mock:
        with pytest.raises(ValueError, match="unsafe or ambiguous path segment"):
            save_model_metadata_in_cache(
                dataset_id="victim",
                version_id=".",
                project_task_type="object-detection",
                model_type="yolov8n",
            )

    lock_mock.assert_not_called()


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
    try:
        result = get_model_type(model_id=model_id, api_key="my_api_key")

        # then
        assert result == expected_result
    finally:
        clear_recorded_model_descriptors()


@pytest.mark.parametrize(
    "model_id, expected_descriptor",
    [
        ("sam2/hiera_large", ModelDescriptor("sam2", "hiera_large")),
        ("sam2/hiera_small", ModelDescriptor("sam2", "hiera_small")),
        ("sam2/hiera_tiny", ModelDescriptor("sam2", "hiera_tiny")),
        ("sam2/hiera_b_plus", ModelDescriptor("sam2", "hiera_b_plus")),
        ("sam3/sam3_final", ModelDescriptor("sam3", "sam3_final")),
        ("sam3/sam3_interactive", ModelDescriptor("sam3", "sam3_interactive")),
        ("yolo_world/l", ModelDescriptor("yolo-world", "l")),
        # A bare architecture id is served in a single flavour - no variant.
        ("clip", ModelDescriptor("clip", None)),
        ("qwen3_5-0.8b", ModelDescriptor("qwen3_5-0.8b", None)),
    ],
)
def test_get_model_type_records_coded_model_suffix_as_usage_variant(
    model_id: str,
    expected_descriptor: ModelDescriptor,
) -> None:
    # when
    try:
        task_type, model_type = get_model_type(model_id=model_id, api_key="my_api_key")

        # then - class lookup stays on the architecture, usage keeps both labels
        assert task_type
        assert model_type == expected_descriptor.architecture
        assert get_recorded_model_descriptor(model_id) == expected_descriptor
    finally:
        clear_recorded_model_descriptors()


def test_model_pipelines_enumerate_all_coded_pp_ocr_ids() -> None:
    # given
    stage_variants = ("none", "tiny", "small", "medium")
    expected_combo_ids = {
        f"pp_ocr/{text_detection}-{text_recognition}"
        for text_detection in stage_variants
        for text_recognition in stage_variants
        if (text_detection, text_recognition) != ("none", "none")
    }
    expected_single_token_ids = {"pp_ocr/tiny", "pp_ocr/small", "pp_ocr/medium"}

    # then
    assert set(roboflow.MODEL_PIPELINES) == (
        expected_combo_ids | expected_single_token_ids | {"pp_ocr"}
    )
    for definition in roboflow.MODEL_PIPELINES.values():
        assert (definition.task_type, definition.model_type) == ("ocr", "pp_ocr")
        assert len(definition.downstream_model_ids) > 0
    assert "pp_ocr/none-none" not in roboflow.MODEL_PIPELINES
    assert "pp_ocr/none" not in roboflow.MODEL_PIPELINES
    # pipeline IDs must not leak into GENERIC_MODELS - auth treats them differently
    assert all(
        model_id not in roboflow.GENERIC_MODELS for model_id in roboflow.MODEL_PIPELINES
    )


@pytest.mark.parametrize(
    "model_id, expected_downstream",
    [
        ("pp_ocr/small-small", ("pp-ocrv6-det/small", "pp-ocrv6-rec/small")),
        ("pp_ocr/tiny-medium", ("pp-ocrv6-det/tiny", "pp-ocrv6-rec/medium")),
        ("pp_ocr/none-small", ("pp-ocrv6-rec/small",)),
        ("pp_ocr/medium-none", ("pp-ocrv6-det/medium",)),
        ("pp_ocr/tiny", ("pp-ocrv6-det/tiny", "pp-ocrv6-rec/tiny")),
        ("pp_ocr", ("pp-ocrv6-det/small", "pp-ocrv6-rec/small")),
    ],
)
def test_model_pipelines_map_to_expected_downstream_models(
    model_id: str, expected_downstream: Tuple[str, ...]
) -> None:
    assert (
        roboflow.MODEL_PIPELINES[model_id].downstream_model_ids == expected_downstream
    )


@pytest.mark.parametrize(
    "model_id, expected_downstream",
    [
        ("pp_ocr/tiny-medium", ("pp-ocrv6-det/tiny", "pp-ocrv6-rec/medium")),
        ("pp_ocr/none-small", ("pp-ocrv6-rec/small",)),
        ("pp_ocr/medium-none", ("pp-ocrv6-det/medium",)),
    ],
)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
def test_check_api_key_for_pp_ocr_pipeline_authorizes_downstream_models(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    model_id: str,
    expected_downstream: Tuple[str, ...],
) -> None:
    # when
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key=f"my_api_key-{model_id}",
        model_id=model_id,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then - the synthetic pipeline ID itself must never reach the remote registry,
    # but every downstream stage model must be authorized against it
    assert result is True
    checked_model_ids = [
        call.kwargs["model_id"]
        for call in get_model_metadata_from_inference_models_registry_mock.call_args_list
    ]
    assert checked_model_ids == list(expected_downstream)


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
def test_check_api_key_for_pp_ocr_pipeline_fails_when_downstream_model_not_authorized(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
) -> None:
    # given - detection stage authorized, recognition stage not
    def _registry_response(api_key: str, model_id: str, **kwargs):
        if model_id == "pp-ocrv6-rec/medium":
            raise RoboflowAPINotAuthorizedError()
        return {"taskType": "ocr"}

    get_model_metadata_from_inference_models_registry_mock.side_effect = (
        _registry_response
    )

    # when
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="pp_ocr/tiny-medium",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False


@pytest.mark.parametrize("model_id", ["pp_ocr/small-small", "pp_ocr"])
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(
    roboflow,
    "get_roboflow_instant_model_data",
    side_effect=RoboflowAPINotAuthorizedError,
)
@mock.patch.object(
    roboflow,
    "get_roboflow_model_data",
    side_effect=RoboflowAPINotAuthorizedError,
)
def test_check_api_key_for_pp_ocr_pipeline_not_recognized_without_inference_models(
    get_roboflow_model_data_mock: MagicMock,
    get_roboflow_instant_model_data_mock: MagicMock,
    model_id: str,
) -> None:
    # when - with USE_INFERENCE_MODELS disabled, pipeline IDs fall through to the
    # regular resolution and fail closed there
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id=model_id,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False


@pytest.mark.parametrize("model_id", ["pp_ocr/none-none", "pp_ocr/huge-small"])
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(
    roboflow,
    "get_model_metadata_from_inference_models_registry",
    side_effect=RoboflowAPINotAuthorizedError,
)
def test_check_api_key_for_invalid_pp_ocr_pipeline_fails_closed(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    model_id: str,
) -> None:
    # when - IDs outside the coded pipeline set are not treated as pipelines
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key=f"my_api_key-{model_id}",
        model_id=model_id,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key=f"my_api_key-{model_id}",
        model_id=model_id,
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(
    roboflow,
    "get_model_metadata_from_inference_models_registry",
    side_effect=RoboflowAPINotAuthorizedError,
)
def test_check_api_key_does_not_blanket_trust_generic_models(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
) -> None:
    # when - full-ID GENERIC_MODELS entries (e.g. sam3/sam3_interactive) must still
    # be authorized remotely; regression guard against trusting GENERIC_MODELS as such
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="sam3/sam3_interactive",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False
    get_model_metadata_from_inference_models_registry_mock.assert_called_once()


@pytest.mark.parametrize(
    "model_id",
    ["pp_ocr", "pp_ocr/small", "pp_ocr/tiny-medium", "pp_ocr/none-small"],
)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_roboflow_instant_model_data")
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
def test_get_model_type_for_pipeline_when_inference_models_enabled(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    get_roboflow_instant_model_data_mock: MagicMock,
    model_id: str,
) -> None:
    # when
    try:
        result = get_model_type(model_id=model_id, api_key="my_api_key")

        # then - pipeline recognition is static and must not call any remote API.
        # Usage keeps the requested stage sizes so tiny vs small stay distinct.
        assert result == ("ocr", "pp_ocr")
        _, _, expected_variant = model_id.partition("/")
        assert get_recorded_model_descriptor(model_id) == ModelDescriptor(
            "pp_ocr", expected_variant or None
        )
        get_model_metadata_from_inference_models_registry_mock.assert_not_called()
        get_roboflow_model_data_mock.assert_not_called()
        get_roboflow_instant_model_data_mock.assert_not_called()
    finally:
        clear_recorded_model_descriptors()


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_get_model_type_for_pipeline_when_inference_models_disabled(
    get_roboflow_model_data_mock: MagicMock,
) -> None:
    # given - with the flag off, pipeline IDs are not recognized and resolution
    # falls through to the regular Roboflow API pathway
    get_roboflow_model_data_mock.side_effect = RoboflowAPINotAuthorizedError()

    # when / then
    with pytest.raises(RoboflowAPINotAuthorizedError):
        get_model_type(model_id="pp_ocr/small-small", api_key="my_api_key")
    get_roboflow_model_data_mock.assert_called_once()


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


@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_records_registry_variant_for_usage_tracking(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "yolov8",
        "taskType": "instance-segmentation",
        "modelVariant": "yolov8-n",
    }

    try:
        result = get_model_type(
            model_id="yolov8n-seg-640",
            api_key="my_api_key",
        )

        assert result == ("instance-segmentation", "yolov8")
        assert get_recorded_model_descriptor("yolov8n-seg-640") == ModelDescriptor(
            "yolov8", "yolov8-n"
        )
        with open(metadata_path) as f:
            persisted_metadata = json.load(f)
        assert persisted_metadata["model_type"] == "yolov8"
        assert persisted_metadata["model_variant"] == "yolov8-n"

        _in_process_metadata_cache.cache.clear()
        clear_recorded_model_descriptors()
        cached_result = get_model_type(
            model_id="yolov8n-seg-640",
            api_key="my_api_key",
        )
        assert cached_result == ("instance-segmentation", "yolov8")
        assert get_recorded_model_descriptor("yolov8n-seg-640") == ModelDescriptor(
            "yolov8", "yolov8-n"
        )
        get_model_metadata_from_inference_models_registry_mock.assert_called_once()
    finally:
        clear_recorded_model_descriptors()


@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_records_architecture_when_registry_omits_variant(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given - older projects resolve without a modelVariant field
    construct_model_type_cache_path_mock.return_value = os.path.join(
        empty_local_dir, "model_type.json"
    )
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "yolov8",
        "taskType": "object-detection",
    }

    # when
    try:
        result = get_model_type(model_id="some-project/3", api_key="my_api_key")

        # then - the architecture is still labelled, the variant stays absent
        assert result == ("object-detection", "yolov8")
        assert get_recorded_model_descriptor("some-project/3") == ModelDescriptor(
            "yolov8", None
        )
    finally:
        clear_recorded_model_descriptors()


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
    ) as find_cached_package, mock.patch.object(
        roboflow, "load_record_raw", return_value=None
    ):
        result = roboflow._get_model_metadata_from_inference_models_cache(
            model_id="coco/22",
            api_key="credential-a",
        )

    # then
    assert result == ("object-detection", "yolov8", None)
    find_cached_package.assert_called_once_with(
        model_id="coco/22",
        api_key="credential-a",
    )


def test_get_model_metadata_from_inference_models_cache_reads_offline_registry_variant(
    empty_local_dir: str,
) -> None:
    package_dir = os.path.join(empty_local_dir, "pkg001")
    os.makedirs(package_dir, exist_ok=True)
    with open(os.path.join(package_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "model_id": "coco/38",
                "task_type": "object-detection",
                "model_architecture": "rfdetr",
                "backend_type": "torch",
            },
            f,
        )

    with mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True), mock.patch.object(
        roboflow, "find_cached_model_package_dir", return_value=package_dir
    ), mock.patch.object(
        roboflow,
        "load_record_raw",
        return_value={
            "canonical_model_id": "coco/38",
            "model": {
                "model_architecture": "rfdetr",
                "task_type": "object-detection",
                "model_variant": "rfdetr-nano",
            },
        },
    ) as load_record_raw_mock:
        result = roboflow._get_model_metadata_from_inference_models_cache(
            model_id="coco/38",
            api_key="credential-a",
        )

    assert result == ("object-detection", "rfdetr", "rfdetr-nano")
    load_record_raw_mock.assert_called_once_with(model_id="coco/38")


def test_get_model_metadata_from_inference_models_cache_reads_variant_by_canonical_id(
    empty_local_dir: str,
) -> None:
    package_dir = os.path.join(empty_local_dir, "pkg001")
    os.makedirs(package_dir, exist_ok=True)
    with open(os.path.join(package_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "model_id": "rfdetr-nano",
                "canonical_model_id": "coco/38",
                "task_type": "object-detection",
                "model_architecture": "rfdetr",
                "backend_type": "torch",
            },
            f,
        )

    def _load_record(model_id: str):
        if model_id == "coco/38":
            return {
                "canonical_model_id": "coco/38",
                "model": {"model_variant": "rfdetr-nano"},
            }
        return None

    with mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True), mock.patch.object(
        roboflow, "find_cached_model_package_dir", return_value=package_dir
    ), mock.patch.object(
        roboflow, "load_record_raw", side_effect=_load_record
    ) as load_record_raw_mock:
        result = roboflow._get_model_metadata_from_inference_models_cache(
            model_id="rfdetr-nano"
        )

    assert result == ("object-detection", "rfdetr", "rfdetr-nano")
    assert [
        call.kwargs["model_id"] for call in load_record_raw_mock.call_args_list
    ] == [
        "rfdetr-nano",
        "coco/38",
    ]


@pytest.mark.parametrize("corrupt_value", [["coco/38"], {"id": "coco/38"}, 38, None])
def test_get_model_metadata_from_inference_models_cache_survives_corrupt_id_fields(
    corrupt_value: Any,
    empty_local_dir: str,
) -> None:
    # given - model_config.json is read without schema validation, so a
    # hand-edited or truncated file must degrade to "no variant", not raise
    package_dir = os.path.join(empty_local_dir, "pkg001")
    os.makedirs(package_dir, exist_ok=True)
    with open(os.path.join(package_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "model_id": corrupt_value,
                "canonical_model_id": corrupt_value,
                "task_type": "object-detection",
                "model_architecture": "rfdetr",
                "backend_type": "torch",
            },
            f,
        )

    # when
    with mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True), mock.patch.object(
        roboflow, "find_cached_model_package_dir", return_value=package_dir
    ), mock.patch.object(
        roboflow, "load_record_raw", return_value=None
    ) as load_record_raw_mock:
        result = roboflow._get_model_metadata_from_inference_models_cache(
            model_id="coco/38"
        )

    # then - the unusable candidates are skipped, the requested id is still tried
    assert result == ("object-detection", "rfdetr", None)
    assert [
        call.kwargs["model_id"] for call in load_record_raw_mock.call_args_list
    ] == ["coco/38"]


@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "find_cached_model_package_dir")
@mock.patch.object(roboflow, "load_record_raw")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_records_offline_registry_variant_when_model_type_json_missing(
    load_record_raw_mock: MagicMock,
    find_cached_model_package_dir_mock: MagicMock,
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    construct_model_type_cache_path_mock.return_value = os.path.join(
        empty_local_dir, "missing", "model_type.json"
    )
    package_dir = os.path.join(empty_local_dir, "pkg001")
    os.makedirs(package_dir, exist_ok=True)
    with open(os.path.join(package_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "model_id": "coco/38",
                "task_type": "object-detection",
                "model_architecture": "rfdetr",
                "backend_type": "torch",
            },
            f,
        )
    find_cached_model_package_dir_mock.return_value = package_dir
    load_record_raw_mock.return_value = {
        "canonical_model_id": "coco/38",
        "model": {
            "model_architecture": "rfdetr",
            "model_variant": "rfdetr-nano",
        },
    }

    try:
        result = get_model_type(model_id="coco/38", api_key="my_api_key")

        assert result == ("object-detection", "rfdetr")
        assert get_recorded_model_descriptor("coco/38") == ModelDescriptor(
            "rfdetr", "rfdetr-nano"
        )
        get_model_metadata_from_inference_models_registry_mock.assert_not_called()
    finally:
        clear_recorded_model_descriptors()


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
