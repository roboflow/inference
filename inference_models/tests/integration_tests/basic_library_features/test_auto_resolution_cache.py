import json
import os.path
from datetime import datetime, timedelta
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference_models.models.auto_loaders import auto_resolution_cache
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCacheEntry,
    BaseAutoLoadMetadataCache,
    generate_auto_resolution_cache_path,
)
from inference_models.models.auto_loaders.entities import BackendType


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_registration_for_base_cache(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    cache_entry = AutoResolutionCacheEntry(
        model_id="my-model",
        model_package_id="my-package",
        resolved_files=["a", "b", "c"],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
    )
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    cache.register(
        auto_negotiation_hash="my-hash",
        cache_entry=cache_entry,
    )

    # then
    expected_cache_path = os.path.join(empty_local_dir, "my-hash.json")
    on_file_created.assert_called_once_with(
        expected_cache_path, "my-model", "my-package"
    )
    on_file_deleted.assert_not_called()
    assert os.path.exists(expected_cache_path)
    with open(expected_cache_path) as f:
        cache_contents = json.load(f)
    retrieved_entry = AutoResolutionCacheEntry.model_validate(cache_contents)
    assert cache_entry == retrieved_entry


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_retrieval_when_cache_file_does_not_exist(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    result = cache.retrieve(auto_negotiation_hash="some-hash")

    # then
    assert result is None
    on_file_deleted.assert_not_called()
    on_file_created.assert_not_called()


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_retrieval_when_cache_file_exist_but_is_not_json(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    expected_cache_path = os.path.join(empty_local_dir, f"some-hash.json")
    with open(expected_cache_path, "w") as f:
        f.write("ups...")
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    result = cache.retrieve(auto_negotiation_hash="some-hash")

    # then
    assert result is None
    on_file_deleted.assert_called_once_with(expected_cache_path)
    on_file_created.assert_not_called()


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_retrieval_when_cache_file_exist_but_is_empty(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    expected_cache_path = os.path.join(empty_local_dir, f"some-hash.json")
    with open(expected_cache_path, "w") as f:
        pass
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    result = cache.retrieve(auto_negotiation_hash="some-hash")

    # then
    assert result is None
    on_file_deleted.assert_called_once_with(expected_cache_path)
    on_file_created.assert_not_called()


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_retrieval_when_cache_file_exist_but_is_invalid_json(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    expected_cache_path = os.path.join(empty_local_dir, f"some-hash.json")
    with open(expected_cache_path, "w") as f:
        json.dump({"some": "value"}, f)
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    result = cache.retrieve(auto_negotiation_hash="some-hash")

    # then
    assert result is None
    on_file_deleted.assert_called_once_with(expected_cache_path)
    on_file_created.assert_not_called()


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_retrieval_when_cache_file_exist_but_is_expired(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    expected_cache_path = os.path.join(empty_local_dir, f"some-hash.json")
    cache_entry = AutoResolutionCacheEntry(
        model_id="my-model",
        model_package_id="my-package",
        resolved_files=["a", "b", "c"],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now() - timedelta(days=1, hours=1),
    )
    with open(expected_cache_path, "w") as f:
        json.dump(cache_entry.model_dump(mode="json"), f)
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    result = cache.retrieve(auto_negotiation_hash="some-hash")

    # then
    assert result is None
    on_file_deleted.assert_called_once_with(expected_cache_path)
    on_file_created.assert_not_called()
    assert not os.path.exists(expected_cache_path)


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_invalidation_when_file_exists(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    expected_cache_path = os.path.join(empty_local_dir, f"some-hash.json")
    cache_entry = AutoResolutionCacheEntry(
        model_id="my-model",
        model_package_id="my-package",
        resolved_files=["a", "b", "c"],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
    )
    with open(expected_cache_path, "w") as f:
        json.dump(cache_entry.model_dump(mode="json"), f)
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    cache.invalidate(auto_negotiation_hash="some-hash")

    # then
    on_file_deleted.assert_called_once_with(expected_cache_path)
    on_file_created.assert_not_called()
    assert not os.path.exists(expected_cache_path)


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
def test_cache_entry_invalidation_when_file_does_not_exist(
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: os.path.join(
            empty_local_dir, f"{auto_negotiation_hash}.json"
        )
    )
    expected_cache_path = os.path.join(empty_local_dir, f"some-hash.json")
    on_file_created, on_file_deleted = MagicMock(), MagicMock()
    cache = BaseAutoLoadMetadataCache(
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        on_file_deleted=on_file_deleted,
    )

    # when
    cache.invalidate(auto_negotiation_hash="some-hash")

    # then
    on_file_deleted.assert_not_called()
    on_file_created.assert_not_called()
    assert not os.path.exists(expected_cache_path)


@pytest.mark.torch_models
@pytest.mark.cpu_only
@mock.patch.object(auto_resolution_cache, "INFERENCE_HOME", "/some")
def test_generate_auto_resolution_cache_path() -> None:
    # when
    result = generate_auto_resolution_cache_path(auto_negotiation_hash="my-hash")

    # then
    assert result == "/some/auto-resolution-cache/my-hash.json"
