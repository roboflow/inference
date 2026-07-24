import json
import os.path
from datetime import datetime, timedelta, timezone
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
def test_cache_entry_registration_for_base_cache_with_model_features(
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
        model_features={"some": "value"},
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


@pytest.mark.torch_models
@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "auto_negotiation_hash",
    ["../outside", "/absolute/path", "nested/path", "hash.with.dots", ""],
)
def test_generate_auto_resolution_cache_path_rejects_unsafe_hash(
    auto_negotiation_hash: str,
) -> None:
    with pytest.raises(ValueError):
        generate_auto_resolution_cache_path(
            auto_negotiation_hash=auto_negotiation_hash
        )


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_find_compatible_reuses_newest_entry_across_api_key_hashes(
    empty_local_dir: str,
) -> None:
    cache = BaseAutoLoadMetadataCache(file_lock_acquire_timeout=10)
    compatibility_hash = "c" * 64
    older_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="olderPackage",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now() - timedelta(seconds=1),
        offline_compatibility_hash=compatibility_hash,
        trusted_source=True,
    )
    newer_entry = older_entry.model_copy(
        update={
            "model_package_id": "newerPackage",
            # Mixed aware/naive timestamps are accepted by pydantic and must
            # remain orderable when cache entries came from different clients.
            "created_at": datetime.now(timezone.utc),
        }
    )

    with mock.patch.object(
        auto_resolution_cache, "INFERENCE_HOME", empty_local_dir
    ):
        # These exact hashes model requests made with different API keys. Their
        # credential-free compatibility hash intentionally remains the same.
        cache.register(auto_negotiation_hash="a" * 64, cache_entry=older_entry)
        cache.register(auto_negotiation_hash="b" * 64, cache_entry=newer_entry)
        candidates = cache.find_compatible_candidates(
            offline_compatibility_hash=compatibility_hash
        )
        result = cache.find_compatible(
            offline_compatibility_hash=compatibility_hash
        )

    assert candidates == [
        ("b" * 64, newer_entry),
        ("a" * 64, older_entry),
    ]
    assert result == ("b" * 64, newer_entry)


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_auto_resolution_cache_rejects_symlinked_cache_directory(
    empty_local_dir: str,
) -> None:
    outside_dir = os.path.join(empty_local_dir, "outside")
    os.makedirs(outside_dir)
    cache_dir = os.path.join(empty_local_dir, "auto-resolution-cache")
    os.symlink(outside_dir, cache_dir)
    cache = BaseAutoLoadMetadataCache(file_lock_acquire_timeout=10)
    cache_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        offline_compatibility_hash="c" * 64,
        trusted_source=True,
    )

    with mock.patch.object(
        auto_resolution_cache, "INFERENCE_HOME", empty_local_dir
    ):
        cache.register(auto_negotiation_hash="a" * 64, cache_entry=cache_entry)
        retrieved = cache.retrieve(auto_negotiation_hash="a" * 64)
        compatible = cache.find_compatible(
            offline_compatibility_hash="c" * 64
        )

    assert retrieved is None
    assert compatible is None
    assert os.listdir(outside_dir) == []


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_auto_resolution_cache_does_not_follow_symlinked_entry(
    empty_local_dir: str,
) -> None:
    outside_file = os.path.join(empty_local_dir, "outside.json")
    with open(outside_file, "w") as file:
        json.dump({"sentinel": True}, file)
    cache_dir = os.path.join(empty_local_dir, "auto-resolution-cache")
    os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, f"{'a' * 64}.json")
    os.symlink(outside_file, cache_path)
    cache = BaseAutoLoadMetadataCache(file_lock_acquire_timeout=10)
    cache_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        trusted_source=True,
    )

    with mock.patch.object(
        auto_resolution_cache, "INFERENCE_HOME", empty_local_dir
    ):
        cache.register(auto_negotiation_hash="a" * 64, cache_entry=cache_entry)
        assert cache.retrieve(auto_negotiation_hash="a" * 64) is None
        cache.invalidate(auto_negotiation_hash="a" * 64)

    assert os.path.islink(cache_path)
    with open(outside_file) as file:
        assert json.load(file) == {"sentinel": True}


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_auto_resolution_cache_does_not_follow_symlinked_lock_file(
    empty_local_dir: str,
) -> None:
    outside_file = os.path.join(empty_local_dir, "outside.lock")
    with open(outside_file, "w") as file:
        file.write("sentinel")
    cache_dir = os.path.join(empty_local_dir, "auto-resolution-cache")
    os.makedirs(cache_dir)
    cache_name = f"{'a' * 64}.json"
    cache_path = os.path.join(cache_dir, cache_name)
    lock_path = os.path.join(cache_dir, f".{cache_name}.lock")
    os.symlink(outside_file, lock_path)
    cache = BaseAutoLoadMetadataCache(file_lock_acquire_timeout=10)
    cache_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        trusted_source=True,
    )

    with mock.patch.object(
        auto_resolution_cache, "INFERENCE_HOME", empty_local_dir
    ):
        cache.register(auto_negotiation_hash="a" * 64, cache_entry=cache_entry)

    assert not os.path.exists(cache_path)
    assert os.path.islink(lock_path)
    with open(outside_file) as file:
        assert file.read() == "sentinel"
