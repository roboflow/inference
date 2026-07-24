import hashlib
import inspect
import json
import os.path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from inference_models import ClassificationPrediction
from inference_models.errors import (
    CorruptedModelPackageError,
    InsecureModelIdentifierError,
    InvalidParameterError,
    ModelLoadingError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
    UntrustedFileError,
)
from inference_models.models.auto_loaders import core, model_cache_paths
from inference_models.models.auto_loaders.access_manager import (
    AccessIdentifiers,
    LiberalModelAccessManager,
)
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCache,
    AutoResolutionCacheEntry,
)
from inference_models.models.auto_loaders.core import (
    attempt_loading_matching_model_packages,
    attempt_loading_model_from_local_storage,
    attempt_loading_model_from_offline_cache,
    attempt_loading_model_with_auto_load_cache,
    create_symlinks_to_shared_blobs,
    dump_auto_resolution_cache,
    dump_model_config_for_offline_use,
    find_cached_model_package_dir,
    generate_model_package_cache_path,
    initialize_model,
    load_class_from_path,
    parse_model_config,
    resolve_recommended_parameters,
)
from inference_models.models.auto_loaders.entities import (
    MODEL_CONFIG_FILE_NAME,
    BackendType,
    InferenceModelConfig,
)
from inference_models.weights_providers import core as weights_providers_core
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    LocalFileArtefactSpecs,
    ModelDependency,
    ModelMetadata,
    ModelPackageMetadata,
    PackageSourceType,
    Quantization,
    RecommendedParameters,
)

TEST_PACKAGE_MANIFEST_HASH = "d" * 64


@pytest.fixture(autouse=True)
def clear_implicit_roboflow_api_key():
    with mock.patch.object(core, "ROBOFLOW_API_KEY", None):
        yield


def test_load_class_from_path_when_valid_python_module_provided(
    example_module_path: str,
) -> None:
    # when
    my_class = load_class_from_path(
        module_path=example_module_path, class_name="MyClass"
    )
    my_object = my_class()

    # then
    assert my_object.hello() == "HELLO"


def test_load_class_from_path_when_valid_python_module_provided_but_invalid_class_name(
    example_module_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = load_class_from_path(module_path=example_module_path, class_name="Invalid")


def test_load_class_from_path_when_broken_python_module_provided(
    example_broken_module_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = load_class_from_path(
            module_path=example_broken_module_path, class_name="MyClass"
        )


def test_load_class_from_path_when_not_a_valid_python_module_provided(
    example_non_python_file_path: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = load_class_from_path(
            module_path=example_non_python_file_path, class_name="MyClass"
        )


def test_load_class_from_path_when_not_existing_module_path_specified() -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = load_class_from_path(module_path="/not/existing.py", class_name="MyClass")


def test_attempt_loading_model_from_local_storage_when_valid_model_package_provided(
    example_model_package_dir: str,
) -> None:
    # when
    my_model = attempt_loading_model_from_local_storage(
        model_dir_or_weights_path=example_model_package_dir,
        allow_local_code_packages=True,
        model_init_kwargs={"some": "value"},
    )

    # then
    assert my_model.class_names == ["a", "b"]
    prediction = my_model(np.zeros((192, 168, 3), dtype=np.uint8))
    assert isinstance(prediction, ClassificationPrediction)


def test_attempt_loading_model_from_local_storage_when_local_packages_forbidden(
    example_model_package_dir: str,
) -> None:
    # when
    with pytest.raises(ModelLoadingError):
        _ = attempt_loading_model_from_local_storage(
            model_dir_or_weights_path=example_model_package_dir,
            allow_local_code_packages=False,
            model_init_kwargs={"some": "value"},
        )


def test_parse_model_config_when_invalid_path_provided() -> None:
    # when
    with pytest.raises(ModelLoadingError):
        _ = parse_model_config(config_path="/some/invalid/path.json")


def test_parse_model_config_when_not_a_json_file_path_provided(
    not_a_json_file_config_path: str,
) -> None:
    # when
    with pytest.raises(ModelLoadingError):
        _ = parse_model_config(config_path=not_a_json_file_config_path)


def test_parse_model_config_when_config_file_does_not_contain_dict(
    not_a_dict_inside_config_path: str,
) -> None:
    # when
    with pytest.raises(ModelLoadingError):
        _ = parse_model_config(config_path=not_a_dict_inside_config_path)


def test_parse_model_config_when_config_file_defines_unknown_backend(
    unknown_backend_config_path: str,
) -> None:
    # when
    with pytest.raises(ModelLoadingError):
        _ = parse_model_config(config_path=unknown_backend_config_path)


@pytest.mark.parametrize("backend_type", [None, [], {}, True])
def test_parse_model_config_wraps_malformed_backend_type(
    empty_local_dir: str,
    backend_type,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    with open(config_path, "w") as file_handle:
        json.dump(
            {
                "model_architecture": "yolov8",
                "task_type": "object-detection",
                "backend_type": backend_type,
            },
            file_handle,
        )

    with pytest.raises(CorruptedModelPackageError) as error:
        parse_model_config(config_path=config_path)
    assert f"({backend_type})" in str(error.value)


def test_parse_model_config_when_full_config_provided(full_config_path: str) -> None:
    # when
    result = parse_model_config(config_path=full_config_path)

    # then
    with open(full_config_path) as config_file:
        expected_manifest_hash = core.hash_dict_content(content=json.load(config_file))
    assert result == InferenceModelConfig(
        model_architecture="some",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        model_module="model.py",
        model_class="MyClassificationModel",
        manifest_content_hash=expected_manifest_hash,
    )


@mock.patch.object(model_cache_paths, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path() -> None:
    # when
    result = generate_model_package_cache_path(
        model_id="my-model", package_id="mypackage"
    )

    # then
    assert (
        result
        == "/some/models-cache/v2-my-model-83fafb8ac0ed88eaaebd92414d983ae1/mypackage"
    )


@mock.patch.object(model_cache_paths, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_at_front() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="/my-model", package_id="mypackage"
    )

    # then
    assert (
        result
        == "/some/models-cache/v2--my-model-c3716d2ea11a9ad5e6941a321157674f/mypackage"
    )


@mock.patch.object(model_cache_paths, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_in_the_middle() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="my-model/../../home", package_id="mypackage"
    )

    # then
    assert (
        result
        == "/some/models-cache/v2-my-model-home-162436562dcaf7bcb23c30148c5a4e64/mypackage"
    )


@mock.patch.object(model_cache_paths, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_package_id_is_not_sanitized() -> None:
    # when
    with pytest.raises(InsecureModelIdentifierError):
        _ = generate_model_package_cache_path(
            model_id="my-model", package_id="/my-package"
        )


def test_dump_auto_resolution_cache_when_cache_disabled() -> None:
    # given
    auto_resolution_cache = MagicMock()

    # when
    dump_auto_resolution_cache(
        use_auto_resolution_cache=False,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash="my-hash",
        model_id="my-model",
        cache_model_id="my-model",
        canonical_model_id="my-model",
        model_package_id="my-package",
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features=None,
    )

    # then
    auto_resolution_cache.assert_not_called()


@mock.patch.object(core, "datetime")
def test_dump_auto_resolution_cache_when_cache_enabled(
    datetime_mock: MagicMock,
) -> None:
    # given
    now = datetime.now()
    auto_resolution_cache = MagicMock()
    datetime_mock.now.return_value = now

    # when
    dump_auto_resolution_cache(
        use_auto_resolution_cache=True,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash="my-hash",
        model_id="my-model",
        cache_model_id="my-model",
        canonical_model_id="my-model",
        model_package_id="my-package",
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features={"some": "value"},
        trusted_source=True,
        offline_compatibility_hash="c" * 64,
        package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
    )

    # then
    auto_resolution_cache.register.assert_called_once_with(
        auto_negotiation_hash="my-hash",
        cache_entry=AutoResolutionCacheEntry(
            model_id="my-model",
            cache_model_id="my-model",
            canonical_model_id="my-model",
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            credential_hash=core._credential_hash(api_key=None),
            model_package_id="my-package",
            resolved_files={"some/file.txt"},
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=now,
            model_features={"some": "value"},
            trusted_source=True,
            offline_compatibility_hash="c" * 64,
            package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
        ),
    )


def test_dump_auto_resolution_cache_preserves_legacy_positional_slots() -> None:
    auto_resolution_cache = MagicMock()
    recommended_parameters = RecommendedParameters(confidence=0.42)

    dump_auto_resolution_cache(
        True,
        auto_resolution_cache,
        "my-hash",
        "requested-model",
        "my-package",
        "yolov8",
        "object-detection",
        BackendType.ONNX,
        {"some/file.txt"},
        None,
        None,
        recommended_parameters,
        "canonical-model",
        canonical_model_id="canonical-model",
        package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
    )

    registered_entry = auto_resolution_cache.register.call_args.kwargs["cache_entry"]
    assert registered_entry.recommended_parameters == recommended_parameters
    assert registered_entry.cache_model_id == "canonical-model"
    assert registered_entry.trusted_source is None
    assert registered_entry.offline_compatibility_hash is None


@pytest.mark.parametrize(
    ("model_id", "cache_model_id", "canonical_model_id"),
    [
        (None, "canonical-model", "canonical-model"),
        ("", "canonical-model", "canonical-model"),
        ("requested-model", None, "canonical-model"),
        ("requested-model", "", "canonical-model"),
        ("requested-model", "canonical-model", None),
        ("requested-model", "canonical-model", ""),
    ],
)
def test_dump_auto_resolution_cache_rejects_incomplete_attribution(
    model_id: Optional[str],
    cache_model_id: Optional[str],
    canonical_model_id: Optional[str],
) -> None:
    auto_resolution_cache = MagicMock()

    with pytest.raises(ModelLoadingError, match="Cannot cache model resolution"):
        dump_auto_resolution_cache(
            use_auto_resolution_cache=True,
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="my-hash",
            model_id=model_id,
            cache_model_id=cache_model_id,
            canonical_model_id=canonical_model_id,
            model_package_id="my-package",
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            resolved_files={"some/file.txt"},
            model_dependencies=None,
            model_features=None,
        )

    auto_resolution_cache.register.assert_not_called()


@mock.patch.object(core, "datetime")
def test_dump_auto_resolution_cache_persists_cache_model_id(
    datetime_mock: MagicMock,
) -> None:
    # For locally-discovered packages resolved under an alias, the cache entry
    # must record the model id whose on-disk cache holds the package so cache
    # hits rebuild the correct directory.
    now = datetime.now()
    auto_resolution_cache = MagicMock()
    datetime_mock.now.return_value = now

    dump_auto_resolution_cache(
        use_auto_resolution_cache=True,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash="my-hash",
        model_id="rfdetr-nano",
        cache_model_id="workspace/coco-38",
        canonical_model_id="workspace/coco-38",
        model_package_id="localtrtabc123",
        model_architecture="rfdetr",
        task_type="object-detection",
        backend_type=BackendType.TRT,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features=None,
        package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
    )

    registered_entry = auto_resolution_cache.register.call_args.kwargs["cache_entry"]
    assert registered_entry.model_id == "rfdetr-nano"
    assert registered_entry.cache_model_id == "workspace/coco-38"
    assert registered_entry.canonical_model_id == "workspace/coco-38"
    assert registered_entry.cache_attribution_version == core.CACHE_ATTRIBUTION_VERSION


@mock.patch.object(core, "datetime")
def test_dump_auto_resolution_cache_persists_recommended_parameters(
    datetime_mock: MagicMock,
) -> None:
    # When recommended_parameters is provided at load time, the cache entry must
    # store it so subsequent cache hits get the same value without re-fetching
    # model metadata from the weights provider.

    now = datetime.now()
    auto_resolution_cache = MagicMock()
    datetime_mock.now.return_value = now
    recommended_parameters = RecommendedParameters(confidence=0.42)

    dump_auto_resolution_cache(
        use_auto_resolution_cache=True,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash="my-hash",
        model_id="my-model",
        cache_model_id="my-model",
        canonical_model_id="my-model",
        model_package_id="my-package",
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features=None,
        recommended_parameters=recommended_parameters,
        package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
    )

    auto_resolution_cache.register.assert_called_once_with(
        auto_negotiation_hash="my-hash",
        cache_entry=AutoResolutionCacheEntry(
            model_id="my-model",
            cache_model_id="my-model",
            canonical_model_id="my-model",
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            credential_hash=core._credential_hash(api_key=None),
            model_package_id="my-package",
            resolved_files={"some/file.txt"},
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=now,
            model_features=None,
            recommended_parameters=recommended_parameters,
            package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
        ),
    )


@mock.patch.object(core, "datetime")
def test_dump_auto_resolution_cache_omits_recommended_parameters_when_none(
    datetime_mock: MagicMock,
) -> None:
    # The default for the kwarg is None, and that should round-trip cleanly through
    # the cache entry. This guards the backward-compat path: model loads that don't
    # have recommended_parameters work exactly as before.
    now = datetime.now()
    auto_resolution_cache = MagicMock()
    datetime_mock.now.return_value = now

    dump_auto_resolution_cache(
        use_auto_resolution_cache=True,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash="my-hash",
        model_id="my-model",
        cache_model_id="my-model",
        canonical_model_id="my-model",
        model_package_id="my-package",
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features=None,
        package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
    )

    auto_resolution_cache.register.assert_called_once_with(
        auto_negotiation_hash="my-hash",
        cache_entry=AutoResolutionCacheEntry(
            model_id="my-model",
            cache_model_id="my-model",
            canonical_model_id="my-model",
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            credential_hash=core._credential_hash(api_key=None),
            model_package_id="my-package",
            resolved_files={"some/file.txt"},
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=now,
            model_features=None,
            recommended_parameters=None,
            package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
        ),
    )


def test_resolve_recommended_parameters_package_overrides_model() -> None:
    package_params = RecommendedParameters(confidence=0.8)
    model_params = RecommendedParameters(confidence=0.4)
    assert (
        resolve_recommended_parameters(package_params, model_params) is package_params
    )


def test_resolve_recommended_parameters_falls_back_to_model() -> None:
    model_params = RecommendedParameters(confidence=0.4)
    assert resolve_recommended_parameters(None, model_params) is model_params


def test_resolve_recommended_parameters_none_when_both_absent() -> None:
    assert resolve_recommended_parameters(None, None) is None


def test_dump_model_config_for_offline_use_when_file_exists(
    empty_local_dir: str,
) -> None:
    """Existing package metadata is upgraded without reporting a new file."""
    # given
    config_path = os.path.join(empty_local_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_architecture": "yolov8",
                "task_type": "object-detection",
                "backend_type": BackendType.ONNX,
            },
            f,
        )
    on_file_created = MagicMock()

    # then
    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        model_id="workspace/project/3",
        canonical_model_id="workspace/project/3",
    )

    # then
    on_file_created.assert_not_called()
    with open(config_path) as file:
        decoded = json.load(file)
    assert decoded["offline_manifest_version"] == core.OFFLINE_CACHE_MANIFEST_VERSION
    assert decoded["model_architecture"] == "yolov8"
    assert decoded["task_type"] == "object-detection"
    assert decoded["backend_type"] == "onnx"
    assert decoded["model_id"] == "workspace/project/3"
    assert decoded["canonical_model_id"] == "workspace/project/3"


def test_dump_model_config_for_offline_use_upgrades_legacy_config(
    empty_local_dir: str,
) -> None:
    """A legacy config is upgraded to a complete current manifest in place."""
    # given
    config_path = os.path.join(empty_local_dir, "model_config.json")
    existing_config = {
        "model_architecture": "yolov8",
        "task_type": "object-detection",
        "backend_type": "onnx",
    }
    with open(config_path, "w") as file:
        json.dump(existing_config, file)
    on_file_created = MagicMock()

    # when
    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=10,
        model_id="workspace/project/3",
        on_file_created=on_file_created,
        model_features={"nms_fused": {"max_detections": 100}},
        trusted_source=True,
        model_dependencies=[],
        recommended_parameters={"confidence": 0.4},
        quantization="fp32",
        dynamic_batch_size_supported=False,
        static_batch_size=1,
        runtime_compatibility_hash="a" * 64,
        offline_compatibility_hash="b" * 64,
        canonical_model_id="workspace/project/3",
    )

    # then
    with open(config_path) as file:
        decoded = json.load(file)
    assert decoded == {
        "offline_manifest_version": core.OFFLINE_CACHE_MANIFEST_VERSION,
        "model_architecture": "yolov8",
        "task_type": "object-detection",
        "backend_type": "onnx",
        "model_id": "workspace/project/3",
        "model_features": {"nms_fused": {"max_detections": 100}},
        "trusted_source": True,
        "model_dependencies": [],
        "recommended_parameters": {"confidence": 0.4},
        "quantization": "fp32",
        "dynamic_batch_size_supported": False,
        "static_batch_size": 1,
        "runtime_compatibility_hash": "a" * 64,
        "offline_compatibility_hash": "b" * 64,
        "canonical_model_id": "workspace/project/3",
        "package_artifacts": [],
        "dependency_package_paths": [],
    }
    on_file_created.assert_not_called()


def test_dump_model_config_refuses_same_owner_provenance_change(
    empty_local_dir: str,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    runtime_hash = "a" * 64
    original_hash = dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=10,
        model_id="workspace/project/3",
        canonical_model_id="workspace/project/3",
        trusted_source=True,
        model_dependencies=[],
        runtime_compatibility_hash=runtime_hash,
    )
    with open(config_path, "rb") as config_file:
        original_bytes = config_file.read()

    with pytest.raises(CorruptedModelPackageError, match="different provenance"):
        dump_model_config_for_offline_use(
            config_path=config_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=10,
            model_id="workspace/project/3",
            canonical_model_id="workspace/project/3",
            trusted_source=False,
            model_dependencies=[],
            runtime_compatibility_hash=runtime_hash,
        )

    with open(config_path, "rb") as config_file:
        assert config_file.read() == original_bytes
    assert parse_model_config(config_path).manifest_content_hash == original_hash


def test_dump_model_config_preserves_manifest_for_alias_request_hash(
    empty_local_dir: str,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    original_hash = dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=10,
        model_id="workspace/canonical/3",
        canonical_model_id="workspace/canonical/3",
        trusted_source=True,
        model_dependencies=[],
        runtime_compatibility_hash="a" * 64,
        offline_compatibility_hash="b" * 64,
    )
    with open(config_path, "rb") as config_file:
        original_bytes = config_file.read()
    original_stat = os.stat(config_path)

    with mock.patch.object(
        core.os,
        "replace",
        side_effect=AssertionError("an immutable manifest must not be replaced"),
    ) as replace_mock:
        alias_hash = dump_model_config_for_offline_use(
            config_path=config_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=10,
            model_id="workspace/canonical/3",
            canonical_model_id="workspace/canonical/3",
            trusted_source=True,
            model_dependencies=[],
            runtime_compatibility_hash="a" * 64,
            offline_compatibility_hash="c" * 64,
        )

    with open(config_path, "rb") as config_file:
        assert config_file.read() == original_bytes
    alias_stat = os.stat(config_path)
    assert alias_stat.st_ino == original_stat.st_ino
    assert alias_stat.st_mtime_ns == original_stat.st_mtime_ns
    replace_mock.assert_not_called()
    assert alias_hash == original_hash
    assert parse_model_config(config_path).offline_compatibility_hash == "b" * 64


def test_dump_model_config_for_offline_use_rejects_conflicting_existing_model_id(
    empty_local_dir: str,
) -> None:
    """A current manifest must never bless a package owned by another model."""
    # given
    config_path = os.path.join(empty_local_dir, "model_config.json")
    existing_config = {
        "model_architecture": "yolov8",
        "task_type": "object-detection",
        "backend_type": "onnx",
        "model_id": "workspace/canonical-project/3",
    }
    with open(config_path, "w") as file:
        json.dump(existing_config, file)

    # when
    with pytest.raises(CorruptedModelPackageError):
        dump_model_config_for_offline_use(
            config_path=config_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=10,
            model_id="workspace/alias/3",
            canonical_model_id="workspace/alias/3",
        )

    # then
    with open(config_path) as file:
        decoded = json.load(file)
    assert decoded["model_id"] == "workspace/canonical-project/3"
    assert "offline_manifest_version" not in decoded
    assert "canonical_model_id" not in decoded


def test_dump_model_config_rejects_conflicting_existing_canonical_id(
    empty_local_dir: str,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    existing_config = {
        "model_id": "cache-owner/1",
        "canonical_model_id": "tenant-a/canonical/1",
    }
    with open(config_path, "w") as file:
        json.dump(existing_config, file)

    with pytest.raises(CorruptedModelPackageError):
        dump_model_config_for_offline_use(
            config_path=config_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=10,
            model_id="cache-owner/1",
            canonical_model_id="tenant-b/canonical/1",
        )

    with open(config_path) as file:
        assert json.load(file) == existing_config


@pytest.mark.parametrize(
    "existing_model_id",
    [None, "", [], {}, 7, False],
)
def test_dump_model_config_rejects_malformed_existing_model_id(
    empty_local_dir: str,
    existing_model_id: object,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    with open(config_path, "w") as file:
        json.dump(
            {
                "model_architecture": "yolov8",
                "task_type": "object-detection",
                "backend_type": "onnx",
                "model_id": existing_model_id,
            },
            file,
        )

    with pytest.raises(CorruptedModelPackageError):
        dump_model_config_for_offline_use(
            config_path=config_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=10,
            model_id="workspace/project/3",
            canonical_model_id="workspace/project/3",
        )

    with open(config_path) as file:
        assert json.load(file)["model_id"] == existing_model_id


@pytest.mark.parametrize(
    "model_id, canonical_model_id",
    [
        (None, "workspace/project/3"),
        ("", "workspace/project/3"),
        ("   ", "workspace/project/3"),
        ("workspace/project/3", None),
        ("workspace/project/3", ""),
        ("workspace/project/3", "   "),
    ],
)
def test_dump_model_config_rejects_incomplete_new_attribution(
    empty_local_dir: str,
    model_id: Optional[str],
    canonical_model_id: Optional[str],
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")

    with pytest.raises(CorruptedModelPackageError):
        dump_model_config_for_offline_use(
            config_path=config_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=10,
            model_id=model_id,
            canonical_model_id=canonical_model_id,
        )

    assert not os.path.exists(config_path)


@pytest.mark.parametrize(
    "existing_canonical_model_id",
    [None, "", [], {}, 7, False],
)
def test_dump_model_config_rejects_malformed_existing_canonical_id(
    empty_local_dir: str,
    existing_canonical_model_id: object,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    existing_config = {
        "model_id": "workspace/project/3",
        "canonical_model_id": existing_canonical_model_id,
    }
    with open(config_path, "w") as file:
        json.dump(existing_config, file)

    with pytest.raises(CorruptedModelPackageError):
        dump_model_config_for_offline_use(
            config_path=config_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=10,
            model_id="workspace/project/3",
            canonical_model_id="workspace/project/3",
        )

    with open(config_path) as file:
        assert json.load(file) == existing_config


def test_dump_model_config_for_offline_use_when_file_does_not_exists(
    empty_local_dir: str,
) -> None:
    """A new package config is written and reported through the callback."""
    # given
    config_path = os.path.join(empty_local_dir, "model_config.json")
    on_file_created = MagicMock()

    # then
    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=10,
        on_file_created=on_file_created,
        model_id="workspace/project/3",
        canonical_model_id="workspace/project/3",
    )

    # then
    on_file_created.assert_called_once_with(config_path)
    with open(config_path) as f:
        decoded = json.load(f)
    assert decoded == {
        "offline_manifest_version": core.OFFLINE_CACHE_MANIFEST_VERSION,
        "model_architecture": "yolov8",
        "task_type": "object-detection",
        "backend_type": "onnx",
        "model_features": None,
        "trusted_source": None,
        "model_dependencies": None,
        "recommended_parameters": None,
        "quantization": None,
        "dynamic_batch_size_supported": None,
        "static_batch_size": None,
        "runtime_compatibility_hash": None,
        "offline_compatibility_hash": None,
        "model_id": "workspace/project/3",
        "canonical_model_id": "workspace/project/3",
        "package_artifacts": [],
        "dependency_package_paths": [],
    }


@pytest.mark.parametrize("corrupt_content", ["not-json", "[]"])
def test_dump_model_config_for_offline_use_rejects_corrupt_existing_config(
    empty_local_dir: str,
    corrupt_content: str,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    _create_file(config_path, corrupt_content)
    on_file_created = MagicMock()

    with pytest.raises(CorruptedModelPackageError):
        dump_model_config_for_offline_use(
            config_path,
            "yolov8",
            "object-detection",
            BackendType.ONNX,
            10,
            model_id="workspace/project/3",
            on_file_created=on_file_created,
            trusted_source=True,
            model_dependencies=[],
            quantization="fp32",
            runtime_compatibility_hash="a" * 64,
            offline_compatibility_hash="b" * 64,
            canonical_model_id="workspace/project/3",
        )

    with open(config_path) as file:
        assert file.read() == corrupt_content
    on_file_created.assert_not_called()


def test_dump_model_config_for_offline_use_preserves_legacy_positional_slots(
    empty_local_dir: str,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    on_file_created = MagicMock()

    dump_model_config_for_offline_use(
        config_path,
        "yolov8",
        "object-detection",
        BackendType.ONNX,
        10,
        on_file_created,
        "workspace/project/3",
        canonical_model_id="workspace/project/3",
    )

    with open(config_path) as file:
        assert json.load(file)["model_id"] == "workspace/project/3"
    on_file_created.assert_called_once_with(config_path)


def test_new_offline_parameters_are_appended_to_existing_helper_signatures() -> None:
    assert list(inspect.signature(dump_model_config_for_offline_use).parameters)[
        :7
    ] == [
        "config_path",
        "model_architecture",
        "task_type",
        "backend_type",
        "file_lock_acquire_timeout",
        "on_file_created",
        "model_id",
    ]
    assert list(inspect.signature(initialize_model).parameters)[-2:] == [
        "offline_compatibility_hash",
        "api_key",
    ]
    assert (
        list(inspect.signature(attempt_loading_matching_model_packages).parameters)[-1]
        == "offline_compatibility_hash"
    )
    assert list(inspect.signature(dump_auto_resolution_cache).parameters)[-3:] == [
        "canonical_model_id",
        "package_manifest_hash",
        "api_key",
    ]


@pytest.mark.parametrize(
    ("requested_model_id", "cache_model_id", "expected_message"),
    [
        (" ", None, "requested model identity"),
        (123, None, "requested model identity"),
        (None, " ", "cache owner identity"),
        (None, 123, "cache owner identity"),
    ],
)
def test_initialize_model_rejects_invalid_attribution_before_path_work(
    requested_model_id: object,
    cache_model_id: object,
    expected_message: str,
) -> None:
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[],
        trusted_source=True,
        cache_model_id=cache_model_id,
    )
    auto_resolution_cache = MagicMock()

    with mock.patch.object(core, "resolve_model_class") as resolve_model_class:
        with pytest.raises(CorruptedModelPackageError, match=expected_message):
            initialize_model(
                model_id="tenant/canonical/1",
                requested_model_id=requested_model_id,
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

    resolve_model_class.assert_not_called()
    auto_resolution_cache.register.assert_not_called()


@pytest.mark.parametrize(
    "unsafe_handle",
    [
        "",
        "../weights.onnx",
        "/weights.onnx",
        "nested//weights.onnx",
        "nested/../weights.onnx",
        "nested/CON",
        "nested/.weights.onnx.lock",
        r"nested\weights.onnx",
        "MODEL_CONFIG.JSON",
        "model_dependencies/weights.onnx",
        "dependencies",
        ".model_config.json.lock",
        ".weights.onnx.lock",
        "CON",
        "NUL.bin",
        "x:stream",
        "weights.",
        "weïghts.onnx",
        "weights\x01.onnx",
    ],
)
def test_initialize_model_rejects_unsafe_artifact_handles_before_path_work(
    unsafe_handle: str,
) -> None:
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/weights",
                file_handle=unsafe_handle,
                md5_hash="a" * 32,
            )
        ],
        trusted_source=True,
    )

    with mock.patch.object(
        core, "generate_model_package_cache_path"
    ) as generate_path, mock.patch.object(core, "resolve_model_class") as resolve_class:
        with pytest.raises(CorruptedModelPackageError):
            initialize_model(
                model_id="workspace/project/1",
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

    generate_path.assert_not_called()
    resolve_class.assert_not_called()


def test_initialize_model_rejects_case_ambiguous_artifact_handles_before_path_work() -> (
    None
):
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/weights-a",
                file_handle="weights.onnx",
                md5_hash="a" * 32,
            ),
            FileDownloadSpecs(
                download_url="https://example.com/weights-b",
                file_handle="WEIGHTS.ONNX",
                md5_hash="b" * 32,
            ),
        ],
        trusted_source=True,
    )

    with mock.patch.object(core, "generate_model_package_cache_path") as generate_path:
        with pytest.raises(CorruptedModelPackageError, match="case-ambiguous"):
            initialize_model(
                model_id="workspace/project/1",
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

    generate_path.assert_not_called()


@pytest.mark.parametrize(
    "file_handles",
    [
        ["base", "base/config.json"],
        ["base/config.json", "base"],
        ["Base/config.json", "base/model.safetensors"],
    ],
)
def test_initialize_model_rejects_artifact_path_prefix_and_segment_collisions(
    file_handles: list,
) -> None:
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"https://example.com/{index}",
                file_handle=file_handle,
                md5_hash=f"{index + 1:032x}",
            )
            for index, file_handle in enumerate(file_handles)
        ],
        trusted_source=True,
    )

    with mock.patch.object(core, "generate_model_package_cache_path") as generate_path:
        with pytest.raises(CorruptedModelPackageError):
            initialize_model(
                model_id="workspace/project/1",
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

    generate_path.assert_not_called()


def test_initialize_and_raw_offline_load_support_nested_artifact_handles(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/vit/1"
    package_id = "nestedPackage"
    artifact_contents = {
        "vit/config.json": b'{"model_type": "vit"}',
        "vit/model.safetensors": b"nested model weights",
    }
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.TORCH,
        package_artefacts=[
            FileDownloadSpecs(
                download_url=f"https://example.com/{file_handle}",
                file_handle=file_handle,
                md5_hash=hashlib.md5(content).hexdigest(),
            )
            for file_handle, content in artifact_contents.items()
        ],
        trusted_source=True,
    )
    model_class = MagicMock()
    model_class.from_pretrained.side_effect = [MagicMock(), MagicMock()]

    def download_files(**kwargs):
        if kwargs.get("name_after") != "md5_hash":
            return {}
        os.makedirs(kwargs["target_dir"], exist_ok=True)
        result = {}
        for file_handle, _, md5_hash in kwargs["files_specs"]:
            blob_path = os.path.join(kwargs["target_dir"], md5_hash)
            with open(blob_path, "wb") as blob_file:
                blob_file.write(artifact_contents[file_handle])
            result[file_handle] = blob_path
        return result

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", side_effect=download_files
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ), mock.patch.object(
        core, "model_implementation_exists", return_value=True
    ):
        _, package_dir = initialize_model(
            model_id=model_id,
            model_architecture="vit",
            task_type="classification",
            model_package=package,
            model_init_kwargs={},
            auto_resolution_cache=MagicMock(),
            auto_negotiation_hash="a" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
        )
        cached_dir = find_cached_model_package_dir(model_id=model_id)
        offline_result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )

    assert cached_dir == package_dir
    assert offline_result is not None
    assert model_class.from_pretrained.call_count == 2
    assert os.path.islink(os.path.join(package_dir, "vit/config.json"))
    assert os.path.islink(os.path.join(package_dir, "vit/model.safetensors"))


def test_initialize_model_rejects_unhashed_remote_artifact_before_existing_file_skip() -> (
    None
):
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/weights",
                file_handle="weights.onnx",
                md5_hash=None,
            )
        ],
        trusted_source=True,
    )

    with mock.patch.object(
        core, "generate_model_package_cache_path"
    ) as generate_path, mock.patch.object(
        core, "download_files_to_directory"
    ) as download:
        with pytest.raises(UntrustedFileError):
            initialize_model(
                model_id="workspace/project/1",
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
                download_files_without_hash=False,
            )

    generate_path.assert_not_called()
    download.assert_not_called()


def test_unhashed_download_source_identity_preserves_content_selecting_queries() -> (
    None
):
    first = core._package_artifact_declarations(
        [
            FileDownloadSpecs(
                download_url="https://host/model?v=1",
                file_handle="weights.bin",
                md5_hash=None,
            )
        ]
    )
    second = core._package_artifact_declarations(
        [
            FileDownloadSpecs(
                download_url="https://host/model?v=2",
                file_handle="weights.bin",
                md5_hash=None,
            )
        ]
    )

    assert first[0]["source_hash"] != second[0]["source_hash"]


def test_unhashed_download_source_identity_ignores_only_rotating_signed_auth() -> None:
    first = core._package_artifact_declarations(
        [
            FileDownloadSpecs(
                download_url=(
                    "https://Storage.Example/model?"
                    "versionId=7&X-Amz-Date=20260101T000000Z&"
                    "X-Amz-Signature=first"
                ),
                file_handle="weights.bin",
                md5_hash=None,
            )
        ]
    )
    second = core._package_artifact_declarations(
        [
            FileDownloadSpecs(
                download_url=(
                    "https://Storage.Example/model?"
                    "X-Amz-Signature=second&versionId=7&"
                    "X-Amz-Date=20260701T000000Z"
                ),
                file_handle="weights.bin",
                md5_hash=None,
            )
        ]
    )
    changed_content_selector = core._package_artifact_declarations(
        [
            FileDownloadSpecs(
                download_url=(
                    "https://Storage.Example/model?" "X-Amz-Signature=third&versionId=8"
                ),
                file_handle="weights.bin",
                md5_hash=None,
            )
        ]
    )

    assert first[0]["source_hash"] == second[0]["source_hash"]
    assert first[0]["source_hash"] != changed_content_selector[0]["source_hash"]


def test_initialize_model_rejects_dependency_lock_name_collision_before_path_work(
    empty_local_dir: str,
) -> None:
    dependency = ModelDependency(
        name=".encoder.lock",
        model_id="workspace/encoder/1",
    )
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[],
        trusted_source=True,
    )

    with mock.patch.object(core, "generate_model_package_cache_path") as generate_path:
        with pytest.raises(CorruptedModelPackageError, match="dependency lock path"):
            initialize_model(
                model_id="workspace/project/1",
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[dependency],
                model_dependencies_instances={"encoder": MagicMock()},
                model_dependencies_directories={".encoder.lock": empty_local_dir},
            )

    generate_path.assert_not_called()


def test_initialize_model_does_not_publish_offline_manifest_before_success(
    empty_local_dir: str,
) -> None:
    package_dir = os.path.join(empty_local_dir, "package")
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[],
        trusted_source=True,
    )
    model_class = MagicMock()
    model_class.from_pretrained.side_effect = RuntimeError("initialization failed")

    with mock.patch.object(
        core,
        "generate_model_package_cache_path",
        return_value=package_dir,
    ), mock.patch.object(
        core, "download_files_to_directory", return_value={}
    ), mock.patch.object(
        core, "create_symlinks_to_shared_blobs", return_value={}
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ), mock.patch.object(
        core, "dump_model_config_for_offline_use"
    ) as dump_manifest:
        with pytest.raises(RuntimeError, match="initialization failed"):
            initialize_model(
                model_id="workspace/project/1",
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

    dump_manifest.assert_not_called()
    assert not os.path.exists(os.path.join(package_dir, "model_config.json"))


def test_failed_unhashed_warm_cannot_launder_stale_same_handle_on_retry(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    package_id = "package"
    file_handle = "adapter_config.json"
    package_a = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/revision-a/adapter_config.json",
                file_handle=file_handle,
                md5_hash=None,
            )
        ],
        trusted_source=True,
    )
    package_b = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/revision-b/adapter_config.json",
                file_handle=file_handle,
                md5_hash=None,
            )
        ],
        trusted_source=True,
    )
    content_by_url = {
        package_a.package_artefacts[0].download_url: b"revision-a",
        package_b.package_artefacts[0].download_url: b"revision-b",
    }

    def download_files(**kwargs):
        if kwargs.get("name_after") == "md5_hash":
            return {}
        os.makedirs(kwargs["target_dir"], exist_ok=True)
        result = {}
        for handle, download_url, _ in kwargs["files_specs"]:
            target_path = os.path.join(kwargs["target_dir"], handle)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if not os.path.exists(target_path):
                with open(target_path, "wb") as target_file:
                    target_file.write(content_by_url[download_url])
            result[handle] = target_path
        return result

    model_class = MagicMock()
    model_class.from_pretrained.side_effect = [
        RuntimeError("first initialization failed"),
        MagicMock(),
    ]
    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", side_effect=download_files
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ):
        with pytest.raises(RuntimeError, match="first initialization failed"):
            initialize_model(
                model_id=model_id,
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package_a,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
                download_files_without_hash=True,
            )
        _, package_dir = initialize_model(
            model_id=model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=package_b,
            model_init_kwargs={},
            auto_resolution_cache=MagicMock(),
            auto_negotiation_hash="b" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
            download_files_without_hash=True,
        )

    with open(os.path.join(package_dir, file_handle), "rb") as artifact_file:
        assert artifact_file.read() == b"revision-b"
    manifest = parse_model_config(
        config_path=os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
    )
    assert (
        manifest.package_artifacts[0]["sha256_hash"]
        == hashlib.sha256(b"revision-b").hexdigest()
    )


def test_concurrent_package_revisions_cannot_interleave_materialization(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    file_handle = "weights.bin"

    def package(revision: str) -> ModelPackageMetadata:
        return ModelPackageMetadata(
            package_id="sharedPackage",
            backend=BackendType.ONNX,
            package_artefacts=[
                FileDownloadSpecs(
                    download_url=f"https://example.com/{revision}/weights.bin",
                    file_handle=file_handle,
                    md5_hash=None,
                )
            ],
            trusted_source=True,
        )

    first_constructor_entered = Event()
    allow_first_constructor_to_finish = Event()
    second_download_started = Event()

    def download_files(**kwargs):
        if kwargs.get("name_after") == "md5_hash":
            return {}
        handle, download_url, _ = kwargs["files_specs"][0]
        if "/revision-b/" in download_url:
            second_download_started.set()
        target_path = os.path.join(kwargs["target_dir"], handle)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if not os.path.exists(target_path):
            with open(target_path, "wb") as target_file:
                target_file.write(
                    b"revision-b" if "/revision-b/" in download_url else b"revision-a"
                )
        return {handle: target_path}

    class BlockingModel:
        @classmethod
        def from_pretrained(cls, package_dir, **kwargs):
            with open(os.path.join(package_dir, file_handle), "rb") as weights:
                if weights.read() == b"revision-a":
                    first_constructor_entered.set()
                    assert allow_first_constructor_to_finish.wait(timeout=5)
            return cls()

    def initialize(model_package: ModelPackageMetadata, cache_hash: str):
        return initialize_model(
            model_id=model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=model_package,
            model_init_kwargs={},
            auto_resolution_cache=MagicMock(),
            auto_negotiation_hash=cache_hash,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
            download_files_without_hash=True,
        )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", side_effect=download_files
    ), mock.patch.object(
        core, "resolve_model_class", return_value=BlockingModel
    ), ThreadPoolExecutor(
        max_workers=2
    ) as executor:
        first_warm = executor.submit(initialize, package("revision-a"), "a" * 64)
        assert first_constructor_entered.wait(timeout=5)
        second_warm = executor.submit(initialize, package("revision-b"), "b" * 64)
        assert not second_download_started.wait(timeout=0.1)
        allow_first_constructor_to_finish.set()
        _, package_dir = first_warm.result(timeout=5)
        with pytest.raises(CorruptedModelPackageError, match="incoming provenance"):
            second_warm.result(timeout=5)

    with open(os.path.join(package_dir, file_handle), "rb") as weights:
        assert weights.read() == b"revision-a"
    manifest = parse_model_config(
        config_path=os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
    )
    assert manifest.package_artifacts[0]["source_hash"] == (
        core._package_artifact_declarations(package("revision-a").package_artefacts)[0][
            "source_hash"
        ]
    )


def test_initialize_model_rejects_constructor_mutating_declared_artifact(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/canonical/1"
    package_id = "localPackage"
    original_content = b'{"image_processor_type": "wrong"}'
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        with open(
            os.path.join(package_dir, MODEL_CONFIG_FILE_NAME), "w"
        ) as manifest_file:
            json.dump({"model_id": model_id}, manifest_file)
        artifact_path = os.path.join(package_dir, "preprocessor_config.json")
        with open(artifact_path, "wb") as artifact_file:
            artifact_file.write(original_content)

        package = ModelPackageMetadata(
            package_id=package_id,
            backend=BackendType.HF,
            package_artefacts=[
                LocalFileArtefactSpecs(
                    file_handle="preprocessor_config.json",
                    md5_hash=hashlib.md5(original_content).hexdigest(),
                )
            ],
            package_source=PackageSourceType.LOCAL_CACHE,
            trusted_source=True,
            cache_model_id=model_id,
        )
        model_class = MagicMock()

        def mutate_artifact(*args, **kwargs):
            with open(artifact_path, "wb") as artifact_file:
                artifact_file.write(b"constructor mutation")
            return MagicMock()

        model_class.from_pretrained.side_effect = mutate_artifact
        auto_resolution_cache = MagicMock()
        with mock.patch.object(core, "resolve_model_class", return_value=model_class):
            with pytest.raises(
                CorruptedModelPackageError,
                match="MD5 identity|changed while the model was being initialized",
            ):
                initialize_model(
                    model_id=model_id,
                    model_architecture="qwen25vl",
                    task_type="lmm",
                    model_package=package,
                    model_init_kwargs={},
                    auto_resolution_cache=auto_resolution_cache,
                    auto_negotiation_hash="a" * 64,
                    model_dependencies=[],
                    model_dependencies_instances={},
                    model_dependencies_directories={},
                )

    auto_resolution_cache.register.assert_not_called()


def test_initialize_model_cannot_bless_owner_changed_after_prevalidation(
    empty_local_dir: str,
) -> None:
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[],
        trusted_source=True,
    )
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()
    auto_resolution_cache = MagicMock()

    def replace_manifest_after_validation(
        package_dir: str,
        cache_model_id: str,
        canonical_model_id: str,
        **kwargs,
    ) -> None:
        _create_file(
            os.path.join(package_dir, "model_config.json"),
            json.dumps({"model_id": "other-tenant/model/1"}),
        )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core,
        "_validate_existing_cache_package_attribution",
        side_effect=replace_manifest_after_validation,
    ), mock.patch.object(
        core, "download_files_to_directory", return_value={}
    ), mock.patch.object(
        core, "create_symlinks_to_shared_blobs", return_value={}
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ):
        with pytest.raises(CorruptedModelPackageError):
            initialize_model(
                model_id="tenant-a/canonical/1",
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

    auto_resolution_cache.register.assert_not_called()


def test_initialize_model_uses_canonical_identity_for_path_manifest_and_cache(
    empty_local_dir: str,
) -> None:
    requested_model_id = "shared-alias/1"
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[],
        trusted_source=True,
    )
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()
    first_cache = MagicMock()
    second_cache = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", return_value={}
    ), mock.patch.object(
        core, "create_symlinks_to_shared_blobs", return_value={}
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ):
        _, first_dir = initialize_model(
            model_id="tenant-a/canonical/1",
            requested_model_id=requested_model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=package,
            model_init_kwargs={},
            auto_resolution_cache=first_cache,
            auto_negotiation_hash="a" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
        )
        _, second_dir = initialize_model(
            model_id="tenant-b/canonical/1",
            requested_model_id=requested_model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=package,
            model_init_kwargs={},
            auto_resolution_cache=second_cache,
            auto_negotiation_hash="b" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
        )

    assert first_dir != second_dir
    for package_dir, canonical_model_id, cache in (
        (first_dir, "tenant-a/canonical/1", first_cache),
        (second_dir, "tenant-b/canonical/1", second_cache),
    ):
        with open(os.path.join(package_dir, "model_config.json")) as file:
            manifest = json.load(file)
        assert manifest["model_id"] == canonical_model_id
        assert manifest["canonical_model_id"] == canonical_model_id
        assert (
            manifest["offline_manifest_version"] == core.OFFLINE_CACHE_MANIFEST_VERSION
        )
        cache_entry = cache.register.call_args.kwargs["cache_entry"]
        assert cache_entry.model_id == requested_model_id
        assert cache_entry.cache_model_id == canonical_model_id
        assert cache_entry.canonical_model_id == canonical_model_id
        assert cache_entry.cache_attribution_version == core.CACHE_ATTRIBUTION_VERSION


def test_initialize_model_allows_same_bound_package_to_be_rewarmed(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    package_id = "package"
    file_handle = "weights.onnx"
    blob_content = b"stable model bytes"
    md5_hash = hashlib.md5(blob_content).hexdigest()
    shared_blobs_dir = os.path.join(empty_local_dir, "shared-blobs")
    os.makedirs(shared_blobs_dir)
    shared_blob_path = os.path.join(shared_blobs_dir, md5_hash)
    with open(shared_blob_path, "wb") as shared_blob:
        shared_blob.write(blob_content)
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/weights",
                file_handle=file_handle,
                md5_hash=md5_hash,
            )
        ],
        trusted_source=True,
    )
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()
    auto_resolution_cache = MagicMock()

    def resolve_download(**kwargs):
        if kwargs.get("name_after") == "md5_hash":
            return {file_handle: shared_blob_path}
        return {}

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", side_effect=resolve_download
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ):
        first_model, first_dir = initialize_model(
            model_id=model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=package,
            model_init_kwargs={},
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
        )
        second_model, second_dir = initialize_model(
            model_id=model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=package,
            model_init_kwargs={},
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="b" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
        )

    assert first_model is model_class.from_pretrained.return_value
    assert second_model is model_class.from_pretrained.return_value
    assert first_dir == second_dir
    assert os.path.islink(os.path.join(first_dir, file_handle))
    assert os.path.realpath(os.path.join(first_dir, file_handle)) == shared_blob_path
    assert model_class.from_pretrained.call_count == 2
    assert auto_resolution_cache.register.call_count == 2


def test_initialize_model_rejects_same_path_provenance_change_before_repoint(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    file_handle = "weights.onnx"
    trusted_content = b"trusted model bytes"
    replacement_content = b"replacement model bytes"
    trusted_md5 = hashlib.md5(trusted_content).hexdigest()
    replacement_md5 = hashlib.md5(replacement_content).hexdigest()
    shared_blobs_dir = os.path.join(empty_local_dir, "shared-blobs")
    os.makedirs(shared_blobs_dir)
    blob_paths = {}
    for md5_hash, content in (
        (trusted_md5, trusted_content),
        (replacement_md5, replacement_content),
    ):
        blob_path = os.path.join(shared_blobs_dir, md5_hash)
        with open(blob_path, "wb") as shared_blob:
            shared_blob.write(content)
        blob_paths[md5_hash] = blob_path

    def package(md5_hash: str, trusted_source: bool) -> ModelPackageMetadata:
        return ModelPackageMetadata(
            package_id="package",
            backend=BackendType.ONNX,
            package_artefacts=[
                FileDownloadSpecs(
                    download_url="https://example.com/weights",
                    file_handle=file_handle,
                    md5_hash=md5_hash,
                )
            ],
            trusted_source=trusted_source,
        )

    def resolve_download(**kwargs):
        if kwargs.get("name_after") != "md5_hash":
            return {}
        md5_hash = kwargs["files_specs"][0][2]
        return {file_handle: blob_paths[md5_hash]}

    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()
    auto_resolution_cache = MagicMock()
    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", side_effect=resolve_download
    ) as download, mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ):
        _, package_dir = initialize_model(
            model_id=model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=package(trusted_md5, True),
            model_init_kwargs={},
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
        )
        package_link = os.path.join(package_dir, file_handle)
        with pytest.raises(CorruptedModelPackageError, match="incoming provenance"):
            initialize_model(
                model_id=model_id,
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package(replacement_md5, False),
                model_init_kwargs={},
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="b" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

    assert download.call_count == 2
    assert model_class.from_pretrained.call_count == 1
    assert auto_resolution_cache.register.call_count == 1
    assert os.path.realpath(package_link) == blob_paths[trusted_md5]


def test_initialize_model_rejects_changed_unhashed_bytes_before_constructor(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    file_handle = "weights.onnx"
    package = ModelPackageMetadata(
        package_id="package",
        backend=BackendType.ONNX,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/weights",
                file_handle=file_handle,
                md5_hash=None,
            )
        ],
        trusted_source=True,
    )
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()
    auto_resolution_cache = MagicMock()
    unhashed_downloads = iter((b"original bytes", b"changed bytes"))

    def resolve_download(**kwargs):
        if kwargs.get("name_after") == "md5_hash":
            return {}
        package_path = os.path.join(kwargs["target_dir"], file_handle)
        with open(package_path, "wb") as package_file:
            package_file.write(next(unhashed_downloads))
        return {file_handle: package_path}

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", side_effect=resolve_download
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ):
        initialize_model(
            model_id=model_id,
            model_architecture="yolov8",
            task_type="object-detection",
            model_package=package,
            model_init_kwargs={},
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_dependencies=[],
            model_dependencies_instances={},
            model_dependencies_directories={},
            download_files_without_hash=True,
        )
        with pytest.raises(CorruptedModelPackageError, match="incoming provenance"):
            initialize_model(
                model_id=model_id,
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="b" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
                download_files_without_hash=True,
            )

    assert model_class.from_pretrained.call_count == 1
    assert auto_resolution_cache.register.call_count == 1


def test_initialize_model_rejects_regular_dependency_path_before_constructor(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    package_id = "package"
    dependency_dir = os.path.join(empty_local_dir, "dependency-package")
    os.makedirs(dependency_dir)
    dump_model_config_for_offline_use(
        config_path=os.path.join(dependency_dir, "model_config.json"),
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=1,
        model_id="workspace/encoder/1",
        canonical_model_id="workspace/encoder/1",
        trusted_source=True,
        model_dependencies=[],
        runtime_compatibility_hash=core._runtime_compatibility_hash(
            runtime_x_ray=core.x_ray_runtime_environment()
        ),
    )
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(
            os.path.join(
                package_dir,
                core.MODEL_DEPENDENCIES_SUB_DIR,
                "encoder",
            )
        )
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.ONNX,
        package_artefacts=[],
        trusted_source=True,
    )
    model_class = MagicMock()
    auto_resolution_cache = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "download_files_to_directory", return_value={}
    ), mock.patch.object(
        core, "resolve_model_class", return_value=model_class
    ):
        with pytest.raises(
            CorruptedModelPackageError,
            match="undeclared or unsafe|not a symbolic link",
        ):
            initialize_model(
                model_id=model_id,
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_dependencies=[
                    ModelDependency(
                        name="encoder",
                        model_id="workspace/encoder/1",
                    )
                ],
                model_dependencies_instances={"encoder": MagicMock()},
                model_dependencies_directories={"encoder": dependency_dir},
            )

    model_class.from_pretrained.assert_not_called()
    auto_resolution_cache.register.assert_not_called()
    assert not os.path.exists(os.path.join(package_dir, "model_config.json"))


def test_initialize_local_cache_uses_discovered_legacy_package_path(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/canonical/1"
    package_id = "localtrtpackage"
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.TRT,
        package_artefacts=[],
        package_source=PackageSourceType.LOCAL_CACHE,
        trusted_source=False,
        cache_model_id=model_id,
    )
    model_class = MagicMock()
    model_class.from_pretrained.return_value = MagicMock()

    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        legacy_package_dir = model_cache_paths.generate_legacy_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(legacy_package_dir)
        with open(
            os.path.join(legacy_package_dir, "model_config.json"),
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump({"model_id": model_id}, config_file)
        with mock.patch.object(core, "resolve_model_class", return_value=model_class):
            _, resolved_package_dir = initialize_model(
                model_id=model_id,
                model_architecture="rfdetr",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=MagicMock(),
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )

        v2_package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )

    assert resolved_package_dir == legacy_package_dir
    assert not os.path.exists(v2_package_dir)
    with open(os.path.join(legacy_package_dir, "model_config.json")) as file:
        manifest = json.load(file)
    assert manifest["model_id"] == model_id
    assert manifest["canonical_model_id"] == model_id


@pytest.mark.parametrize("use_symlink", [False, True])
def test_initialize_local_cache_revalidates_regular_artifact_md5_before_constructor(
    empty_local_dir: str,
    use_symlink: bool,
) -> None:
    model_id = "workspace/canonical/1"
    package_id = "localtrtpackage"
    original_content = b"engine bytes observed during discovery"
    expected_md5 = hashlib.md5(original_content).hexdigest()
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        with open(os.path.join(package_dir, "model_config.json"), "w") as manifest_file:
            json.dump({"model_id": model_id}, manifest_file)
        artifact_path = os.path.join(package_dir, "model.engine")
        if use_symlink:
            outside_artifact = os.path.join(empty_local_dir, "outside.engine")
            with open(outside_artifact, "wb") as artifact_file:
                artifact_file.write(original_content)
            os.symlink(outside_artifact, artifact_path)
        else:
            with open(artifact_path, "wb") as artifact_file:
                artifact_file.write(b"bytes changed after discovery")
        package = ModelPackageMetadata(
            package_id=package_id,
            backend=BackendType.TRT,
            package_artefacts=[
                LocalFileArtefactSpecs(
                    file_handle="model.engine",
                    md5_hash=expected_md5,
                )
            ],
            package_source=PackageSourceType.LOCAL_CACHE,
            trusted_source=False,
            cache_model_id=model_id,
        )
        model_class = MagicMock()
        with mock.patch.object(core, "resolve_model_class", return_value=model_class):
            with pytest.raises(CorruptedModelPackageError):
                initialize_model(
                    model_id=model_id,
                    model_architecture="rfdetr",
                    task_type="object-detection",
                    model_package=package,
                    model_init_kwargs={},
                    auto_resolution_cache=MagicMock(),
                    auto_negotiation_hash="a" * 64,
                    model_dependencies=[],
                    model_dependencies_instances={},
                    model_dependencies_directories={},
                )

    model_class.from_pretrained.assert_not_called()


def test_initialize_rejects_local_alias_package_owned_by_other_canonical_model(
    empty_local_dir: str,
) -> None:
    alias_model_id = "shared-alias/1"
    package_id = "localtrtpackage"
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.TRT,
        package_artefacts=[],
        package_source=PackageSourceType.LOCAL_CACHE,
        trusted_source=False,
        cache_model_id=alias_model_id,
    )
    model_class = MagicMock()

    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=alias_model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        manifest_hash = dump_model_config_for_offline_use(
            config_path=os.path.join(package_dir, "model_config.json"),
            model_architecture="rfdetr",
            task_type="object-detection",
            backend_type=BackendType.TRT,
            file_lock_acquire_timeout=1,
            model_id=alias_model_id,
            canonical_model_id="tenant-a/canonical/1",
        )
        with mock.patch.object(core, "resolve_model_class", return_value=model_class):
            with pytest.raises(
                CorruptedModelPackageError,
                match="different canonical model",
            ):
                initialize_model(
                    model_id="tenant-b/canonical/1",
                    requested_model_id=alias_model_id,
                    model_architecture="rfdetr",
                    task_type="object-detection",
                    model_package=package,
                    model_init_kwargs={},
                    auto_resolution_cache=MagicMock(),
                    auto_negotiation_hash="b" * 64,
                    model_dependencies=[],
                    model_dependencies_instances={},
                    model_dependencies_directories={},
                )

    model_class.from_pretrained.assert_not_called()


def test_auto_load_exact_cache_rejects_manifest_canonical_owner_mismatch(
    empty_local_dir: str,
) -> None:
    cache_model_id = "tenant-a/canonical/1"
    package_id = "package"
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=cache_model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        manifest_hash = dump_model_config_for_offline_use(
            config_path=os.path.join(package_dir, "model_config.json"),
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=1,
            model_id=cache_model_id,
            canonical_model_id="tenant-b/canonical/1",
            trusted_source=True,
            model_dependencies=[],
            runtime_compatibility_hash=core._runtime_compatibility_hash(
                runtime_x_ray=core.x_ray_runtime_environment()
            ),
        )
        cache_entry = AutoResolutionCacheEntry(
            model_id="shared-alias/1",
            cache_model_id=cache_model_id,
            canonical_model_id=cache_model_id,
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            model_package_id=package_id,
            resolved_files=[],
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=datetime.now(),
            trusted_source=True,
            package_manifest_hash=manifest_hash,
        )
        auto_resolution_cache = MagicMock()
        auto_resolution_cache.retrieve.return_value = cache_entry
        model_access_manager = MagicMock()

        result = attempt_loading_model_with_auto_load_cache(
            use_auto_resolution_cache=True,
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_access_manager=model_access_manager,
            model_name_or_path="shared-alias/1",
            model_init_kwargs={},
            api_key="original-key",
            allow_loading_dependency_models=True,
            forwarded_kwargs_values={},
        )

    assert result is None
    model_access_manager.is_model_package_access_granted.assert_not_called()


def test_auto_load_exact_cache_rejects_same_owner_manifest_rewrite(
    empty_local_dir: str,
) -> None:
    model_id = "tenant-a/canonical/1"
    package_id = "package"
    runtime_hash = core._runtime_compatibility_hash(
        runtime_x_ray=core.x_ray_runtime_environment()
    )
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        manifest_path = os.path.join(package_dir, "model_config.json")
        original_hash = dump_model_config_for_offline_use(
            config_path=manifest_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=1,
            model_id=model_id,
            canonical_model_id=model_id,
            trusted_source=True,
            model_dependencies=[],
            runtime_compatibility_hash=runtime_hash,
        )
        with open(manifest_path) as manifest_file:
            rewritten_manifest = json.load(manifest_file)
        rewritten_manifest["trusted_source"] = False
        with open(manifest_path, "w") as manifest_file:
            json.dump(rewritten_manifest, manifest_file)
        cache_entry = AutoResolutionCacheEntry(
            model_id=model_id,
            cache_model_id=model_id,
            canonical_model_id=model_id,
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            model_package_id=package_id,
            resolved_files=[manifest_path],
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=datetime.now(),
            model_dependencies=[],
            trusted_source=True,
            package_manifest_hash=original_hash,
        )
        auto_resolution_cache = MagicMock()
        auto_resolution_cache.retrieve.return_value = cache_entry
        model_access_manager = MagicMock()

        with mock.patch.object(core, "resolve_model_class") as resolve_model_class:
            result = attempt_loading_model_with_auto_load_cache(
                use_auto_resolution_cache=True,
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_access_manager=model_access_manager,
                model_name_or_path=model_id,
                model_init_kwargs={},
                api_key="original-key",
                allow_loading_dependency_models=True,
                forwarded_kwargs_values={},
            )

    assert result is None
    resolve_model_class.assert_not_called()
    model_access_manager.is_model_package_access_granted.assert_not_called()


def test_auto_load_exact_keyed_cache_accepts_matching_canonical_manifest(
    empty_local_dir: str,
) -> None:
    requested_model_id = "shared-alias/1"
    canonical_model_id = "tenant-a/canonical/1"
    package_id = "package"
    expected_model = MagicMock()

    class CachedModel:
        @classmethod
        def from_pretrained(cls, model_dir_or_weights_path, **kwargs):
            assert model_dir_or_weights_path == package_dir
            return expected_model

    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=canonical_model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        manifest_path = os.path.join(package_dir, "model_config.json")
        manifest_hash = dump_model_config_for_offline_use(
            config_path=manifest_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=1,
            model_id=canonical_model_id,
            canonical_model_id=canonical_model_id,
            trusted_source=True,
            model_dependencies=[],
            runtime_compatibility_hash=core._runtime_compatibility_hash(
                runtime_x_ray=core.x_ray_runtime_environment()
            ),
            offline_compatibility_hash="a" * 64,
        )
        cache_entry = AutoResolutionCacheEntry(
            model_id=requested_model_id,
            cache_model_id=canonical_model_id,
            canonical_model_id=canonical_model_id,
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            model_package_id=package_id,
            resolved_files=[manifest_path],
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=datetime.now(),
            offline_compatibility_hash="b" * 64,
            trusted_source=True,
            package_manifest_hash=manifest_hash,
        )
        auto_resolution_cache = MagicMock()
        auto_resolution_cache.retrieve.return_value = cache_entry
        model_access_manager = MagicMock()
        model_access_manager.is_model_package_access_granted.return_value = True

        with mock.patch.object(core, "resolve_model_class", return_value=CachedModel):
            result = attempt_loading_model_with_auto_load_cache(
                use_auto_resolution_cache=True,
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_access_manager=model_access_manager,
                model_name_or_path=requested_model_id,
                model_init_kwargs={},
                api_key="original-key",
                allow_loading_dependency_models=True,
                forwarded_kwargs_values={},
                expected_offline_compatibility_hash="b" * 64,
            )

    assert result is expected_model
    model_access_manager.is_model_package_access_granted.assert_called_once_with(
        model_id=canonical_model_id,
        package_id=package_id,
        api_key="original-key",
    )


def test_auto_load_exact_cache_rejects_resolution_constraints_mismatch() -> None:
    cache_entry = AutoResolutionCacheEntry(
        model_id="shared-alias/1",
        cache_model_id="workspace/canonical/1",
        canonical_model_id="workspace/canonical/1",
        cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        offline_compatibility_hash="a" * 64,
        trusted_source=True,
        package_manifest_hash=TEST_PACKAGE_MANIFEST_HASH,
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.retrieve.return_value = cache_entry
    model_access_manager = MagicMock()

    result = attempt_loading_model_with_auto_load_cache(
        use_auto_resolution_cache=True,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash="c" * 64,
        model_access_manager=model_access_manager,
        model_name_or_path="shared-alias/1",
        model_init_kwargs={},
        api_key="original-key",
        allow_loading_dependency_models=True,
        forwarded_kwargs_values={},
        expected_offline_compatibility_hash="b" * 64,
    )

    assert result is None
    model_access_manager.is_model_package_access_granted.assert_not_called()


def test_auto_load_exact_cache_rejects_constructor_artifact_mutation(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/canonical/1"
    package_id = "localPackage"
    original_content = b"trusted weights"
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        artifact_path = os.path.join(package_dir, "weights.bin")
        with open(artifact_path, "wb") as artifact_file:
            artifact_file.write(original_content)
        with open(
            os.path.join(package_dir, MODEL_CONFIG_FILE_NAME),
            "w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump({"model_id": model_id}, manifest_file)
        package = ModelPackageMetadata(
            package_id=package_id,
            backend=BackendType.ONNX,
            package_artefacts=[
                LocalFileArtefactSpecs(
                    file_handle="weights.bin",
                    md5_hash=hashlib.md5(original_content).hexdigest(),
                )
            ],
            package_source=PackageSourceType.LOCAL_CACHE,
            trusted_source=True,
            cache_model_id=model_id,
        )
        model_class = MagicMock()
        model_class.from_pretrained.return_value = MagicMock()
        auto_resolution_cache = MagicMock()
        with mock.patch.object(
            core,
            "resolve_model_class",
            return_value=model_class,
        ):
            initialize_model(
                model_id=model_id,
                model_architecture="yolov8",
                task_type="object-detection",
                model_package=package,
                model_init_kwargs={},
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_dependencies=[],
                model_dependencies_instances={},
                model_dependencies_directories={},
            )
            cache_entry = auto_resolution_cache.register.call_args.kwargs["cache_entry"]
            auto_resolution_cache.retrieve.return_value = cache_entry
            model_access_manager = MagicMock()
            model_access_manager.is_model_package_access_granted.return_value = True

            def mutate_artifact(*args, **kwargs):
                with open(artifact_path, "wb") as artifact_file:
                    artifact_file.write(b"constructor mutation")
                return MagicMock()

            model_class.from_pretrained.side_effect = mutate_artifact
            result = attempt_loading_model_with_auto_load_cache(
                use_auto_resolution_cache=True,
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_access_manager=model_access_manager,
                model_name_or_path=model_id,
                model_init_kwargs={},
                api_key="original-key",
                allow_loading_dependency_models=True,
                forwarded_kwargs_values={},
            )

    assert result is None


@pytest.mark.parametrize("trusted_source", [None, False])
def test_auto_load_cache_rejects_entry_without_trusted_provenance(
    trusted_source: Optional[bool],
) -> None:
    cache_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        trusted_source=trusted_source,
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.retrieve.return_value = cache_entry
    model_access_manager = MagicMock()

    result = attempt_loading_model_with_auto_load_cache(
        use_auto_resolution_cache=True,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash="a" * 64,
        model_access_manager=model_access_manager,
        model_name_or_path="workspace/model/1",
        model_init_kwargs={},
        api_key="api-key",
        allow_loading_dependency_models=True,
        forwarded_kwargs_values={},
    )

    assert result is None
    model_access_manager.is_model_package_access_granted.assert_not_called()


def test_auto_load_cache_rejects_dependencies_when_disabled() -> None:
    cache_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        trusted_source=True,
        model_dependencies=[
            ModelDependency(
                name="encoder",
                model_id="dependency/1",
                model_package_id="dependencyPackage",
            )
        ],
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.retrieve.return_value = cache_entry
    model_access_manager = MagicMock()
    model_access_manager.is_model_package_access_granted.return_value = True

    with mock.patch.object(core.AutoModel, "from_pretrained") as dependency_load:
        result = attempt_loading_model_with_auto_load_cache(
            use_auto_resolution_cache=True,
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_access_manager=model_access_manager,
            model_name_or_path="workspace/model/1",
            model_init_kwargs={},
            api_key="api-key",
            allow_loading_dependency_models=False,
            forwarded_kwargs_values={},
        )

    assert result is None
    dependency_load.assert_not_called()


def test_auto_load_cache_does_not_mutate_dependency_model_parameters() -> None:
    cache_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        trusted_source=True,
        model_dependencies=[
            ModelDependency(
                name="encoder",
                model_id="dependency/1",
                model_package_id="dependencyPackage",
            )
        ],
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.retrieve.return_value = cache_entry
    model_access_manager = MagicMock()
    model_access_manager.is_model_package_access_granted.return_value = True
    dependency_model = MagicMock()
    parent_model = MagicMock()
    caller_owned_params = {
        "encoder": {
            "device": "cpu",
            "custom_option": "keep-me",
        }
    }

    class CachedModel:
        @classmethod
        def from_pretrained(cls, model_dir_or_weights_path, **kwargs):
            return parent_model

    expected_dependency_identity = {
        "name": "encoder",
        "target_path": "/cached/dependency",
        "cache_model_id": "dependency/1",
        "canonical_model_id": "dependency/1",
        "model_package_id": "dependencyPackage",
        "package_manifest_hash": TEST_PACKAGE_MANIFEST_HASH,
    }
    package_config = InferenceModelConfig(
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        model_module=None,
        model_class=None,
        dependency_package_paths=[expected_dependency_identity],
    )

    def load_dependency(**kwargs):
        kwargs["point_model_directory"]("/cached/dependency")
        return dependency_model

    with mock.patch.object(
        core.AutoModel, "from_pretrained", side_effect=load_dependency
    ) as dependency_load, mock.patch.object(
        core, "resolve_model_class", return_value=CachedModel
    ), mock.patch.object(
        core, "generate_model_package_cache_path", return_value="/cached/model"
    ), mock.patch.object(
        core, "_verified_auto_cache_package_dir", return_value="/cached/model"
    ), mock.patch.object(
        core, "parse_model_config", return_value=package_config
    ), mock.patch.object(
        core,
        "_dependency_package_identity_for_path",
        return_value=expected_dependency_identity,
    ):
        result = attempt_loading_model_with_auto_load_cache(
            use_auto_resolution_cache=True,
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_access_manager=model_access_manager,
            model_name_or_path="workspace/model/1",
            model_init_kwargs={},
            api_key="api-key",
            allow_loading_dependency_models=True,
            forwarded_kwargs_values={},
            dependency_models_params=caller_owned_params,
        )

    assert result is parent_model
    assert caller_owned_params == {
        "encoder": {
            "device": "cpu",
            "custom_option": "keep-me",
        }
    }
    dependency_load.assert_called_once()
    assert dependency_load.call_args.kwargs["custom_option"] == "keep-me"


def test_exact_cache_rejects_dependency_resolved_to_unbound_package() -> None:
    dependency = ModelDependency(
        name="encoder",
        model_id="dependency/1",
        model_package_id="dependencyPackage",
    )
    cache_entry = AutoResolutionCacheEntry(
        model_id="workspace/model/1",
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        trusted_source=True,
        model_dependencies=[dependency],
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.retrieve.return_value = cache_entry
    model_access_manager = MagicMock()
    model_access_manager.is_model_package_access_granted.return_value = True
    expected_dependency_identity = {
        "name": "encoder",
        "target_path": "/cached/dependency-package-a",
        "cache_model_id": "dependency/1",
        "canonical_model_id": "dependency/1",
        "model_package_id": "dependencyPackage",
        "package_manifest_hash": TEST_PACKAGE_MANIFEST_HASH,
    }
    package_config = InferenceModelConfig(
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        model_module=None,
        model_class=None,
        dependency_package_paths=[expected_dependency_identity],
    )

    def load_wrong_dependency(**kwargs):
        kwargs["point_model_directory"]("/cached/dependency-package-b")
        return MagicMock()

    with mock.patch.object(
        core.AutoModel,
        "from_pretrained",
        side_effect=load_wrong_dependency,
    ), mock.patch.object(
        core, "_verified_auto_cache_package_dir", return_value="/cached/model"
    ), mock.patch.object(
        core, "parse_model_config", return_value=package_config
    ), mock.patch.object(
        core,
        "_dependency_package_identity_for_path",
        return_value={
            **expected_dependency_identity,
            "target_path": "/cached/dependency-package-b",
        },
    ), mock.patch.object(
        core, "resolve_model_class"
    ) as resolve_parent_model:
        result = attempt_loading_model_with_auto_load_cache(
            use_auto_resolution_cache=True,
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_access_manager=model_access_manager,
            model_name_or_path="workspace/model/1",
            model_init_kwargs={},
            api_key="api-key",
            allow_loading_dependency_models=True,
            forwarded_kwargs_values={},
        )

    assert result is None
    resolve_parent_model.assert_not_called()


def test_dump_model_config_for_offline_use_rejects_symlink_target(
    empty_local_dir: str,
) -> None:
    outside_config = os.path.join(empty_local_dir, "outside.json")
    _create_file(outside_config, '{"sentinel": true}')
    package_dir = os.path.join(empty_local_dir, "package")
    os.makedirs(package_dir)
    config_path = os.path.join(package_dir, "model_config.json")
    os.symlink(outside_config, config_path)

    with pytest.raises(CorruptedModelPackageError):
        dump_model_config_for_offline_use(
            config_path,
            "yolov8",
            "object-detection",
            BackendType.ONNX,
            10,
            model_id="workspace/project/3",
            canonical_model_id="workspace/project/3",
        )

    with open(outside_config) as file:
        assert json.load(file) == {"sentinel": True}


def test_create_symlinks_to_shared_blobs_when_hooks_provided(
    empty_local_dir: str,
) -> None:
    # given
    shared_dir = os.path.join(empty_local_dir, "shared")
    shared_file_a = os.path.join(shared_dir, "a.txt")
    _create_file(path=shared_file_a, content="a")
    shared_file_b = os.path.join(shared_dir, "b.txt")
    _create_file(path=shared_file_b, content="b")
    model_dir = os.path.join(shared_dir, "model_dir")
    broken_file = os.path.join(shared_dir, "broken.txt")
    _create_file(path=broken_file, content="broken")
    os.makedirs(model_dir, exist_ok=True)
    existing_model_file = os.path.join(model_dir, "existing.txt")
    _create_file(path=existing_model_file, content="existing")
    initially_broken_link = os.path.join(model_dir, "initially_broken.txt")
    os.symlink(broken_file, initially_broken_link)
    os.remove(broken_file)
    shared_files_mapping = {
        "my_file_a.txt": shared_file_a,
        "my_file_b.txt": shared_file_b,
        "existing.txt": shared_file_a,
        "initially_broken.txt": shared_file_b,
    }
    on_symlink_created = MagicMock()
    on_symlink_deleted = MagicMock()

    # when
    result = create_symlinks_to_shared_blobs(
        model_dir=model_dir,
        shared_files_mapping=shared_files_mapping,
        on_symlink_deleted=on_symlink_deleted,
        on_symlink_created=on_symlink_created,
    )

    # then
    assert result == {
        "my_file_a.txt": os.path.join(model_dir, "my_file_a.txt"),
        "my_file_b.txt": os.path.join(model_dir, "my_file_b.txt"),
        "existing.txt": os.path.join(model_dir, "existing.txt"),
        "initially_broken.txt": os.path.join(model_dir, "initially_broken.txt"),
    }
    on_symlink_deleted.assert_called_once_with(initially_broken_link)
    on_symlink_created.assert_has_calls(
        [
            call.__bool__(),
            call(shared_file_a, os.path.join(model_dir, "my_file_a.txt")),
            call.__bool__(),
            call(shared_file_b, os.path.join(model_dir, "my_file_b.txt")),
            call.__bool__(),
            call(shared_file_b, os.path.join(model_dir, "initially_broken.txt")),
        ]
    )
    assert _read_file(result["my_file_a.txt"]) == "a"
    assert _read_file(result["my_file_b.txt"]) == "b"
    assert _read_file(result["existing.txt"]) == "existing"
    assert _read_file(result["initially_broken.txt"]) == "b"


def test_create_symlinks_to_shared_blobs_when_hooks_not_provided(
    empty_local_dir: str,
) -> None:
    # given
    shared_dir = os.path.join(empty_local_dir, "shared")
    shared_file_a = os.path.join(shared_dir, "a.txt")
    _create_file(path=shared_file_a, content="a")
    shared_file_b = os.path.join(shared_dir, "b.txt")
    _create_file(path=shared_file_b, content="b")
    model_dir = os.path.join(shared_dir, "model_dir")
    broken_file = os.path.join(shared_dir, "broken.txt")
    _create_file(path=broken_file, content="broken")
    os.makedirs(model_dir, exist_ok=True)
    existing_model_file = os.path.join(model_dir, "existing.txt")
    _create_file(path=existing_model_file, content="existing")
    initially_broken_link = os.path.join(model_dir, "initially_broken.txt")
    os.symlink(broken_file, initially_broken_link)
    os.remove(broken_file)
    shared_files_mapping = {
        "my_file_a.txt": shared_file_a,
        "my_file_b.txt": shared_file_b,
        "existing.txt": shared_file_a,
        "initially_broken.txt": shared_file_b,
    }

    # when
    result = create_symlinks_to_shared_blobs(
        model_dir=model_dir,
        shared_files_mapping=shared_files_mapping,
    )

    # then
    assert result == {
        "my_file_a.txt": os.path.join(model_dir, "my_file_a.txt"),
        "my_file_b.txt": os.path.join(model_dir, "my_file_b.txt"),
        "existing.txt": os.path.join(model_dir, "existing.txt"),
        "initially_broken.txt": os.path.join(model_dir, "initially_broken.txt"),
    }
    assert _read_file(result["my_file_a.txt"]) == "a"
    assert _read_file(result["my_file_b.txt"]) == "b"
    assert _read_file(result["existing.txt"]) == "existing"
    assert _read_file(result["initially_broken.txt"]) == "b"


def _create_file(path: str, content: str) -> None:
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Offline cache discovery and OFFLINE_MODE behaviour
# ---------------------------------------------------------------------------


def _write_offline_package(
    inference_home: str,
    model_id: str,
    package_id: str,
    config: Optional[dict] = None,
) -> str:
    slug = model_cache_paths.slugify_model_id_to_os_safe_format(model_id=model_id)
    package_dir = os.path.join(inference_home, "models-cache", slug, package_id)
    os.makedirs(package_dir, exist_ok=True)
    if config is not None:
        config = {**config, "model_id": model_id}
        config.setdefault("canonical_model_id", model_id)
        _create_file(os.path.join(package_dir, "model_config.json"), json.dumps(config))
    # scanning helpers yield realpath-resolved paths
    return os.path.realpath(package_dir)


def _offline_compatibility_hash_for_default_request(model_id: str) -> str:
    return core.hash_dict_content(
        content={
            "provider": "roboflow",
            "model_id": model_id,
            "requested_model_package_id": None,
            "requested_backends": None,
            "requested_batch_size": None,
            "requested_quantization": None,
            "device": str(core.DEFAULT_DEVICE),
            "onnx_execution_providers": None,
            "default_onnx_trt_options": True,
            "allow_untrusted_packages": False,
            "trt_engine_host_code_allowed": True,
            "allow_local_code_packages": True,
            "verify_hash_while_download": True,
            "download_files_without_hash": False,
            "allow_loading_dependency_models": True,
            "max_package_loading_attempts": None,
            "nms_fusion_preferences": None,
            "weights_provider_extra_query_params": None,
            "weights_provider_extra_headers": None,
            "dependency_models_params": {},
            "forwarded_dependency_kwargs": {},
            "runtime_compatibility": core._runtime_compatibility_content(
                runtime_x_ray=core.x_ray_runtime_environment()
            ),
        }
    )


_OFFLINE_PACKAGE_CONFIG = {
    "offline_manifest_version": core.OFFLINE_CACHE_MANIFEST_VERSION,
    "model_architecture": "yolov8",
    "task_type": "object-detection",
    "backend_type": "onnx",
    "model_id": "yolov8n-640",
    "model_features": None,
    "trusted_source": True,
    "model_dependencies": [],
    "recommended_parameters": None,
    "quantization": "unknown",
    "dynamic_batch_size_supported": None,
    "static_batch_size": None,
    "runtime_compatibility_hash": core._runtime_compatibility_hash(
        runtime_x_ray=core.x_ray_runtime_environment()
    ),
    "offline_compatibility_hash": None,
    "package_artifacts": [],
    "dependency_package_paths": [],
}


def test_cached_package_rejects_tampered_shared_blob_for_exact_and_raw_loads(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    package_id = "package"
    file_handle = "weights.onnx"
    original_content = b"original trusted weights"
    md5_hash = hashlib.md5(original_content).hexdigest()
    runtime_hash = core._runtime_compatibility_hash(
        runtime_x_ray=core.x_ray_runtime_environment()
    )
    artifact_identity = {
        "file_handle": file_handle,
        "md5_hash": md5_hash,
        "unhashed": False,
        "sha256_hash": None,
        "source_hash": None,
        "storage": "shared_blob",
    }

    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        shared_blobs_dir = core.generate_shared_blobs_path()
        os.makedirs(shared_blobs_dir)
        shared_blob_path = os.path.join(shared_blobs_dir, md5_hash)
        with open(shared_blob_path, "wb") as shared_blob:
            shared_blob.write(original_content)
        package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        package_link = os.path.join(package_dir, file_handle)
        os.symlink(shared_blob_path, package_link)
        manifest_path = os.path.join(package_dir, "model_config.json")
        manifest_hash = dump_model_config_for_offline_use(
            config_path=manifest_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=1,
            model_id=model_id,
            canonical_model_id=model_id,
            trusted_source=True,
            model_dependencies=[],
            runtime_compatibility_hash=runtime_hash,
            package_artifacts=[artifact_identity],
        )
        cache_entry = AutoResolutionCacheEntry(
            model_id=model_id,
            cache_model_id=model_id,
            canonical_model_id=model_id,
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            model_package_id=package_id,
            resolved_files=[shared_blob_path, package_link, manifest_path],
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            model_dependencies=[],
            created_at=datetime.now(),
            trusted_source=True,
            package_manifest_hash=manifest_hash,
        )
        with open(shared_blob_path, "wb") as shared_blob:
            shared_blob.write(b"tampered weights")

        auto_resolution_cache = MagicMock()
        auto_resolution_cache.retrieve.return_value = cache_entry
        model_access_manager = MagicMock()
        with mock.patch.object(
            core, "attempt_loading_model_from_local_storage"
        ) as local_load, mock.patch.object(
            core, "resolve_model_class"
        ) as resolve_model_class:
            exact_result = attempt_loading_model_with_auto_load_cache(
                use_auto_resolution_cache=True,
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_access_manager=model_access_manager,
                model_name_or_path=model_id,
                model_init_kwargs={},
                api_key="test-key",
                allow_loading_dependency_models=True,
                forwarded_kwargs_values={},
            )
            raw_result = attempt_loading_model_from_offline_cache(
                model_id=model_id,
                model_init_kwargs={},
            )
            discovered_dir = find_cached_model_package_dir(model_id=model_id)

    assert exact_result is None
    assert raw_result is None
    assert discovered_dir is None
    model_access_manager.is_model_package_access_granted.assert_not_called()
    resolve_model_class.assert_not_called()
    local_load.assert_not_called()


def test_cached_package_rejects_undeclared_file_for_exact_raw_and_metadata_loads(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    package_id = "package"
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id=package_id,
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    manifest_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
    with open(os.path.join(package_dir, "adapter_config.json"), "w") as undeclared_file:
        json.dump({"base_model_name_or_path": "attacker/model"}, undeclared_file)
    cache_entry = AutoResolutionCacheEntry(
        model_id=model_id,
        cache_model_id=model_id,
        canonical_model_id=model_id,
        cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
        credential_hash=core._credential_hash(api_key="test-key"),
        model_package_id=package_id,
        resolved_files=[manifest_path],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        model_dependencies=[],
        created_at=datetime.now(),
        trusted_source=True,
        package_manifest_hash=parse_model_config(
            config_path=manifest_path
        ).manifest_content_hash,
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.retrieve.return_value = cache_entry
    model_access_manager = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "resolve_model_class"
    ) as resolve_model_class, mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as local_load:
        exact_result = attempt_loading_model_with_auto_load_cache(
            use_auto_resolution_cache=True,
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_access_manager=model_access_manager,
            model_name_or_path=model_id,
            model_init_kwargs={},
            api_key="test-key",
            allow_loading_dependency_models=True,
            forwarded_kwargs_values={},
        )
        raw_result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )
        metadata_result = find_cached_model_package_dir(model_id=model_id)

    assert exact_result is None
    assert raw_result is None
    assert metadata_result is None
    resolve_model_class.assert_not_called()
    local_load.assert_not_called()


def test_stale_manifest_rejects_repointed_artifact_link_for_exact_and_raw_loads(
    empty_local_dir: str,
) -> None:
    model_id = "workspace/project/1"
    package_id = "package"
    file_handle = "weights.onnx"
    original_content = b"original trusted weights"
    md5_hash = hashlib.md5(original_content).hexdigest()
    runtime_hash = core._runtime_compatibility_hash(
        runtime_x_ray=core.x_ray_runtime_environment()
    )
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        shared_blobs_dir = core.generate_shared_blobs_path()
        os.makedirs(shared_blobs_dir)
        expected_blob_path = os.path.join(shared_blobs_dir, md5_hash)
        rogue_blob_path = os.path.join(shared_blobs_dir, "rogue")
        with open(expected_blob_path, "wb") as expected_blob:
            expected_blob.write(original_content)
        with open(rogue_blob_path, "wb") as rogue_blob:
            rogue_blob.write(b"untrusted replacement")
        package_dir = generate_model_package_cache_path(
            model_id=model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        package_link = os.path.join(package_dir, file_handle)
        os.symlink(expected_blob_path, package_link)
        manifest_path = os.path.join(package_dir, "model_config.json")
        manifest_hash = dump_model_config_for_offline_use(
            config_path=manifest_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=1,
            model_id=model_id,
            canonical_model_id=model_id,
            trusted_source=True,
            model_dependencies=[],
            runtime_compatibility_hash=runtime_hash,
            package_artifacts=[
                {
                    "file_handle": file_handle,
                    "md5_hash": md5_hash,
                    "unhashed": False,
                    "sha256_hash": None,
                    "source_hash": None,
                    "storage": "shared_blob",
                }
            ],
        )
        os.unlink(package_link)
        os.symlink(rogue_blob_path, package_link)
        cache_entry = AutoResolutionCacheEntry(
            model_id=model_id,
            cache_model_id=model_id,
            canonical_model_id=model_id,
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            model_package_id=package_id,
            resolved_files=[expected_blob_path, package_link, manifest_path],
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            model_dependencies=[],
            created_at=datetime.now(),
            trusted_source=True,
            package_manifest_hash=manifest_hash,
        )
        auto_resolution_cache = MagicMock()
        auto_resolution_cache.retrieve.return_value = cache_entry
        model_access_manager = MagicMock()

        with mock.patch.object(
            core, "attempt_loading_model_from_local_storage"
        ) as local_load:
            exact_result = attempt_loading_model_with_auto_load_cache(
                use_auto_resolution_cache=True,
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash="a" * 64,
                model_access_manager=model_access_manager,
                model_name_or_path=model_id,
                model_init_kwargs={},
                api_key="test-key",
                allow_loading_dependency_models=True,
                forwarded_kwargs_values={},
            )
            raw_result = attempt_loading_model_from_offline_cache(
                model_id=model_id,
                model_init_kwargs={},
            )

    assert exact_result is None
    assert raw_result is None
    model_access_manager.is_model_package_access_granted.assert_not_called()
    local_load.assert_not_called()


def test_parent_cache_rejects_dependency_rewarm_at_same_package_path(
    empty_local_dir: str,
) -> None:
    dependency_model_id = "workspace/encoder/1"
    dependency_package_id = "dependencyPackage"
    dependency_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=dependency_model_id,
        package_id=dependency_package_id,
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    dependency_manifest_path = os.path.join(dependency_dir, "model_config.json")
    dependency_manifest_hash = parse_model_config(
        dependency_manifest_path
    ).manifest_content_hash
    dependency_identity = {
        "name": "encoder",
        "target_path": dependency_dir,
        "cache_model_id": dependency_model_id,
        "canonical_model_id": dependency_model_id,
        "model_package_id": dependency_package_id,
        "package_manifest_hash": dependency_manifest_hash,
    }
    parent_model_id = "workspace/parent/1"
    parent_package_id = "parentPackage"
    parent_dependencies = [
        {
            "name": "encoder",
            "model_id": dependency_model_id,
            "model_package_id": dependency_package_id,
        }
    ]
    parent_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=parent_model_id,
        package_id=parent_package_id,
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "model_dependencies": parent_dependencies,
            "dependency_package_paths": [dependency_identity],
        },
    )
    dependency_links_dir = os.path.join(parent_dir, core.MODEL_DEPENDENCIES_SUB_DIR)
    os.makedirs(dependency_links_dir)
    os.symlink(
        dependency_dir,
        os.path.join(dependency_links_dir, "encoder"),
    )
    parent_manifest_path = os.path.join(parent_dir, "model_config.json")
    parent_manifest_hash = parse_model_config(
        parent_manifest_path
    ).manifest_content_hash
    parent_entry = AutoResolutionCacheEntry(
        model_id=parent_model_id,
        cache_model_id=parent_model_id,
        canonical_model_id=parent_model_id,
        cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
        model_package_id=parent_package_id,
        resolved_files=[parent_manifest_path],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        model_dependencies=[ModelDependency.model_validate(parent_dependencies[0])],
        created_at=datetime.now(),
        trusted_source=True,
        package_manifest_hash=parent_manifest_hash,
    )
    with open(dependency_manifest_path) as dependency_manifest_file:
        rewritten_dependency_manifest = json.load(dependency_manifest_file)
    rewritten_dependency_manifest["recommended_parameters"] = {"confidence": 0.25}
    with open(dependency_manifest_path, "w") as dependency_manifest_file:
        json.dump(rewritten_dependency_manifest, dependency_manifest_file)

    auto_resolution_cache = MagicMock()
    auto_resolution_cache.retrieve.return_value = parent_entry
    model_access_manager = MagicMock()
    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as local_load:
        exact_result = attempt_loading_model_with_auto_load_cache(
            use_auto_resolution_cache=True,
            auto_resolution_cache=auto_resolution_cache,
            auto_negotiation_hash="a" * 64,
            model_access_manager=model_access_manager,
            model_name_or_path=parent_model_id,
            model_init_kwargs={},
            api_key="test-key",
            allow_loading_dependency_models=True,
            forwarded_kwargs_values={},
        )
        raw_result = attempt_loading_model_from_offline_cache(
            model_id=parent_model_id,
            model_init_kwargs={},
        )

    assert exact_result is None
    assert raw_result is None
    model_access_manager.is_model_package_access_granted.assert_not_called()
    local_load.assert_not_called()


def test_raw_cache_rejects_tampered_unsafe_artifact_handle(
    empty_local_dir: str,
) -> None:
    config = {
        **_OFFLINE_PACKAGE_CONFIG,
        "package_artifacts": [
            {
                "file_handle": "../weights.onnx",
                "md5_hash": "a" * 32,
                "unhashed": False,
                "sha256_hash": None,
                "source_hash": None,
                "storage": "shared_blob",
            }
        ],
    }
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id="workspace/project/1",
        package_id="package",
        config=config,
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as local_load:
        result = attempt_loading_model_from_offline_cache(
            model_id="workspace/project/1",
            model_init_kwargs={},
        )

    assert result is None
    local_load.assert_not_called()


def test_find_cached_model_package_dir_when_valid_package_exists(
    empty_local_dir: str,
) -> None:
    # given
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id="coco/22",
        package_id="pkg001",
        config=_OFFLINE_PACKAGE_CONFIG,
    )

    # when
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        result = find_cached_model_package_dir(model_id="coco/22")

    # then
    assert result == package_dir


def test_find_cached_model_package_dir_resolves_alias_with_credential_scope(
    empty_local_dir: str,
) -> None:
    from inference_models.models.auto_loaders import auto_resolution_cache

    alias_model_id = "workspace/alias/1"
    canonical_model_id = "workspace/canonical/1"
    api_key = "credential-a"
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=canonical_model_id,
        package_id="pkg001",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    manifest_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
    manifest_hash = parse_model_config(config_path=manifest_path).manifest_content_hash
    cache = auto_resolution_cache.BaseAutoLoadMetadataCache(file_lock_acquire_timeout=1)
    entry = AutoResolutionCacheEntry(
        model_id=alias_model_id,
        cache_model_id=canonical_model_id,
        canonical_model_id=canonical_model_id,
        cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
        credential_hash=core._credential_hash(api_key=api_key),
        model_package_id="pkg001",
        resolved_files=[manifest_path],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        model_dependencies=[],
        created_at=datetime.now(),
        trusted_source=True,
        package_manifest_hash=manifest_hash,
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(auto_resolution_cache, "INFERENCE_HOME", empty_local_dir):
        cache.register(auto_negotiation_hash="a" * 64, cache_entry=entry)
        exact_result = find_cached_model_package_dir(
            model_id=alias_model_id,
            api_key=api_key,
        )
        rotated_key_result = find_cached_model_package_dir(
            model_id=alias_model_id,
            api_key="credential-b",
        )
        with mock.patch.object(core, "ROBOFLOW_API_KEY", api_key):
            implicit_env_key_result = find_cached_model_package_dir(
                model_id=alias_model_id,
            )
        with mock.patch.object(core, "ROBOFLOW_API_KEY", None):
            keyless_result = find_cached_model_package_dir(
                model_id=alias_model_id,
            )

    assert exact_result == package_dir
    assert rotated_key_result is None
    assert implicit_env_key_result == package_dir
    assert keyless_result == package_dir


def test_find_cached_model_package_dir_prefers_keyed_alias_over_direct_collision(
    empty_local_dir: str,
) -> None:
    from inference_models.models.auto_loaders import auto_resolution_cache

    requested_model_id = "shared/name/1"
    canonical_model_id = "tenant-b/canonical/1"
    api_key = "tenant-b-key"
    direct_package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=requested_model_id,
        package_id="directpkg",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    alias_package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=canonical_model_id,
        package_id="aliaspkg",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    manifest_path = os.path.join(alias_package_dir, MODEL_CONFIG_FILE_NAME)
    cache = auto_resolution_cache.BaseAutoLoadMetadataCache(file_lock_acquire_timeout=1)

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(auto_resolution_cache, "INFERENCE_HOME", empty_local_dir):
        cache.register(
            auto_negotiation_hash="a" * 64,
            cache_entry=AutoResolutionCacheEntry(
                model_id=requested_model_id,
                cache_model_id=canonical_model_id,
                canonical_model_id=canonical_model_id,
                cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
                credential_hash=core._credential_hash(api_key=api_key),
                model_package_id="aliaspkg",
                resolved_files=[manifest_path],
                model_architecture="yolov8",
                task_type="object-detection",
                backend_type=BackendType.ONNX,
                model_dependencies=[],
                created_at=datetime.now(),
                trusted_source=True,
                package_manifest_hash=parse_model_config(
                    config_path=manifest_path
                ).manifest_content_hash,
            ),
        )
        result = find_cached_model_package_dir(
            model_id=requested_model_id,
            api_key=api_key,
        )

    assert result == alias_package_dir
    assert result != direct_package_dir


def test_find_cached_model_package_dir_no_key_rejects_direct_alias_collision(
    empty_local_dir: str,
) -> None:
    from inference_models.models.auto_loaders import auto_resolution_cache

    requested_model_id = "shared/name/1"
    canonical_model_id = "tenant-b/canonical/1"
    direct_package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=requested_model_id,
        package_id="directpkg",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    alias_package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=canonical_model_id,
        package_id="aliaspkg",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    manifest_path = os.path.join(alias_package_dir, MODEL_CONFIG_FILE_NAME)
    cache = auto_resolution_cache.BaseAutoLoadMetadataCache(file_lock_acquire_timeout=1)

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(auto_resolution_cache, "INFERENCE_HOME", empty_local_dir):
        cache.register(
            auto_negotiation_hash="a" * 64,
            cache_entry=AutoResolutionCacheEntry(
                model_id=requested_model_id,
                cache_model_id=canonical_model_id,
                canonical_model_id=canonical_model_id,
                cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
                credential_hash=core._credential_hash(api_key="tenant-b-key"),
                model_package_id="aliaspkg",
                resolved_files=[manifest_path],
                model_architecture="yolov8",
                task_type="object-detection",
                backend_type=BackendType.ONNX,
                model_dependencies=[],
                created_at=datetime.now(),
                trusted_source=True,
                package_manifest_hash=parse_model_config(
                    config_path=manifest_path
                ).manifest_content_hash,
            ),
        )
        with mock.patch.object(core, "ROBOFLOW_API_KEY", None):
            result = find_cached_model_package_dir(model_id=requested_model_id)

    assert direct_package_dir != alias_package_dir
    assert result is None


def test_keyless_alias_cache_discovery_fails_closed_on_canonical_ambiguity(
    empty_local_dir: str,
) -> None:
    from inference_models.models.auto_loaders import auto_resolution_cache

    alias_model_id = "workspace/alias/1"
    cache = auto_resolution_cache.BaseAutoLoadMetadataCache(file_lock_acquire_timeout=1)
    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(auto_resolution_cache, "INFERENCE_HOME", empty_local_dir):
        for index, canonical_model_id in enumerate(
            ("workspace/canonical-a/1", "workspace/canonical-b/1")
        ):
            package_id = f"pkg00{index}"
            package_dir = _write_offline_package(
                inference_home=empty_local_dir,
                model_id=canonical_model_id,
                package_id=package_id,
                config=_OFFLINE_PACKAGE_CONFIG,
            )
            manifest_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
            cache.register(
                auto_negotiation_hash=chr(ord("a") + index) * 64,
                cache_entry=AutoResolutionCacheEntry(
                    model_id=alias_model_id,
                    cache_model_id=canonical_model_id,
                    canonical_model_id=canonical_model_id,
                    cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
                    credential_hash=core._credential_hash(
                        api_key=f"credential-{index}"
                    ),
                    model_package_id=package_id,
                    resolved_files=[manifest_path],
                    model_architecture="yolov8",
                    task_type="object-detection",
                    backend_type=BackendType.ONNX,
                    model_dependencies=[],
                    created_at=datetime.now(),
                    trusted_source=True,
                    package_manifest_hash=parse_model_config(
                        config_path=manifest_path
                    ).manifest_content_hash,
                ),
            )

        with mock.patch.object(core, "ROBOFLOW_API_KEY", None):
            result = find_cached_model_package_dir(model_id=alias_model_id)

    assert result is None


def test_find_cached_model_package_dir_when_no_cache_present(
    empty_local_dir: str,
) -> None:
    # when
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        result = find_cached_model_package_dir(model_id="nonexistent/model")

    # then
    assert result is None


def test_find_cached_model_package_dir_rejects_mismatched_model_config(
    empty_local_dir: str,
) -> None:
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id="requested/model",
        package_id="pkg001",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    _create_file(
        os.path.join(package_dir, "model_config.json"),
        json.dumps({**_OFFLINE_PACKAGE_CONFIG, "model_id": "different/model"}),
    )

    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        result = find_cached_model_package_dir(model_id="requested/model")

    assert result is None


def test_find_cached_model_package_dir_when_package_has_no_config(
    empty_local_dir: str,
) -> None:
    # given
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id="my/model",
        package_id="pkg001",
        config=None,
    )
    _create_file(os.path.join(package_dir, "weights.onnx"), "fake")

    # when
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        result = find_cached_model_package_dir(model_id="my/model")

    # then
    assert result is None


def test_attempt_loading_model_from_offline_cache_when_no_cache_dir(
    empty_local_dir: str,
) -> None:
    # when
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        result = attempt_loading_model_from_offline_cache(
            model_id="yolov8n-640",
            model_init_kwargs={},
        )

    # then
    assert result is None


@pytest.mark.parametrize("model_id", [None, 123, "", "   ", "bad\0id"])
def test_cache_discovery_entrypoints_reject_invalid_remote_model_identity(
    model_id: object,
) -> None:
    with mock.patch.object(
        core, "_iterate_cached_model_package_dirs"
    ) as iterate_packages, mock.patch.object(
        core, "model_provider_requires_network"
    ) as provider_requires_network:
        with pytest.raises(
            InvalidParameterError,
            match="Remote model identity must be a non-empty string",
        ):
            find_cached_model_package_dir(model_id=model_id)
        with pytest.raises(
            InvalidParameterError,
            match="Remote model identity must be a non-empty string",
        ):
            attempt_loading_model_from_offline_cache(
                model_id=model_id,
                model_init_kwargs={},
            )

    iterate_packages.assert_not_called()
    provider_requires_network.assert_not_called()


@pytest.mark.parametrize("api_key", [None, "local"])
def test_raw_offline_cache_rejects_implicit_env_api_key(
    api_key: Optional[str],
) -> None:
    with mock.patch.object(
        core, "ROBOFLOW_API_KEY", "configured-env-key"
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "_iterate_cached_model_package_dirs"
    ) as iterate_packages:
        result = attempt_loading_model_from_offline_cache(
            model_id="workspace/model/1",
            model_init_kwargs={},
            api_key=api_key,
        )

    assert result is None
    iterate_packages.assert_not_called()


def test_attempt_loading_model_from_offline_cache_when_valid_package_found(
    empty_local_dir: str,
) -> None:
    # given
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id="yolov8n-640",
        package_id="pkg001",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    mock_model = MagicMock()

    # when
    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id="yolov8n-640",
            model_init_kwargs={"device": core.torch.device("cpu")},
            allow_loading_dependency_models=False,
        )

    # then
    assert result is not None
    model, cache_dir = result
    assert model is mock_model
    assert cache_dir == package_dir
    mock_load.assert_called_once_with(
        model_dir_or_weights_path=package_dir,
        allow_local_code_packages=True,
        model_init_kwargs={
            "device": core.torch.device("cpu"),
            core.MODEL_DEPENDENCIES_KEY: {},
        },
    )


def test_attempt_loading_model_from_offline_cache_skips_hidden_dirs(
    empty_local_dir: str,
) -> None:
    # given - only hidden directory, no visible package dirs
    model_id = "yolov8n-640"
    slug = model_cache_paths.slugify_model_id_to_os_safe_format(model_id=model_id)
    hidden_dir = os.path.join(empty_local_dir, "models-cache", slug, ".locks")
    os.makedirs(hidden_dir, exist_ok=True)
    _create_file(
        os.path.join(hidden_dir, "model_config.json"),
        json.dumps(_OFFLINE_PACKAGE_CONFIG),
    )

    # when
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )

    # then
    assert result is None


def test_attempt_loading_model_from_offline_cache_rejects_case_alias_packages(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        cache_root = model_cache_paths.generate_model_cache_root_for_model_id(
            model_id=model_id
        )
        uppercase_dir = os.path.join(cache_root, "Package")
        lowercase_dir = os.path.join(cache_root, "package")
        os.makedirs(uppercase_dir)
        os.makedirs(lowercase_dir, exist_ok=True)
        if os.path.samefile(uppercase_dir, lowercase_dir):
            pytest.skip("filesystem is case-insensitive")
        for package_dir in (uppercase_dir, lowercase_dir):
            _create_file(
                os.path.join(package_dir, "model_config.json"),
                json.dumps(
                    {
                        **_OFFLINE_PACKAGE_CONFIG,
                        "model_id": model_id,
                        "canonical_model_id": model_id,
                    }
                ),
            )
        with mock.patch.object(
            core, "attempt_loading_model_from_local_storage"
        ) as model_load:
            result = attempt_loading_model_from_offline_cache(
                model_id=model_id,
                model_init_kwargs={},
            )

    assert result is None
    model_load.assert_not_called()


def test_attempt_loading_model_from_offline_cache_tries_next_package_on_failure(
    empty_local_dir: str,
) -> None:
    # given - first package fails to load, second succeeds
    model_id = "yolov8n-640"
    for package_id in ["pkg001", "pkg002"]:
        _write_offline_package(
            inference_home=empty_local_dir,
            model_id=model_id,
            package_id=package_id,
            config={
                **_OFFLINE_PACKAGE_CONFIG,
                "backend_type": ("torch-script" if package_id == "pkg002" else "onnx"),
            },
        )
    mock_model = MagicMock()

    def load_side_effect(model_dir_or_weights_path, **kwargs):
        if model_dir_or_weights_path.endswith("pkg001"):
            raise RuntimeError("corrupted package")
        return mock_model

    # when
    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", side_effect=load_side_effect
    ):
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )

    # then
    assert result is not None
    model, cache_dir = result
    assert model is mock_model
    assert cache_dir.endswith("pkg002")


def test_attempt_loading_model_from_offline_cache_honors_requested_package(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    for package_id in ["pkg001", "pkg002"]:
        _write_offline_package(
            inference_home=empty_local_dir,
            model_id=model_id,
            package_id=package_id,
            config={
                **_OFFLINE_PACKAGE_CONFIG,
                "backend_type": ("torch-script" if package_id == "pkg002" else "onnx"),
            },
        )
    mock_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            requested_model_package_id="pkg002",
            requested_backends=BackendType.ONNX,
        )

    assert result is not None
    assert result[1].endswith("pkg002")
    assert mock_load.call_args[1]["model_dir_or_weights_path"].endswith("pkg002")


def test_attempt_loading_model_from_offline_cache_honors_requested_backend(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={**_OFFLINE_PACKAGE_CONFIG, "backend_type": "onnx"},
    )
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg002",
        config={**_OFFLINE_PACKAGE_CONFIG, "backend_type": "torch-script"},
    )
    mock_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            requested_backends=BackendType.TORCH_SCRIPT,
        )

    assert result is not None
    assert result[1].endswith("pkg002")
    assert mock_load.call_args[1]["model_dir_or_weights_path"].endswith("pkg002")


def test_offline_cache_ranks_allowed_backends_deterministically(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={**_OFFLINE_PACKAGE_CONFIG, "backend_type": "onnx"},
    )
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg002",
        config={**_OFFLINE_PACKAGE_CONFIG, "backend_type": "torch-script"},
    )
    mock_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            requested_backends=[BackendType.TORCH_SCRIPT, BackendType.ONNX],
        )

    assert result is not None
    assert result[1].endswith("pkg001")
    assert mock_load.call_args[1]["model_dir_or_weights_path"].endswith("pkg001")


def test_attempt_loading_model_from_offline_cache_rejects_unverifiable_constraints(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config=_OFFLINE_PACKAGE_CONFIG,
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(core, "attempt_loading_model_from_local_storage") as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            requested_quantization="fp16",
        )

    assert result is None
    mock_load.assert_not_called()


def test_attempt_loading_model_from_offline_cache_respects_access_manager(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    access_manager = MagicMock()
    access_manager.is_model_package_access_granted.return_value = False

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(core, "attempt_loading_model_from_local_storage") as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            model_access_manager=access_manager,
            api_key=None,
        )

    assert result is None
    access_manager.is_model_package_access_granted.assert_called_once_with(
        model_id=model_id,
        package_id="pkg001",
        api_key=None,
    )
    mock_load.assert_not_called()


def test_attempt_loading_model_from_offline_cache_skips_malformed_manifest(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={**_OFFLINE_PACKAGE_CONFIG, "task_type": ["not", "a", "string"]},
    )
    valid_package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg002",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    mock_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )

    assert result == (mock_model, valid_package_dir)
    mock_load.assert_called_once()


def test_attempt_loading_model_from_offline_cache_enforces_trust_provenance(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={**_OFFLINE_PACKAGE_CONFIG, "trusted_source": False},
    )
    mock_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        rejected = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )
        accepted = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            allow_untrusted_packages=True,
        )

    assert rejected is None
    assert accepted == (mock_model, package_dir)
    mock_load.assert_called_once()


def test_attempt_loading_model_from_offline_cache_rejects_legacy_manifest_even_with_opt_in(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            "model_architecture": "yolov8",
            "task_type": "object-detection",
            "backend_type": "onnx",
        },
    )
    mock_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        rejected = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )
        accepted = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            allow_untrusted_packages=True,
        )

    assert rejected is None
    assert accepted is None
    mock_load.assert_not_called()


def test_attempt_loading_model_from_offline_cache_rejects_dependencies_when_disabled(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "model_dependencies": [
                {
                    "name": "encoder",
                    "model_id": "dependency/1",
                    "model_package_id": "dependencyPackage",
                }
            ],
        },
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core.AutoModel, "from_pretrained"
    ) as dependency_load, mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as model_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            allow_loading_dependency_models=False,
        )

    assert result is None
    dependency_load.assert_not_called()
    model_load.assert_not_called()


def test_offline_cache_rejects_unknown_dependencies_when_disabled(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={**_OFFLINE_PACKAGE_CONFIG, "model_dependencies": None},
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core.AutoModel, "from_pretrained"
    ) as dependency_load, mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as model_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            allow_loading_dependency_models=False,
        )

    assert result is None
    dependency_load.assert_not_called()
    model_load.assert_not_called()


def test_attempt_loading_model_from_offline_cache_reconstructs_dependencies(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    dependency_package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id="dependency/1",
        package_id="dependencyPackage",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    dependency_manifest_hash = parse_model_config(
        os.path.join(dependency_package_dir, "model_config.json")
    ).manifest_content_hash
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "model_dependencies": [
                {
                    "name": "encoder",
                    "model_id": "dependency/1",
                    "model_package_id": "dependencyPackage",
                }
            ],
            "dependency_package_paths": [
                {
                    "name": "encoder",
                    "target_path": dependency_package_dir,
                    "cache_model_id": "dependency/1",
                    "canonical_model_id": "dependency/1",
                    "model_package_id": "dependencyPackage",
                    "package_manifest_hash": dependency_manifest_hash,
                }
            ],
        },
    )
    dependency_links_dir = os.path.join(package_dir, core.MODEL_DEPENDENCIES_SUB_DIR)
    os.makedirs(dependency_links_dir)
    os.symlink(
        dependency_package_dir,
        os.path.join(dependency_links_dir, "encoder"),
    )
    dependency_model = MagicMock()
    parent_model = MagicMock()

    def load_dependency(**kwargs):
        kwargs["point_model_directory"](dependency_package_dir)
        return dependency_model

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core.AutoModel, "from_pretrained", side_effect=load_dependency
    ) as dependency_load, mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=parent_model
    ) as model_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={"device": core.torch.device("cpu")},
        )

    assert result == (parent_model, package_dir)
    dependency_load.assert_called_once()
    assert dependency_load.call_args.kwargs["model_id_or_path"] == "dependency/1"
    assert dependency_load.call_args.kwargs["model_package_id"] == "dependencyPackage"
    assert model_load.call_args.kwargs["model_init_kwargs"][
        core.MODEL_DEPENDENCIES_KEY
    ] == {"encoder": dependency_model}


def test_raw_cache_rejects_dependency_resolved_to_unbound_package(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    bound_dependency_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id="dependency/1",
        package_id="dependencyPackageA",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    wrong_dependency_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id="dependency/1",
        package_id="dependencyPackageB",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    bound_manifest_hash = parse_model_config(
        os.path.join(bound_dependency_dir, "model_config.json")
    ).manifest_content_hash
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "model_dependencies": [
                {
                    "name": "encoder",
                    "model_id": "dependency/1",
                    "model_package_id": "dependencyPackage",
                }
            ],
            "dependency_package_paths": [
                {
                    "name": "encoder",
                    "target_path": bound_dependency_dir,
                    "cache_model_id": "dependency/1",
                    "canonical_model_id": "dependency/1",
                    "model_package_id": "dependencyPackageA",
                    "package_manifest_hash": bound_manifest_hash,
                }
            ],
        },
    )
    dependency_links_dir = os.path.join(package_dir, core.MODEL_DEPENDENCIES_SUB_DIR)
    os.makedirs(dependency_links_dir)
    os.symlink(
        bound_dependency_dir,
        os.path.join(dependency_links_dir, "encoder"),
    )

    def load_wrong_dependency(**kwargs):
        kwargs["point_model_directory"](wrong_dependency_dir)
        return MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core.AutoModel,
        "from_pretrained",
        side_effect=load_wrong_dependency,
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as parent_model_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )

    assert result is None
    parent_model_load.assert_not_called()


def test_attempt_loading_model_from_offline_cache_applies_cpu_default_quantization(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={**_OFFLINE_PACKAGE_CONFIG, "quantization": "fp16"},
    )
    fp32_package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg002",
        config={**_OFFLINE_PACKAGE_CONFIG, "quantization": "fp32"},
    )
    mock_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={"device": core.torch.device("cpu")},
        )

    assert result == (mock_model, fp32_package_dir)
    mock_load.assert_called_once()


def test_attempt_loading_model_from_offline_cache_requires_matching_constraints_hash(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "offline_compatibility_hash": "a" * 64,
        },
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(core, "attempt_loading_model_from_local_storage") as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            offline_compatibility_hash="b" * 64,
        )

    assert result is None
    mock_load.assert_not_called()


def test_attempt_loading_model_from_offline_cache_requires_matching_runtime(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "runtime_compatibility_hash": "a" * 64,
        },
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "_runtime_compatibility_hash", return_value="b" * 64
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
        )

    assert result is None
    mock_load.assert_not_called()


def test_from_pretrained_falls_back_to_offline_cache_on_retry_error(
    empty_local_dir: str,
) -> None:
    # given
    model_id = "test/1"
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "offline_compatibility_hash": (
                _offline_compatibility_hash_for_default_request(model_id)
            ),
        },
    )
    mock_model = MagicMock()

    # when
    with mock.patch.object(core, "ROBOFLOW_API_KEY", None), mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core,
        "get_model_from_provider",
        side_effect=RetryError(message="network down", help_url="https://help"),
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ) as mock_load:
        result = core.AutoModel.from_pretrained(
            model_id,
            api_key=None,
            use_auto_resolution_cache=False,
        )

    # then
    assert result is mock_model
    assert mock_load.call_args[1]["model_dir_or_weights_path"] == package_dir


def test_custom_cache_compatible_lookup_supports_no_key_offline_restart() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    compatible_cache_entry = AutoResolutionCacheEntry(
        model_id=model_id,
        cache_model_id="workspace/canonical-model/1",
        canonical_model_id="workspace/canonical-model/1",
        cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
        credential_hash=core._credential_hash(api_key="warmed-key"),
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        offline_compatibility_hash="c" * 64,
        trusted_source=True,
    )

    class ExistingCustomCache(AutoResolutionCache):
        def __init__(self) -> None:
            self.find_compatible_calls = 0

        def register(
            self,
            auto_negotiation_hash: str,
            cache_entry: AutoResolutionCacheEntry,
        ) -> None:
            pass

        def retrieve(
            self, auto_negotiation_hash: str
        ) -> Optional[AutoResolutionCacheEntry]:
            return None

        def invalidate(self, auto_negotiation_hash: str) -> None:
            pass

        def find_compatible(
            self, offline_compatibility_hash: str
        ) -> Optional[tuple[str, AutoResolutionCacheEntry]]:
            self.find_compatible_calls += 1
            return "old-api-key-hash", compatible_cache_entry

    auto_resolution_cache = ExistingCustomCache()

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core,
        "attempt_loading_model_with_auto_load_cache",
        side_effect=[None, cached_model],
    ) as cached_load, mock.patch.object(
        core, "get_model_from_provider"
    ) as provider:
        result = core.AutoModel.from_pretrained(
            model_id,
            api_key=None,
            auto_resolution_cache=auto_resolution_cache,
        )

    assert result is cached_model
    assert cached_load.call_count == 2
    assert cached_load.call_args_list[1].kwargs["auto_negotiation_hash"] == (
        "old-api-key-hash"
    )
    assert auto_resolution_cache.find_compatible_calls == 1
    provider.assert_not_called()


def test_no_key_offline_restart_loads_one_canonically_attributed_candidate(
    empty_local_dir: str,
) -> None:
    requested_model_id = "shared-alias/1"
    canonical_model_id = "tenant-a/canonical/1"
    package_id = "package"
    expected_model = MagicMock()
    request_compatibility_hash = _offline_compatibility_hash_for_default_request(
        requested_model_id
    )

    class CachedModel:
        @classmethod
        def from_pretrained(cls, model_dir_or_weights_path, **kwargs):
            assert model_dir_or_weights_path == package_dir
            return expected_model

    with mock.patch.object(model_cache_paths, "INFERENCE_HOME", empty_local_dir):
        package_dir = generate_model_package_cache_path(
            model_id=canonical_model_id,
            package_id=package_id,
        )
        os.makedirs(package_dir)
        manifest_path = os.path.join(package_dir, "model_config.json")
        manifest_hash = dump_model_config_for_offline_use(
            config_path=manifest_path,
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            file_lock_acquire_timeout=1,
            model_id=canonical_model_id,
            canonical_model_id=canonical_model_id,
            trusted_source=True,
            model_dependencies=[],
            runtime_compatibility_hash=core._runtime_compatibility_hash(
                runtime_x_ray=core.x_ray_runtime_environment()
            ),
            offline_compatibility_hash="a" * 64,
        )
        cache_entry = AutoResolutionCacheEntry(
            model_id=requested_model_id,
            cache_model_id=canonical_model_id,
            canonical_model_id=canonical_model_id,
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            credential_hash=core._credential_hash(api_key="warmed-key"),
            model_package_id=package_id,
            resolved_files=[manifest_path],
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=datetime.now(),
            offline_compatibility_hash=request_compatibility_hash,
            trusted_source=True,
            package_manifest_hash=manifest_hash,
        )

        class WarmedCache(AutoResolutionCache):
            def register(
                self,
                auto_negotiation_hash: str,
                cache_entry: AutoResolutionCacheEntry,
            ) -> None:
                pass

            def retrieve(
                self, auto_negotiation_hash: str
            ) -> Optional[AutoResolutionCacheEntry]:
                if auto_negotiation_hash == "warmed-keyed-hash":
                    return cache_entry
                return None

            def invalidate(self, auto_negotiation_hash: str) -> None:
                pass

            def find_compatible_candidates(
                self,
                offline_compatibility_hash: str,
            ) -> list[tuple[str, AutoResolutionCacheEntry]]:
                return [("warmed-keyed-hash", cache_entry)]

        with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
            core, "ROBOFLOW_API_KEY", None
        ), mock.patch.object(
            core, "model_provider_requires_network", return_value=True
        ), mock.patch.object(
            core, "resolve_model_class", return_value=CachedModel
        ), mock.patch.object(
            core, "get_model_from_provider"
        ) as provider:
            result = core.AutoModel.from_pretrained(
                requested_model_id,
                api_key=None,
                auto_resolution_cache=WarmedCache(),
            )

    assert result is expected_model
    provider.assert_not_called()


def test_from_pretrained_tries_older_compatible_entry_after_newest_fails() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    newest_entry = AutoResolutionCacheEntry(
        model_id=model_id,
        cache_model_id="workspace/canonical-model/1",
        canonical_model_id="workspace/canonical-model/1",
        cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
        credential_hash=core._credential_hash(api_key="warmed-key"),
        model_package_id="newest-package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        offline_compatibility_hash="c" * 64,
        trusted_source=True,
    )
    older_entry = newest_entry.model_copy(update={"model_package_id": "older-package"})
    auto_resolution_cache = MagicMock()
    attempted_hashes = []

    def cache_load_side_effect(**kwargs):
        cache_hash = kwargs["auto_negotiation_hash"]
        attempted_hashes.append(cache_hash)
        if cache_hash == "older-api-key-hash":
            return cached_model
        return None

    def find_candidates(**kwargs):
        exact_hash = attempted_hashes[0]
        return [
            (exact_hash, newest_entry),
            ("newest-api-key-hash", newest_entry),
            ("older-api-key-hash", older_entry),
        ]

    auto_resolution_cache.find_compatible_candidates.side_effect = find_candidates
    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core,
        "attempt_loading_model_with_auto_load_cache",
        side_effect=cache_load_side_effect,
    ), mock.patch.object(
        core, "get_model_from_provider"
    ) as provider:
        result = core.AutoModel.from_pretrained(
            model_id,
            api_key=None,
            auto_resolution_cache=auto_resolution_cache,
        )

    assert result is cached_model
    exact_hash = attempted_hashes[0]
    assert attempted_hashes == [
        exact_hash,
        "newest-api-key-hash",
        "older-api-key-hash",
    ]
    auto_resolution_cache.find_compatible_candidates.assert_called_once()
    provider.assert_not_called()


def test_from_pretrained_no_key_rejects_multiple_canonical_identities() -> None:
    requested_model_id = "shared-alias/1"

    def entry(canonical_model_id: str) -> AutoResolutionCacheEntry:
        return AutoResolutionCacheEntry(
            model_id=requested_model_id,
            cache_model_id=canonical_model_id,
            canonical_model_id=canonical_model_id,
            cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
            model_package_id="package",
            resolved_files=[],
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=datetime.now(),
            offline_compatibility_hash="c" * 64,
            trusted_source=True,
        )

    auto_resolution_cache = MagicMock()
    auto_resolution_cache.find_compatible_candidates.return_value = [
        ("a" * 64, entry("tenant-a/canonical/1")),
        ("b" * 64, entry("tenant-b/canonical/1")),
    ]

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as cache_load, mock.patch.object(
        core,
        "attempt_loading_model_from_offline_cache",
        return_value=(MagicMock(), "/direct-package"),
    ) as raw_cache_load:
        with pytest.raises(ModelRetrievalError):
            core.AutoModel.from_pretrained(
                requested_model_id,
                api_key=None,
                auto_resolution_cache=auto_resolution_cache,
            )

    assert cache_load.call_count == 1
    raw_cache_load.assert_not_called()


def test_from_pretrained_no_key_rejects_direct_package_conflicting_with_alias(
    empty_local_dir: str,
) -> None:
    requested_model_id = "shared-alias/1"
    _write_offline_package(
        inference_home=empty_local_dir,
        model_id=requested_model_id,
        package_id="directpkg",
        config=_OFFLINE_PACKAGE_CONFIG,
    )
    alias_entry = AutoResolutionCacheEntry(
        model_id=requested_model_id,
        cache_model_id="tenant-b/canonical/1",
        canonical_model_id="tenant-b/canonical/1",
        cache_attribution_version=core.CACHE_ATTRIBUTION_VERSION,
        credential_hash=core._credential_hash(api_key="tenant-b-key"),
        model_package_id="aliaspkg",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        offline_compatibility_hash="c" * 64,
        trusted_source=True,
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.find_compatible_candidates.return_value = [
        ("a" * 64, alias_entry)
    ]

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as cache_load, mock.patch.object(
        core,
        "attempt_loading_model_from_offline_cache",
        return_value=(MagicMock(), "/direct-package"),
    ) as raw_cache_load:
        with pytest.raises(ModelRetrievalError):
            core.AutoModel.from_pretrained(
                requested_model_id,
                api_key=None,
                auto_resolution_cache=auto_resolution_cache,
            )

    assert cache_load.call_count == 1
    raw_cache_load.assert_not_called()


def test_from_pretrained_no_key_rejects_unattributed_compatible_entry() -> None:
    requested_model_id = "shared-alias/1"
    legacy_entry = AutoResolutionCacheEntry(
        model_id=requested_model_id,
        model_package_id="package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        offline_compatibility_hash="c" * 64,
        trusted_source=True,
    )
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.find_compatible_candidates.return_value = [
        ("a" * 64, legacy_entry)
    ]

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as cache_load, mock.patch.object(
        core, "attempt_loading_model_from_offline_cache", return_value=None
    ):
        with pytest.raises(ModelRetrievalError):
            core.AutoModel.from_pretrained(
                requested_model_id,
                api_key=None,
                auto_resolution_cache=auto_resolution_cache,
            )

    assert cache_load.call_count == 1


def test_from_pretrained_rotated_api_keys_do_not_use_credential_free_fallback() -> None:
    model_id = "workspace/model/1"
    auto_resolution_cache = MagicMock()

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as exact_cache_load, mock.patch.object(
        core,
        "attempt_loading_model_from_offline_cache",
        return_value=None,
    ) as raw_cache_load:
        with pytest.raises(ModelRetrievalError):
            core.AutoModel.from_pretrained(
                model_id,
                api_key="first-api-key",
                auto_resolution_cache=auto_resolution_cache,
            )
        with pytest.raises(ModelRetrievalError):
            core.AutoModel.from_pretrained(
                model_id,
                api_key="second-api-key",
                auto_resolution_cache=auto_resolution_cache,
            )

    exact_hashes = [
        cache_call.kwargs["auto_negotiation_hash"]
        for cache_call in exact_cache_load.call_args_list
    ]
    assert exact_hashes[0] != exact_hashes[1]
    auto_resolution_cache.find_compatible_candidates.assert_not_called()
    raw_cache_load.assert_not_called()


@pytest.mark.parametrize("api_key", [None, "local"])
def test_from_pretrained_env_api_key_does_not_use_credential_free_fallback(
    api_key: Optional[str],
) -> None:
    model_id = "workspace/model/1"
    auto_resolution_cache = MagicMock()

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as exact_cache_load, mock.patch.object(
        core,
        "attempt_loading_model_from_offline_cache",
        return_value=None,
    ) as raw_cache_load:
        for configured_api_key in ("first-env-key", "second-env-key"):
            with mock.patch.object(core, "ROBOFLOW_API_KEY", configured_api_key):
                with pytest.raises(ModelRetrievalError):
                    core.AutoModel.from_pretrained(
                        model_id,
                        api_key=api_key,
                        auto_resolution_cache=auto_resolution_cache,
                    )

    exact_hashes = [
        cache_call.kwargs["auto_negotiation_hash"]
        for cache_call in exact_cache_load.call_args_list
    ]
    assert len(exact_hashes) == 2
    assert exact_hashes[0] != exact_hashes[1]
    auto_resolution_cache.find_compatible_candidates.assert_not_called()
    raw_cache_load.assert_not_called()


def test_max_package_loading_attempts_is_bound_into_cache_identity() -> None:
    auto_resolution_cache = MagicMock(spec=AutoResolutionCache)
    auto_resolution_cache.find_compatible_candidates.return_value = []

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as exact_cache_load, mock.patch.object(
        core, "attempt_loading_model_from_offline_cache", return_value=None
    ):
        for max_attempts in (1, 2):
            with pytest.raises(ModelRetrievalError):
                core.AutoModel.from_pretrained(
                    "workspace/model/1",
                    max_package_loading_attempts=max_attempts,
                    auto_resolution_cache=auto_resolution_cache,
                )

    exact_hashes = [
        call.kwargs["auto_negotiation_hash"] for call in exact_cache_load.call_args_list
    ]
    compatibility_hashes = [
        call.kwargs["offline_compatibility_hash"]
        for call in auto_resolution_cache.find_compatible_candidates.call_args_list
    ]
    assert len(set(exact_hashes)) == 2
    assert len(set(compatibility_hashes)) == 2


def test_from_pretrained_uses_effective_env_key_for_strict_exact_cache_hit() -> None:
    model_id = "shared-alias/1"
    canonical_model_id = "tenant/canonical/1"
    package_id = "package"
    effective_api_key = "configured-env-key"
    expected_model = MagicMock()
    model_access_manager = MagicMock(spec=LiberalModelAccessManager)
    model_access_manager.is_model_access_forbidden.return_value = False
    model_access_manager.retrieve_model_instance.return_value = None
    model_access_manager.is_model_package_access_granted.side_effect = (
        lambda model_id, package_id, api_key: api_key == effective_api_key
    )

    def strict_exact_cache_load(**kwargs):
        if model_access_manager.is_model_package_access_granted(
            model_id=canonical_model_id,
            package_id=package_id,
            api_key=kwargs["api_key"],
        ):
            return expected_model
        return None

    auto_resolution_cache = MagicMock(spec=AutoResolutionCache)
    with mock.patch.object(
        core, "ROBOFLOW_API_KEY", effective_api_key
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core,
        "attempt_loading_model_with_auto_load_cache",
        side_effect=strict_exact_cache_load,
    ) as exact_cache_load, mock.patch.object(
        core, "get_model_from_provider"
    ) as provider:
        implicit_result = core.AutoModel.from_pretrained(
            model_id,
            api_key=None,
            auto_resolution_cache=auto_resolution_cache,
            model_access_manager=model_access_manager,
        )
        explicit_result = core.AutoModel.from_pretrained(
            model_id,
            api_key=effective_api_key,
            auto_resolution_cache=auto_resolution_cache,
            model_access_manager=model_access_manager,
        )

    assert implicit_result is expected_model
    assert explicit_result is expected_model
    exact_calls = exact_cache_load.call_args_list
    assert [cache_call.kwargs["api_key"] for cache_call in exact_calls] == [
        effective_api_key,
        effective_api_key,
    ]
    assert (
        exact_calls[0].kwargs["auto_negotiation_hash"]
        == exact_calls[1].kwargs["auto_negotiation_hash"]
    )
    assert all(
        cache_call.kwargs["api_key"] == effective_api_key
        for cache_call in model_access_manager.retrieve_model_instance.call_args_list
    )
    provider.assert_not_called()


def test_from_pretrained_returns_preloaded_model_with_verified_storage_pointer(
    empty_local_dir: str,
) -> None:
    package_dir = os.path.join(empty_local_dir, "owlv2-package")
    os.makedirs(package_dir)
    preloaded_model = MagicMock()
    core._record_model_package_path(
        model=preloaded_model,
        package_dir=package_dir,
    )

    class PreloadedModelAccessManager(LiberalModelAccessManager):
        def retrieve_model_instance(
            self,
            model_id: str,
            package_id: Optional[str],
            api_key: Optional[str],
            loading_parameter_digest: Optional[str],
        ):
            return preloaded_model

    point_model_directory = MagicMock()
    with mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(core, "get_model_from_provider") as provider:
        result = core.AutoModel.from_pretrained(
            "owlv2/owlv2-base-patch16",
            api_key="api-key",
            auto_resolution_cache=MagicMock(spec=AutoResolutionCache),
            model_access_manager=PreloadedModelAccessManager(),
            point_model_directory=point_model_directory,
        )

    assert result is preloaded_model
    point_model_directory.assert_called_once_with(os.path.realpath(package_dir))
    provider.assert_not_called()


@pytest.mark.parametrize("api_key", [None, "local"])
def test_from_pretrained_effective_env_key_is_enforced_by_strict_deny_manager(
    api_key: Optional[str],
) -> None:
    effective_api_key = "denied-env-key"
    model_access_manager = MagicMock(spec=LiberalModelAccessManager)
    model_access_manager.is_model_access_forbidden.side_effect = (
        lambda model_id, api_key: api_key == effective_api_key
    )

    with mock.patch.object(
        core, "ROBOFLOW_API_KEY", effective_api_key
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache"
    ) as exact_cache_load, mock.patch.object(
        core, "get_model_from_provider"
    ) as provider:
        with pytest.raises(UnauthorizedModelAccessError):
            core.AutoModel.from_pretrained(
                "workspace/model/1",
                api_key=api_key,
                model_access_manager=model_access_manager,
            )

    model_access_manager.is_model_access_forbidden.assert_called_once_with(
        model_id="workspace/model/1",
        api_key=effective_api_key,
    )
    exact_cache_load.assert_not_called()
    provider.assert_not_called()


def test_from_pretrained_propagates_effective_env_key_to_provider_and_callbacks() -> (
    None
):
    requested_model_id = "shared-alias/1"
    canonical_model_id = "tenant/canonical/1"
    package_id = "package"
    effective_api_key = "configured-env-key"
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.ONNX,
        package_artefacts=[],
        trusted_source=True,
    )
    metadata = ModelMetadata(
        model_id=canonical_model_id,
        model_architecture="yolov8",
        model_packages=[package],
        task_type="object-detection",
    )
    expected_model = MagicMock()
    model_access_manager = MagicMock(spec=LiberalModelAccessManager)
    model_access_manager.is_model_access_forbidden.return_value = False
    model_access_manager.retrieve_model_instance.return_value = None
    auto_resolution_cache = MagicMock(spec=AutoResolutionCache)
    cache_callbacks = {}

    def build_auto_resolution_cache(**kwargs):
        cache_callbacks["on_file_created"] = kwargs["on_file_created"]
        return auto_resolution_cache

    def initialize_with_callbacks(**kwargs):
        kwargs["on_file_created"]("/cache/model-file")
        kwargs["on_file_renamed"]("/cache/old", "/cache/new")
        kwargs["on_symlink_created"]("/cache/target", "/cache/link")
        cache_callbacks["on_file_created"](
            "/cache/auto-resolution.json",
            canonical_model_id,
            package_id,
        )
        return expected_model, "/cache/model-package"

    with mock.patch.object(
        core, "ROBOFLOW_API_KEY", effective_api_key
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core,
        "BaseAutoLoadMetadataCache",
        side_effect=build_auto_resolution_cache,
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ), mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ) as provider, mock.patch.object(
        core, "negotiate_model_packages", return_value=[package]
    ), mock.patch.object(
        core, "initialize_model", side_effect=initialize_with_callbacks
    ):
        result = core.AutoModel.from_pretrained(
            requested_model_id,
            api_key=None,
            model_access_manager=model_access_manager,
        )

    expected_access_identifiers = AccessIdentifiers(
        model_id=canonical_model_id,
        package_id=package_id,
        api_key=effective_api_key,
    )
    assert result is expected_model
    provider.assert_called_once_with(
        provider="roboflow",
        model_id=requested_model_id,
        api_key=effective_api_key,
        weights_provider_extra_query_params=None,
        weights_provider_extra_headers=None,
    )
    model_access_manager.is_model_access_forbidden.assert_called_once_with(
        model_id=requested_model_id,
        api_key=effective_api_key,
    )
    assert len(model_access_manager.retrieve_model_instance.call_args_list) == 2
    assert all(
        retrieve_call.kwargs["api_key"] == effective_api_key
        for retrieve_call in model_access_manager.retrieve_model_instance.call_args_list
    )
    model_access_manager.on_model_package_access_granted.assert_called_once_with(
        expected_access_identifiers
    )
    model_access_manager.on_file_created.assert_has_calls(
        [
            call(
                "/cache/model-file",
                access_identifiers=expected_access_identifiers,
            ),
            call(
                file_path="/cache/auto-resolution.json",
                access_identifiers=expected_access_identifiers,
            ),
        ]
    )
    model_access_manager.on_file_renamed.assert_called_once_with(
        "/cache/old",
        "/cache/new",
        access_identifiers=expected_access_identifiers,
    )
    model_access_manager.on_symlink_created.assert_called_once_with(
        "/cache/target",
        "/cache/link",
        access_identifiers=expected_access_identifiers,
    )
    model_access_manager.on_model_loaded.assert_called_once_with(
        model=expected_model,
        access_identifiers=expected_access_identifiers,
        model_storage_path="/cache/model-package",
    )


def test_from_pretrained_preserves_original_key_for_custom_provider() -> None:
    model_id = "custom-model"
    metadata = ModelMetadata(
        model_id=model_id,
        model_architecture="yolov8",
        model_packages=[],
        task_type="object-detection",
    )
    expected_model = MagicMock()
    model_access_manager = MagicMock(spec=LiberalModelAccessManager)
    model_access_manager.is_model_access_forbidden.return_value = False
    model_access_manager.retrieve_model_instance.return_value = None

    with mock.patch.object(
        core, "ROBOFLOW_API_KEY", "unrelated-roboflow-key"
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=False
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ), mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ) as provider, mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ), mock.patch.object(
        core,
        "attempt_loading_matching_model_packages",
        return_value=expected_model,
    ) as matching_load:
        result = core.AutoModel.from_pretrained(
            model_id,
            weights_provider="custom",
            api_key=None,
            auto_resolution_cache=MagicMock(spec=AutoResolutionCache),
            model_access_manager=model_access_manager,
        )

    assert result is expected_model
    model_access_manager.is_model_access_forbidden.assert_called_once_with(
        model_id=model_id,
        api_key=None,
    )
    provider.assert_called_once_with(
        provider="custom",
        model_id=model_id,
        api_key=None,
        weights_provider_extra_query_params=None,
        weights_provider_extra_headers=None,
    )
    assert matching_load.call_args.kwargs["api_key"] is None


@pytest.mark.parametrize("model_id_or_path", [None, 123, "", "   ", "bad\0id"])
def test_from_pretrained_rejects_invalid_remote_model_identity_before_cache_work(
    model_id_or_path: object,
) -> None:
    model_access_manager = MagicMock(spec=LiberalModelAccessManager)

    with mock.patch.object(
        core, "model_provider_requires_network"
    ) as provider_requires_network, mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache"
    ) as cache_load, mock.patch.object(
        core, "get_model_from_provider"
    ) as provider:
        with pytest.raises(
            InvalidParameterError,
            match="Remote model identity must be a non-empty string",
        ):
            core.AutoModel.from_pretrained(
                model_id_or_path,
                model_access_manager=model_access_manager,
            )

    provider_requires_network.assert_not_called()
    model_access_manager.is_model_access_forbidden.assert_not_called()
    cache_load.assert_not_called()
    provider.assert_not_called()


def test_from_pretrained_preserves_original_key_for_existing_local_path(
    empty_local_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_path = "   "
    os.makedirs(os.path.join(empty_local_dir, local_path))
    monkeypatch.chdir(empty_local_dir)
    expected_model = MagicMock()
    model_access_manager = MagicMock(spec=LiberalModelAccessManager)
    model_access_manager.is_model_access_forbidden.return_value = False

    with mock.patch.object(
        core, "ROBOFLOW_API_KEY", "unrelated-roboflow-key"
    ), mock.patch.object(
        core, "model_provider_requires_network"
    ) as provider_requires_network, mock.patch.object(
        core,
        "attempt_loading_model_from_local_storage",
        return_value=expected_model,
    ) as local_load:
        result = core.AutoModel.from_pretrained(
            local_path,
            api_key=None,
            model_access_manager=model_access_manager,
        )

    assert result is expected_model
    provider_requires_network.assert_not_called()
    model_access_manager.is_model_access_forbidden.assert_called_once_with(
        model_id=local_path,
        api_key=None,
    )
    local_load.assert_called_once()


def test_from_pretrained_preserves_existing_pathlike_local_path(
    empty_local_dir: str,
) -> None:
    local_path = Path(empty_local_dir) / "local-model"
    local_path.mkdir()
    expected_model = MagicMock()

    with mock.patch.object(
        core,
        "attempt_loading_model_from_local_storage",
        return_value=expected_model,
    ) as local_load:
        result = core.AutoModel.from_pretrained(local_path)

    assert result is expected_model
    local_load.assert_called_once()
    assert local_load.call_args.kwargs["model_dir_or_weights_path"] == str(local_path)


def test_from_pretrained_hash_normalizes_semantically_equivalent_choice_sets() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.find_compatible_candidates.return_value = []

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as exact_cache_load, mock.patch.object(
        core,
        "attempt_loading_model_from_offline_cache",
        return_value=(cached_model, "/cached/model"),
    ) as raw_cache_load:
        first = core.AutoModel.from_pretrained(
            model_id,
            backend="ONNX",
            quantization="fp16",
            auto_resolution_cache=auto_resolution_cache,
        )
        second = core.AutoModel.from_pretrained(
            model_id,
            backend=["onnx", BackendType.ONNX, "ONNX"],
            quantization=["fp16", Quantization.FP16, "fp16"],
            auto_resolution_cache=auto_resolution_cache,
        )

    assert first is cached_model
    assert second is cached_model
    exact_hashes = [
        cache_call.kwargs["auto_negotiation_hash"]
        for cache_call in exact_cache_load.call_args_list
    ]
    assert exact_hashes[0] == exact_hashes[1]
    offline_hashes = [
        cache_call.kwargs["offline_compatibility_hash"]
        for cache_call in raw_cache_load.call_args_list
    ]
    assert offline_hashes[0] == offline_hashes[1]


def test_from_pretrained_hash_binds_dependency_overrides_and_forwarded_kwargs() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.find_compatible_candidates.return_value = []

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ) as exact_cache_load, mock.patch.object(
        core,
        "attempt_loading_model_from_offline_cache",
        return_value=(cached_model, "/cached/model"),
    ) as raw_cache_load:
        core.AutoModel.from_pretrained(
            model_id,
            dependency_models_params={"encoder": {"backend": "onnx"}},
            auto_resolution_cache=auto_resolution_cache,
        )
        core.AutoModel.from_pretrained(
            model_id,
            dependency_models_params={"encoder": {"backend": "torch"}},
            auto_resolution_cache=auto_resolution_cache,
        )
        core.AutoModel.from_pretrained(
            model_id,
            dependency_models_params={"encoder": {"backend": "onnx"}},
            owlv2_enforce_model_compilation=True,
            auto_resolution_cache=auto_resolution_cache,
        )

    exact_hashes = [
        cache_call.kwargs["auto_negotiation_hash"]
        for cache_call in exact_cache_load.call_args_list
    ]
    offline_hashes = [
        cache_call.kwargs["offline_compatibility_hash"]
        for cache_call in raw_cache_load.call_args_list
    ]
    assert len(set(exact_hashes)) == 3
    assert len(set(offline_hashes)) == 3


def test_online_dependency_load_does_not_mutate_input_params() -> None:
    model_id = "workspace/model/1"
    dependency_model = MagicMock()
    parent_model = MagicMock()
    caller_owned_params = {
        "encoder": {
            "device": "cpu",
            "custom_option": "keep-me",
        }
    }
    model_metadata = ModelMetadata(
        model_id=model_id,
        model_architecture="yolov8",
        model_packages=[],
        task_type="object-detection",
        model_dependencies=[
            ModelDependency(
                name="encoder",
                model_id="dependency/1",
                model_package_id="dependencyPackage",
            )
        ],
    )
    outer_load = core.AutoModel.from_pretrained

    with mock.patch.object(
        core, "get_model_from_provider", return_value=model_metadata
    ), mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ), mock.patch.object(
        core,
        "attempt_loading_matching_model_packages",
        return_value=parent_model,
    ) as matching_load, mock.patch.object(
        core.AutoModel, "from_pretrained", return_value=dependency_model
    ) as dependency_load:
        result = outer_load(
            model_id,
            use_auto_resolution_cache=False,
            dependency_models_params=caller_owned_params,
        )

    assert result is parent_model
    assert caller_owned_params == {
        "encoder": {
            "device": "cpu",
            "custom_option": "keep-me",
        }
    }
    dependency_load.assert_called_once()
    assert dependency_load.call_args.kwargs["custom_option"] == "keep-me"
    assert matching_load.call_args.kwargs["model_dependencies_instances"] == {
        "encoder": dependency_model
    }


def test_online_env_api_key_propagates_to_dependency_load() -> None:
    model_id = "workspace/model/1"
    effective_api_key = "configured-env-key"
    dependency_model = MagicMock()
    parent_model = MagicMock()
    model_metadata = ModelMetadata(
        model_id=model_id,
        model_architecture="yolov8",
        model_packages=[],
        task_type="object-detection",
        model_dependencies=[
            ModelDependency(
                name="encoder",
                model_id="dependency/1",
                model_package_id="dependencyPackage",
            )
        ],
    )
    outer_load = core.AutoModel.from_pretrained

    with mock.patch.object(
        core, "ROBOFLOW_API_KEY", effective_api_key
    ), mock.patch.object(
        core, "model_provider_requires_network", return_value=True
    ), mock.patch.object(
        core, "get_model_from_provider", return_value=model_metadata
    ) as provider, mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ), mock.patch.object(
        core,
        "attempt_loading_matching_model_packages",
        return_value=parent_model,
    ) as matching_load, mock.patch.object(
        core.AutoModel, "from_pretrained", return_value=dependency_model
    ) as dependency_load:
        result = outer_load(
            model_id,
            api_key=None,
            use_auto_resolution_cache=False,
        )

    assert result is parent_model
    assert provider.call_args.kwargs["api_key"] == effective_api_key
    assert dependency_load.call_args.kwargs["api_key"] == effective_api_key
    assert matching_load.call_args.kwargs["api_key"] == effective_api_key


def test_online_resolution_preserves_provider_canonical_model_id() -> None:
    requested_model_id = "shared-alias/1"
    canonical_model_id = "tenant-a/canonical/1"
    model_metadata = ModelMetadata(
        model_id=canonical_model_id,
        model_architecture="yolov8",
        model_packages=[],
        task_type="object-detection",
    )
    expected_model = MagicMock()

    with mock.patch.object(
        core, "get_model_from_provider", return_value=model_metadata
    ), mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ), mock.patch.object(
        core,
        "attempt_loading_matching_model_packages",
        return_value=expected_model,
    ) as matching_load:
        result = core.AutoModel.from_pretrained(
            requested_model_id,
            api_key="tenant-a-key",
            use_auto_resolution_cache=False,
        )

    assert result is expected_model
    assert matching_load.call_args.kwargs["model_id"] == canonical_model_id
    assert matching_load.call_args.kwargs["requested_model_id"] == requested_model_id


@pytest.mark.parametrize("canonical_model_id", [None, "", "   "])
def test_online_resolution_rejects_invalid_provider_canonical_model_id(
    canonical_model_id: Optional[str],
) -> None:
    model_metadata = ModelMetadata(
        model_id=canonical_model_id,
        model_architecture="yolov8",
        model_packages=[],
        task_type="object-detection",
    )

    with mock.patch.object(
        core, "get_model_from_provider", return_value=model_metadata
    ), mock.patch.object(core, "negotiate_model_packages") as negotiate:
        with pytest.raises(
            CorruptedModelPackageError,
            match="invalid canonical model ID",
        ):
            core.AutoModel.from_pretrained(
                "shared-alias/1",
                api_key="tenant-key",
                use_auto_resolution_cache=False,
            )

    negotiate.assert_not_called()


def test_from_pretrained_reraises_retry_error_when_no_offline_cache(
    empty_local_dir: str,
) -> None:
    # when / then
    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core,
        "get_model_from_provider",
        side_effect=RetryError(message="network down", help_url="https://help"),
    ):
        with pytest.raises(RetryError):
            core.AutoModel.from_pretrained(
                "nonexistent/1",
                api_key="test-key",
                use_auto_resolution_cache=False,
            )


def test_from_pretrained_in_offline_mode_loads_from_cache_without_provider_call(
    empty_local_dir: str,
) -> None:
    # given
    model_id = "test/1"
    package_dir = _write_offline_package(
        inference_home=empty_local_dir,
        model_id=model_id,
        package_id="pkg001",
        config={
            **_OFFLINE_PACKAGE_CONFIG,
            "offline_compatibility_hash": (
                _offline_compatibility_hash_for_default_request(model_id)
            ),
        },
    )
    mock_model = MagicMock()
    mock_provider = MagicMock()

    # when
    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "ROBOFLOW_API_KEY", None
    ), mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "get_model_from_provider", mock_provider
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ):
        result = core.AutoModel.from_pretrained(
            model_id,
            api_key=None,
            local_files_only=False,
            use_auto_resolution_cache=False,
        )

    # then
    assert result is mock_model
    mock_provider.assert_not_called()


def test_offline_library_load_cannot_override_local_files_only() -> None:
    class HuggingFaceModel:
        @classmethod
        def from_pretrained(
            cls,
            model_name_or_path,
            local_files_only=False,
            **kwargs,
        ):
            pass

    with mock.patch.object(core, "OFFLINE_MODE", True):
        result = core._prepare_library_model_init_kwargs(
            model_class=HuggingFaceModel,
            model_init_kwargs={"local_files_only": False, "device": "cpu"},
        )

    assert result == {"local_files_only": True, "device": "cpu"}


def test_offline_library_load_does_not_inject_hugging_face_argument() -> None:
    class OnnxModel:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            pass

    model_init_kwargs = {"device": "cpu"}
    with mock.patch.object(core, "OFFLINE_MODE", True):
        result = core._prepare_library_model_init_kwargs(
            model_class=OnnxModel,
            model_init_kwargs=model_init_kwargs,
        )

    assert result is model_init_kwargs


def test_from_pretrained_in_offline_mode_raises_when_no_cache(
    empty_local_dir: str,
) -> None:
    # when / then
    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ):
        with pytest.raises(ModelRetrievalError):
            core.AutoModel.from_pretrained(
                "nonexistent/1",
                api_key="test-key",
                use_auto_resolution_cache=False,
            )


def test_from_pretrained_uses_custom_local_provider_in_offline_mode() -> None:
    """The full auto-loader path permits a registered local provider offline."""
    # given
    model_id = "local-model"
    model_metadata = ModelMetadata(
        model_id=model_id,
        model_architecture="yolov8",
        model_packages=[],
        task_type="object-detection",
    )
    local_provider = MagicMock(return_value=model_metadata)
    expected_model = MagicMock()

    # when
    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        weights_providers_core, "OFFLINE_MODE", True
    ), mock.patch.dict(
        weights_providers_core.WEIGHTS_PROVIDERS, {}, clear=True
    ), mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ), mock.patch.object(
        core, "attempt_loading_matching_model_packages", return_value=expected_model
    ):
        weights_providers_core.register_model_provider("local", local_provider)
        result = core.AutoModel.from_pretrained(
            model_id,
            weights_provider="local",
            use_auto_resolution_cache=False,
        )

    # then
    assert result is expected_model
    local_provider.assert_called_once_with(
        model_id,
        None,
        weights_provider_extra_query_params=None,
        weights_provider_extra_headers=None,
    )


def test_dump_model_config_for_offline_use_persists_model_id(
    empty_local_dir: str,
) -> None:
    # given
    config_path = os.path.join(empty_local_dir, "model_config.json")

    # when
    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=1,
        model_id="my-workspace/my-project/3",
        canonical_model_id="my-workspace/my-project/3",
    )

    # then
    with open(config_path) as f:
        content = json.load(f)
    assert content["model_id"] == "my-workspace/my-project/3"
    assert content["task_type"] == "object-detection"


def test_auto_resolution_cache_entries_do_not_expire_in_offline_mode(
    empty_local_dir: str,
) -> None:
    # given
    from inference_models.models.auto_loaders import auto_resolution_cache

    cache = auto_resolution_cache.BaseAutoLoadMetadataCache(file_lock_acquire_timeout=1)
    entry = AutoResolutionCacheEntry(
        model_id="some/1",
        model_package_id="pkg001",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime(2020, 1, 1),
    )

    # when
    with mock.patch.object(auto_resolution_cache, "INFERENCE_HOME", empty_local_dir):
        cache.register(auto_negotiation_hash="some-hash", cache_entry=entry)
        with mock.patch.object(auto_resolution_cache, "OFFLINE_MODE", True):
            result_offline = cache.retrieve(auto_negotiation_hash="some-hash")
        result_online = cache.retrieve(auto_negotiation_hash="some-hash")

    # then - expired entry survives in OFFLINE_MODE, expires otherwise
    assert result_offline is not None
    assert result_offline.model_id == "some/1"
    assert result_online is None
