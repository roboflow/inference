import inspect
import json
import os.path
from datetime import datetime
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from inference_models import ClassificationPrediction
from inference_models.errors import (
    CorruptedModelPackageError,
    InsecureModelIdentifierError,
    ModelLoadingError,
    ModelRetrievalError,
    RetryError,
)
from inference_models.models.auto_loaders import core, model_cache_paths
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCache,
    AutoResolutionCacheEntry,
)
from inference_models.models.auto_loaders.core import (
    attempt_loading_model_from_local_storage,
    attempt_loading_model_from_offline_cache,
    attempt_loading_model_with_auto_load_cache,
    attempt_loading_matching_model_packages,
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
    BackendType,
    InferenceModelConfig,
)
from inference_models.weights_providers import core as weights_providers_core
from inference_models.weights_providers.entities import (
    ModelDependency,
    ModelMetadata,
    ModelPackageMetadata,
    Quantization,
    RecommendedParameters,
)


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
    assert result == InferenceModelConfig(
        model_architecture="some",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        model_module="model.py",
        model_class="MyClassificationModel",
    )


@mock.patch.object(model_cache_paths, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path() -> None:
    # when
    result = generate_model_package_cache_path(
        model_id="my-model", package_id="mypackage"
    )

    # then
    assert result == "/some/models-cache/my-model-6fa11b0c/mypackage"


@mock.patch.object(model_cache_paths, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_at_front() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="/my-model", package_id="mypackage"
    )

    # then
    assert result == "/some/models-cache/-my-model-9651d483/mypackage"


@mock.patch.object(model_cache_paths, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_in_the_middle() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="my-model/../../home", package_id="mypackage"
    )

    # then
    assert result == "/some/models-cache/my-model-home-0b1d84f7/mypackage"


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
        model_package_id="my-package",
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features={"some": "value"},
        trusted_source=True,
        offline_compatibility_hash="c" * 64,
    )

    # then
    auto_resolution_cache.register.assert_called_once_with(
        auto_negotiation_hash="my-hash",
        cache_entry=AutoResolutionCacheEntry(
            model_id="my-model",
            model_package_id="my-package",
            resolved_files={"some/file.txt"},
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=now,
            model_features={"some": "value"},
            trusted_source=True,
            offline_compatibility_hash="c" * 64,
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
    )

    registered_entry = auto_resolution_cache.register.call_args.kwargs["cache_entry"]
    assert registered_entry.recommended_parameters == recommended_parameters
    assert registered_entry.cache_model_id == "canonical-model"
    assert registered_entry.trusted_source is None
    assert registered_entry.offline_compatibility_hash is None


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
        model_package_id="localtrtabc123",
        model_architecture="rfdetr",
        task_type="object-detection",
        backend_type=BackendType.TRT,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features=None,
    )

    registered_entry = auto_resolution_cache.register.call_args.kwargs["cache_entry"]
    assert registered_entry.model_id == "rfdetr-nano"
    assert registered_entry.cache_model_id == "workspace/coco-38"


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
        model_package_id="my-package",
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features=None,
        recommended_parameters=recommended_parameters,
    )

    auto_resolution_cache.register.assert_called_once_with(
        auto_negotiation_hash="my-hash",
        cache_entry=AutoResolutionCacheEntry(
            model_id="my-model",
            model_package_id="my-package",
            resolved_files={"some/file.txt"},
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=now,
            model_features=None,
            recommended_parameters=recommended_parameters,
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
        model_package_id="my-package",
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        resolved_files={"some/file.txt"},
        model_dependencies=None,
        model_features=None,
    )

    auto_resolution_cache.register.assert_called_once_with(
        auto_negotiation_hash="my-hash",
        cache_entry=AutoResolutionCacheEntry(
            model_id="my-model",
            model_package_id="my-package",
            resolved_files={"some/file.txt"},
            model_architecture="yolov8",
            task_type="object-detection",
            backend_type=BackendType.ONNX,
            created_at=now,
            model_features=None,
            recommended_parameters=None,
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
    )

    # then
    on_file_created.assert_not_called()
    with open(config_path) as file:
        decoded = json.load(file)
    assert decoded["offline_manifest_version"] == core.OFFLINE_CACHE_MANIFEST_VERSION
    assert decoded["model_architecture"] == "yolov8"
    assert decoded["task_type"] == "object-detection"
    assert decoded["backend_type"] == "onnx"


def test_dump_model_config_for_offline_use_upgrades_legacy_config(
    empty_local_dir: str,
) -> None:
    """A legacy config is upgraded to a complete v2 manifest in place."""
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
    }
    on_file_created.assert_not_called()


def test_dump_model_config_for_offline_use_preserves_existing_model_id(
    empty_local_dir: str,
) -> None:
    """An existing canonical model ID is never replaced by an alias."""
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
    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=10,
        model_id="workspace/alias/3",
    )

    # then
    with open(config_path) as file:
        decoded = json.load(file)
    assert decoded["model_id"] == "workspace/canonical-project/3"
    assert decoded["offline_manifest_version"] == core.OFFLINE_CACHE_MANIFEST_VERSION
    assert decoded["model_architecture"] == "yolov8"
    assert decoded["task_type"] == "object-detection"
    assert decoded["backend_type"] == "onnx"


@pytest.mark.parametrize(
    "existing_model_id",
    [None, "", [], {}, 7, False],
)
def test_dump_model_config_repairs_malformed_existing_model_id(
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

    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        file_lock_acquire_timeout=10,
        model_id="workspace/project/3",
    )

    with open(config_path) as file:
        assert json.load(file)["model_id"] == "workspace/project/3"


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
    }


@pytest.mark.parametrize("corrupt_content", ["not-json", "[]"])
def test_dump_model_config_for_offline_use_repairs_corrupt_existing_config(
    empty_local_dir: str,
    corrupt_content: str,
) -> None:
    config_path = os.path.join(empty_local_dir, "model_config.json")
    _create_file(config_path, corrupt_content)
    on_file_created = MagicMock()

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
    )

    with open(config_path) as file:
        decoded = json.load(file)
    assert decoded["offline_manifest_version"] == core.OFFLINE_CACHE_MANIFEST_VERSION
    assert decoded["model_id"] == "workspace/project/3"
    assert decoded["trusted_source"] is True
    assert decoded["model_dependencies"] == []
    assert decoded["runtime_compatibility_hash"] == "a" * 64
    assert decoded["offline_compatibility_hash"] == "b" * 64
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
    )

    with open(config_path) as file:
        assert json.load(file)["model_id"] == "workspace/project/3"
    on_file_created.assert_called_once_with(config_path)


def test_new_offline_parameters_are_appended_to_existing_helper_signatures() -> None:
    assert list(
        inspect.signature(dump_model_config_for_offline_use).parameters
    )[:7] == [
        "config_path",
        "model_architecture",
        "task_type",
        "backend_type",
        "file_lock_acquire_timeout",
        "on_file_created",
        "model_id",
    ]
    assert list(inspect.signature(initialize_model).parameters)[-1] == (
        "offline_compatibility_hash"
    )
    assert list(
        inspect.signature(attempt_loading_matching_model_packages).parameters
    )[-1] == "offline_compatibility_hash"
    assert list(inspect.signature(dump_auto_resolution_cache).parameters)[-1] == (
        "offline_compatibility_hash"
    )


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

    with mock.patch.object(
        core.AutoModel, "from_pretrained", return_value=dependency_model
    ) as dependency_load, mock.patch.object(
        core, "resolve_model_class", return_value=CachedModel
    ), mock.patch.object(
        core, "generate_model_package_cache_path", return_value="/cached/model"
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
            "nms_fusion_preferences": None,
            "weights_provider_extra_query_params": None,
            "weights_provider_extra_headers": None,
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
}


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
                "backend_type": (
                    "torch-script" if package_id == "pkg002" else "onnx"
                ),
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
                "backend_type": (
                    "torch-script" if package_id == "pkg002" else "onnx"
                ),
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
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as mock_load:
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
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as mock_load:
        result = attempt_loading_model_from_offline_cache(
            model_id=model_id,
            model_init_kwargs={},
            model_access_manager=access_manager,
            api_key="test-key",
        )

    assert result is None
    access_manager.is_model_package_access_granted.assert_called_once_with(
        model_id=model_id,
        package_id="pkg001",
        api_key="test-key",
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


def test_attempt_loading_model_from_offline_cache_requires_opt_in_for_legacy_manifest(
    empty_local_dir: str,
) -> None:
    model_id = "yolov8n-640"
    package_dir = _write_offline_package(
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
    assert accepted == (mock_model, package_dir)
    mock_load.assert_called_once()


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
        },
    )
    dependency_model = MagicMock()
    parent_model = MagicMock()

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core.AutoModel, "from_pretrained", return_value=dependency_model
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
    assert (
        dependency_load.call_args.kwargs["model_package_id"] == "dependencyPackage"
    )
    assert model_load.call_args.kwargs["model_init_kwargs"][
        core.MODEL_DEPENDENCIES_KEY
    ] == {"encoder": dependency_model}


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
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage"
    ) as mock_load:
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
    with mock.patch.object(
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
            api_key="test-key",
            use_auto_resolution_cache=False,
        )

    # then
    assert result is mock_model
    assert mock_load.call_args[1]["model_dir_or_weights_path"] == package_dir


def test_custom_cache_compatible_lookup_survives_api_key_change() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    compatible_cache_entry = AutoResolutionCacheEntry(
        model_id=model_id,
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
            api_key="new-api-key",
            auto_resolution_cache=auto_resolution_cache,
        )

    assert result is cached_model
    assert cached_load.call_count == 2
    assert cached_load.call_args_list[1].kwargs["auto_negotiation_hash"] == (
        "old-api-key-hash"
    )
    assert auto_resolution_cache.find_compatible_calls == 1
    provider.assert_not_called()


def test_from_pretrained_tries_older_compatible_entry_after_newest_fails() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    newest_entry = AutoResolutionCacheEntry(
        model_id=model_id,
        model_package_id="newest-package",
        resolved_files=[],
        model_architecture="yolov8",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
        offline_compatibility_hash="c" * 64,
        trusted_source=True,
    )
    older_entry = newest_entry.model_copy(
        update={"model_package_id": "older-package"}
    )
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
            api_key="new-api-key",
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


def test_from_pretrained_compatibility_hash_excludes_api_key() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.find_compatible_candidates.return_value = []

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
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
            api_key="first-api-key",
            auto_resolution_cache=auto_resolution_cache,
        )
        second = core.AutoModel.from_pretrained(
            model_id,
            api_key="second-api-key",
            auto_resolution_cache=auto_resolution_cache,
        )

    assert first is cached_model
    assert second is cached_model
    exact_hashes = [
        cache_call.kwargs["auto_negotiation_hash"]
        for cache_call in exact_cache_load.call_args_list
    ]
    assert exact_hashes[0] != exact_hashes[1]
    compatible_hashes = [
        cache_call.kwargs["offline_compatibility_hash"]
        for cache_call in (
            auto_resolution_cache.find_compatible_candidates.call_args_list
        )
    ]
    assert compatible_hashes[0] == compatible_hashes[1]
    raw_hashes = [
        cache_call.kwargs["offline_compatibility_hash"]
        for cache_call in raw_cache_load.call_args_list
    ]
    assert raw_hashes == compatible_hashes


def test_from_pretrained_hash_normalizes_semantically_equivalent_choice_sets() -> None:
    model_id = "workspace/model/1"
    cached_model = MagicMock()
    auto_resolution_cache = MagicMock()
    auto_resolution_cache.find_compatible_candidates.return_value = []

    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
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
        model_cache_paths, "INFERENCE_HOME", empty_local_dir
    ), mock.patch.object(
        core, "get_model_from_provider", mock_provider
    ), mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=mock_model
    ):
        result = core.AutoModel.from_pretrained(
            model_id,
            api_key="test-key",
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
