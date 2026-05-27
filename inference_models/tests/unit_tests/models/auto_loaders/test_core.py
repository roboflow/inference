import json
import os.path
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from inference_models import ClassificationPrediction
from inference_models.errors import (
    CorruptedModelPackageError,
    InsecureModelIdentifierError,
    ModelLoadingError,
)
from inference_models.models.auto_loaders import core
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCacheEntry,
)
from inference_models.models.auto_loaders.core import (
    attempt_loading_model_from_local_storage,
    create_symlinks_to_shared_blobs,
    dump_auto_resolution_cache,
    dump_model_config_for_offline_use,
    generate_model_package_cache_path,
    load_class_from_path,
    materialize_model_package,
    materialize_selected_model_package,
    parse_model_config,
    resolve_recommended_parameters,
)
from inference_models.models.auto_loaders.entities import (
    BackendType,
    InferenceModelConfig,
)
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
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


@mock.patch.object(core, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path() -> None:
    # when
    result = generate_model_package_cache_path(
        model_id="my-model", package_id="mypackage"
    )

    # then
    assert result == "/some/models-cache/my-model-6fa11b0c/mypackage"


@mock.patch.object(core, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_at_front() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="/my-model", package_id="mypackage"
    )

    # then
    assert result == "/some/models-cache/-my-model-9651d483/mypackage"


@mock.patch.object(core, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_in_the_middle() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="my-model/../../home", package_id="mypackage"
    )

    # then
    assert result == "/some/models-cache/my-model-home-0b1d84f7/mypackage"


@mock.patch.object(core, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_package_id_is_not_sanitized() -> None:
    # when
    with pytest.raises(InsecureModelIdentifierError):
        _ = generate_model_package_cache_path(
            model_id="my-model", package_id="/my-package"
        )


def test_materialize_selected_model_package_downloads_files_without_instantiating(
    empty_local_dir: str,
) -> None:
    # given
    package_params = RecommendedParameters(confidence=0.8)
    model_params = RecommendedParameters(confidence=0.4)
    model_package = ModelPackageMetadata(
        package_id="pkgHF",
        backend=BackendType.HF,
        quantization=Quantization.UNKNOWN,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/base-config",
                file_handle="base/config.json",
                md5_hash="hashbaseconfig",
            ),
            FileDownloadSpecs(
                download_url="https://example.com/adapter-config",
                file_handle="adapter_config.json",
                md5_hash="hashadapterconfig",
            ),
            FileDownloadSpecs(
                download_url="https://example.com/tokenizer",
                file_handle="tokenizer_config.json",
            ),
        ],
        trusted_source=True,
        recommended_parameters=package_params,
    )
    on_file_created = MagicMock()
    on_file_renamed = MagicMock()
    on_symlink_created = MagicMock()
    on_symlink_deleted = MagicMock()

    # when
    with mock.patch.object(core, "INFERENCE_HOME", empty_local_dir), mock.patch.object(
        core,
        "download_files_to_directory",
        side_effect=_fake_download_files_to_directory,
    ), mock.patch.object(core, "resolve_model_class") as resolve_model_class_mock:
        result = materialize_selected_model_package(
            model_id="vlm-ocr/11",
            model_architecture="qwen3_5",
            model_variant="2b-peft",
            task_type="vlm",
            model_package=model_package,
            model_dependencies=None,
            model_dependencies_directories={},
            recommended_parameters=model_params,
            model_download_file_lock_acquire_timeout=10,
            on_file_created=on_file_created,
            on_file_renamed=on_file_renamed,
            on_symlink_created=on_symlink_created,
            on_symlink_deleted=on_symlink_deleted,
        )

    # then
    expected_package_dir = result.package_dir
    expected_config_path = os.path.join(expected_package_dir, "model_config.json")
    expected_base_link = os.path.join(expected_package_dir, "base", "config.json")
    expected_adapter_link = os.path.join(expected_package_dir, "adapter_config.json")
    expected_tokenizer_path = os.path.join(
        expected_package_dir, "tokenizer_config.json"
    )
    expected_shared_base_path = os.path.join(
        empty_local_dir, "shared-blobs", "hashbaseconfig"
    )
    expected_shared_adapter_path = os.path.join(
        empty_local_dir, "shared-blobs", "hashadapterconfig"
    )

    assert result.model_id == "vlm-ocr/11"
    assert result.model_architecture == "qwen3_5"
    assert result.model_variant == "2b-peft"
    assert result.task_type == "vlm"
    assert result.model_package_id == "pkgHF"
    assert result.backend == BackendType.HF
    assert result.package_dir == expected_package_dir
    assert result.model_package is model_package
    assert result.recommended_parameters is package_params
    assert expected_base_link in result.resolved_files
    assert expected_adapter_link in result.resolved_files
    assert expected_tokenizer_path in result.resolved_files
    assert expected_shared_base_path in result.resolved_files
    assert expected_shared_adapter_path in result.resolved_files
    assert expected_config_path in result.resolved_files
    assert os.path.islink(expected_base_link)
    assert os.path.islink(expected_adapter_link)
    assert not os.path.islink(expected_tokenizer_path)
    assert _read_file(expected_base_link) == "base/config.json"
    assert _read_file(expected_adapter_link) == "adapter_config.json"
    assert _read_file(expected_tokenizer_path) == "tokenizer_config.json"
    resolve_model_class_mock.assert_not_called()


def test_materialize_model_package_resolves_metadata_and_selects_package(
    empty_local_dir: str,
) -> None:
    # given
    qwen_package = ModelPackageMetadata(
        package_id="qwenHF",
        backend=BackendType.HF,
        quantization=Quantization.UNKNOWN,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/adapter",
                file_handle="adapter_config.json",
                md5_hash="hashadapterconfig",
            )
        ],
        trusted_source=True,
    )
    metadata = ModelMetadata(
        model_id="vlm-ocr/11",
        model_architecture="qwen3_5",
        model_variant="2b-peft",
        task_type="vlm",
        model_packages=[qwen_package],
    )

    # when
    with mock.patch.object(core, "INFERENCE_HOME", empty_local_dir), mock.patch.object(
        core,
        "get_model_from_provider",
        return_value=metadata,
    ) as get_model_from_provider_mock, mock.patch.object(
        core,
        "negotiate_model_packages",
        return_value=[qwen_package],
    ) as negotiate_model_packages_mock, mock.patch.object(
        core,
        "download_files_to_directory",
        side_effect=_fake_download_files_to_directory,
    ):
        result = materialize_model_package(
            model_id="vlm-ocr/11",
            weights_provider="test-provider",
            api_key="test-key",
            backend=BackendType.HF,
            quantization=Quantization.UNKNOWN,
            device="cpu",
            verify_hash_while_download=True,
        )

    # then
    assert result.model_id == "vlm-ocr/11"
    assert result.requested_model_id == "vlm-ocr/11"
    assert result.model_architecture == "qwen3_5"
    assert result.model_variant == "2b-peft"
    assert result.backend == BackendType.HF
    get_model_from_provider_mock.assert_called_once_with(
        provider="test-provider",
        model_id="vlm-ocr/11",
        api_key="test-key",
        weights_provider_extra_query_params=None,
        weights_provider_extra_headers=None,
    )
    negotiate_model_packages_mock.assert_called_once()
    _, negotiate_kwargs = negotiate_model_packages_mock.call_args
    assert negotiate_kwargs["model_architecture"] == "qwen3_5"
    assert negotiate_kwargs["task_type"] == "vlm"
    assert negotiate_kwargs["model_packages"] == [qwen_package]
    assert negotiate_kwargs["requested_backends"] == BackendType.HF
    assert negotiate_kwargs["requested_quantization"] == Quantization.UNKNOWN


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
        ),
    )


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


def test_dump_model_config_for_offline_use_when_file_does_not_exists(
    empty_local_dir: str,
) -> None:
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
        "model_architecture": "yolov8",
        "task_type": "object-detection",
        "backend_type": "onnx",
    }


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


def _fake_download_files_to_directory(
    target_dir: str,
    files_specs: list,
    name_after: str = None,
    **kwargs,
) -> dict:
    result = {}
    for file_handle, _download_url, md5_hash in files_specs:
        filename = md5_hash if name_after == "md5_hash" else file_handle
        file_path = os.path.join(target_dir, filename)
        _create_file(path=file_path, content=file_handle)
        result[file_handle] = file_path
    return result


def _create_file(path: str, content: str) -> None:
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
