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
    parse_model_config,
    resolve_recommended_parameters,
)
from inference_models.models.auto_loaders.entities import (
    BackendType,
    InferenceModelConfig,
)
from inference_models.weights_providers.entities import RecommendedParameters


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


def _create_file(path: str, content: str) -> None:
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# find_cached_model_package_dir
# ---------------------------------------------------------------------------


class TestFindCachedModelPackageDir:

    def test_returns_package_dir_when_model_config_exists(self, tmp_path):
        from inference_models.models.auto_loaders.core import (
            find_cached_model_package_dir,
            slugify_model_id_to_os_safe_format,
        )

        model_id = "coco/22"
        slug = slugify_model_id_to_os_safe_format(model_id=model_id)
        pkg_dir = tmp_path / "models-cache" / slug / "pkg001"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "model_config.json").write_text(
            json.dumps({"task_type": "object-detection"})
        )

        with mock.patch(
            "inference_models.models.auto_loaders.core.INFERENCE_HOME", str(tmp_path)
        ):
            result = find_cached_model_package_dir(model_id)

        assert result == str(pkg_dir)

    def test_returns_none_when_no_cache(self, tmp_path):
        from inference_models.models.auto_loaders.core import (
            find_cached_model_package_dir,
        )

        with mock.patch(
            "inference_models.models.auto_loaders.core.INFERENCE_HOME", str(tmp_path)
        ):
            result = find_cached_model_package_dir("nonexistent/model")

        assert result is None

    def test_returns_none_when_no_model_config(self, tmp_path):
        from inference_models.models.auto_loaders.core import (
            find_cached_model_package_dir,
            slugify_model_id_to_os_safe_format,
        )

        model_id = "my/model"
        slug = slugify_model_id_to_os_safe_format(model_id=model_id)
        pkg_dir = tmp_path / "models-cache" / slug / "pkg001"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "weights.onnx").write_text("fake")

        with mock.patch(
            "inference_models.models.auto_loaders.core.INFERENCE_HOME", str(tmp_path)
        ):
            result = find_cached_model_package_dir(model_id)

        assert result is None


# ---------------------------------------------------------------------------
# RetryError offline fallback in from_pretrained
# ---------------------------------------------------------------------------


class TestFromPretrainedOfflineFallback:

    def test_falls_back_to_cached_package_on_retry_error(self, tmp_path):
        from inference_models.errors import RetryError
        from inference_models.models.auto_loaders.core import (
            find_cached_model_package_dir,
            slugify_model_id_to_os_safe_format,
        )

        model_id = "test/model"
        slug = slugify_model_id_to_os_safe_format(model_id=model_id)
        pkg_dir = tmp_path / "models-cache" / slug / "pkg001"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "model_config.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "task_type": "object-detection",
                    "model_architecture": "yolov8n",
                    "backend_type": "onnxruntime",
                }
            )
        )

        fake_model = MagicMock()

        with mock.patch(
            "inference_models.models.auto_loaders.core.INFERENCE_HOME", str(tmp_path)
        ), mock.patch(
            "inference_models.models.auto_loaders.core.get_model_from_provider",
            side_effect=RetryError("network down"),
        ), mock.patch(
            "inference_models.models.auto_loaders.core.attempt_loading_model_from_local_storage",
            return_value=fake_model,
        ) as mock_load:
            from inference_models.models.auto_loaders.core import AutoModel

            result = AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key="test-key",
            )

        assert result is fake_model
        mock_load.assert_called_once()
        assert mock_load.call_args[1]["model_dir_or_weights_path"] == str(pkg_dir)

    def test_reraises_retry_error_when_no_cache(self, tmp_path):
        from inference_models.errors import RetryError

        with mock.patch(
            "inference_models.models.auto_loaders.core.INFERENCE_HOME", str(tmp_path)
        ), mock.patch(
            "inference_models.models.auto_loaders.core.get_model_from_provider",
            side_effect=RetryError("network down"),
        ):
            from inference_models.models.auto_loaders.core import AutoModel

            with pytest.raises(RetryError):
                AutoModel.from_pretrained(
                    model_id_or_path="nonexistent/model",
                    api_key="test-key",
                )
