import json
import os.path
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from inference_exp import ClassificationPrediction
from inference_exp.errors import (
    CorruptedModelPackageError,
    InsecureModelIdentifierError,
    ModelLoadingError,
)
from inference_exp.models.auto_loaders import core
from inference_exp.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCacheEntry,
)
from inference_exp.models.auto_loaders.core import (
    attempt_loading_model_from_local_storage,
    create_symlinks_to_shared_blobs,
    dump_auto_resolution_cache,
    dump_model_config_for_offline_use,
    generate_model_package_cache_path,
    load_class_from_path,
    parse_model_config,
)
from inference_exp.models.auto_loaders.entities import InferenceModelConfig
from inference_exp.weights_providers.entities import BackendType


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
        model_id="my-model", package_id="my-package"
    )

    # then
    assert result == "/some/models-cache/my-model-6fa11b0c/my-package"


@mock.patch.object(core, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_at_front() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="/my-model", package_id="my-package"
    )

    # then
    assert result == "/some/models-cache/-my-model-9651d483/my-package"


@mock.patch.object(core, "INFERENCE_HOME", "/some")
def test_generate_model_package_cache_path_when_id_contains_forward_slash_in_the_middle() -> (
    None
):
    # when
    result = generate_model_package_cache_path(
        model_id="my-model/../../home", package_id="my-package"
    )

    # then
    assert result == "/some/models-cache/my-model-home-0b1d84f7/my-package"


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
        ),
    )


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
