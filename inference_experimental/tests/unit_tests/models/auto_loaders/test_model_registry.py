from unittest import mock
from unittest.mock import MagicMock

import pytest
from inference_exp.errors import ModelImplementationLoaderError
from inference_exp.models.auto_loaders import models_registry
from inference_exp.models.auto_loaders.entities import BackendType
from inference_exp.models.auto_loaders.models_registry import (
    RegistryEntry,
    model_implementation_exists,
    resolve_model_class,
)


def test_model_implementation_exists_when_model_exists() -> None:
    # when
    result = model_implementation_exists(
        model_architecture="yolov8",
        task_type="object-detection",
        backend=BackendType.ONNX,
    )

    # then
    assert result is True


def test_model_implementation_exists_when_model_does_not_exists() -> None:
    # when
    result = model_implementation_exists(
        model_architecture="yolov8",
        task_type="invalid",
        backend=BackendType.ONNX,
    )

    # then
    assert result is False


def test_resolve_model_class_when_implementation_does_not_exist() -> None:
    # when
    with pytest.raises(ModelImplementationLoaderError):
        _ = resolve_model_class(
            model_architecture="yolov8",
            task_type="invalid",
            backend=BackendType.ONNX,
        )


@mock.patch.object(
    models_registry,
    "REGISTERED_MODELS",
    {
        ("rfdetr", "object-detection", BackendType.TORCH): MagicMock(
            resolve=lambda: 2137
        ),
    },
)
def test_resolve_model_class_when_class_found_as_simple_entry() -> None:
    # when
    result = resolve_model_class(
        model_architecture="rfdetr",
        task_type="object-detection",
        backend=BackendType.TORCH,
    )

    # then
    assert result == 2137


@mock.patch.object(
    models_registry,
    "REGISTERED_MODELS",
    {
        ("rfdetr", "object-detection", BackendType.TORCH): RegistryEntry(
            model_class=MagicMock(resolve=lambda: 2137),
            supported_model_features={"nms_fused"},
        ),
    },
)
def test_resolve_model_class_when_class_found_as_registry_entry() -> None:
    # when
    result = resolve_model_class(
        model_architecture="rfdetr",
        task_type="object-detection",
        backend=BackendType.TORCH,
    )

    # then
    assert result == 2137


@mock.patch.object(
    models_registry,
    "REGISTERED_MODELS",
    {
        ("rfdetr", "object-detection", BackendType.TORCH): RegistryEntry(
            model_class=MagicMock(resolve=lambda: 2137),
            supported_model_features={"nms_fused"},
        ),
    },
)
def test_resolve_model_class_when_class_found_as_registry_entry_when_features_specify() -> (
    None
):
    # when
    result = resolve_model_class(
        model_architecture="rfdetr",
        task_type="object-detection",
        backend=BackendType.TORCH,
        model_features={"nms_fused"},
    )

    # then
    assert result == 2137


@mock.patch.object(
    models_registry,
    "REGISTERED_MODELS",
    {
        ("rfdetr", "object-detection", BackendType.TORCH): RegistryEntry(
            model_class=MagicMock(resolve=lambda: 2137),
            supported_model_features={"nms_fused"},
        ),
    },
)
def test_resolve_model_class_when_class_not_found_as_registry_entry_when_features_exceed_specification() -> (
    None
):
    # when
    with pytest.raises(ModelImplementationLoaderError):
        _ = resolve_model_class(
            model_architecture="rfdetr",
            task_type="object-detection",
            backend=BackendType.TORCH,
            model_features={"new_feature"},
        )


@mock.patch.object(
    models_registry,
    "REGISTERED_MODELS",
    {
        ("rfdetr", "object-detection", BackendType.TORCH): MagicMock(
            resolve=lambda: 2137
        ),
    },
)
def test_model_implementation_exists_when_model_features_requested_against_simple_registry_entry() -> (
    None
):
    # when
    result = model_implementation_exists(
        model_architecture="rfdetr",
        task_type="object-detection",
        backend=BackendType.TORCH,
        model_features={"nms_fused"},
    )

    # then
    assert result is False


@mock.patch.object(
    models_registry,
    "REGISTERED_MODELS",
    {
        ("rfdetr", "object-detection", BackendType.TORCH): RegistryEntry(
            model_class=MagicMock(resolve=lambda: 2137),
            supported_model_features={"nms_fused"},
        ),
    },
)
def test_model_implementation_exists_when_unknown_model_features_requested() -> None:
    # when
    result = model_implementation_exists(
        model_architecture="rfdetr",
        task_type="object-detection",
        backend=BackendType.TORCH,
        model_features={"new_feature"},
    )

    # then
    assert result is False


@mock.patch.object(
    models_registry,
    "REGISTERED_MODELS",
    {
        ("rfdetr", "object-detection", BackendType.TORCH): RegistryEntry(
            model_class=MagicMock(resolve=lambda: 2137),
            supported_model_features={"nms_fused", "super_speed", "super_accuracy"},
        ),
    },
)
def test_model_implementation_exists_when_known_model_feature_requested() -> None:
    # when
    result = model_implementation_exists(
        model_architecture="rfdetr",
        task_type="object-detection",
        backend=BackendType.TORCH,
        model_features={"super_speed", "super_accuracy"},
    )

    # then
    assert result is True
