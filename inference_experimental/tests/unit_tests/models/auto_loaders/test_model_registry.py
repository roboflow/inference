import pytest
from inference_exp.errors import ModelImplementationLoaderError
from inference_exp.models.auto_loaders.models_registry import (
    model_implementation_exists,
    resolve_model_class,
)
from inference_exp.weights_providers.entities import BackendType


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
