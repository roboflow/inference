from __future__ import annotations

import pytest

from inference_models import BackendType, Quantization
from inference_models.developer_tools import ModelMetadata, ModelPackageMetadata
from inference_models.weights_providers.entities import FileDownloadSpecs
from profiling.memory.package_selection import (
    PackageSelectionMode,
    classify_package_selection_mode,
    resolve_profiling_package_by_backend_quantization,
    resolve_profiling_package_by_id,
    resolve_profiling_package_by_registry_identity,
)


def _package(
    *,
    package_id: str,
    backend: BackendType,
    quantization: Quantization,
) -> ModelPackageMetadata:
    return ModelPackageMetadata(
        package_id=package_id,
        backend=backend,
        package_artefacts=[
            FileDownloadSpecs(
                download_url="https://example.com/weights",
                file_handle="weights.onnx",
            )
        ],
        quantization=quantization,
    )


def _metadata(
    *,
    packages: list[ModelPackageMetadata],
    model_variant: str | None = "yolov8-n",
) -> ModelMetadata:
    return ModelMetadata(
        model_id="workspace/yolov8n",
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=packages,
        model_variant=model_variant,
    )


def test_classify_path_one() -> None:
    mode = classify_package_selection_mode(
        model_id="workspace/yolov8n",
        package_id="pkg-1",
        backend=None,
        architecture=None,
        task_type=None,
        model_variant=None,
        quantization=None,
    )

    assert mode == PackageSelectionMode.BY_PACKAGE_ID


def test_classify_path_one_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="path 1"):
        classify_package_selection_mode(
            model_id="workspace/yolov8n",
            package_id="pkg-1",
            backend="onnx",
            architecture=None,
            task_type=None,
            model_variant=None,
            quantization=None,
        )


def test_classify_path_two() -> None:
    mode = classify_package_selection_mode(
        model_id="workspace/yolov8n",
        package_id=None,
        backend="onnx",
        architecture=None,
        task_type=None,
        model_variant=None,
        quantization="fp32",
    )

    assert mode == PackageSelectionMode.BY_BACKEND_QUANTIZATION


def test_classify_path_three() -> None:
    mode = classify_package_selection_mode(
        model_id="workspace/yolov8n",
        package_id=None,
        backend="onnx",
        architecture="yolov8",
        task_type="object-detection",
        model_variant=None,
        quantization="fp32",
    )

    assert mode == PackageSelectionMode.BY_REGISTRY_IDENTITY


def test_resolve_by_package_id(monkeypatch: pytest.MonkeyPatch) -> None:
    package = _package(
        package_id="pkg-onnx",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
    )
    metadata = _metadata(packages=[package])

    monkeypatch.setattr(
        "profiling.memory.package_selection.fetch_model_metadata",
        lambda **kwargs: metadata,
    )

    selection = resolve_profiling_package_by_id(
        model_id="workspace/yolov8n",
        package_id="pkg-onnx",
    )

    assert selection.harness_backend == "onnx"
    assert selection.architecture == "yolov8"
    assert selection.task_type == "object-detection"
    assert selection.quantization == "fp32"


def test_resolve_by_backend_quantization_requires_single_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(
        packages=[
            _package(
                package_id="pkg-a",
                backend=BackendType.ONNX,
                quantization=Quantization.FP32,
            ),
            _package(
                package_id="pkg-b",
                backend=BackendType.ONNX,
                quantization=Quantization.FP32,
            ),
        ]
    )
    monkeypatch.setattr(
        "profiling.memory.package_selection.fetch_model_metadata",
        lambda **kwargs: metadata,
    )

    with pytest.raises(ValueError, match="Multiple packages match"):
        resolve_profiling_package_by_backend_quantization(
            model_id="workspace/yolov8n",
            harness_backend="onnx",
            quantization="fp32",
        )


def test_resolve_by_registry_identity_validates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _package(
        package_id="pkg-onnx",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
    )
    metadata = _metadata(packages=[package])
    monkeypatch.setattr(
        "profiling.memory.package_selection.fetch_model_metadata",
        lambda **kwargs: metadata,
    )

    with pytest.raises(ValueError, match="architecture"):
        resolve_profiling_package_by_registry_identity(
            model_id="workspace/yolov8n",
            harness_backend="onnx",
            quantization="fp32",
            architecture="resnet",
            task_type="object-detection",
        )
