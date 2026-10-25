from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from packaging.version import Version

from inference_models.entities import ResolvedModelMetadata
from inference_models.models.auto_loaders import (
    auto_negotiation,
    auto_resolution_cache,
    core,
    model_cache_paths,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    ModelMetadata,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
)


@pytest.fixture(autouse=True)
def available_onnx_backend(monkeypatch):
    runtime = replace(
        core.x_ray_runtime_environment(),
        onnxruntime_version=Version("1.20.1"),
        available_onnx_execution_providers={"CPUExecutionProvider"},
    )
    monkeypatch.setattr(core, "x_ray_runtime_environment", lambda: runtime)
    monkeypatch.setattr(auto_negotiation, "x_ray_runtime_environment", lambda: runtime)
    monkeypatch.setattr(
        auto_negotiation,
        "get_selected_onnx_execution_providers",
        lambda: ["CPUExecutionProvider"],
    )


@pytest.mark.parametrize(
    "missing_cache_field", [None, "backend_type", "canonical_model_id"]
)
def test_auto_model_reports_canonical_package_after_fresh_and_cached_loads(
    tmp_path, monkeypatch, missing_cache_field
):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    monkeypatch.setattr(auto_resolution_cache, "INFERENCE_HOME", str(tmp_path))
    package = ModelPackageMetadata(
        package_id="onnxpackage",
        backend=BackendType.ONNX,
        quantization=Quantization.FP32,
        package_artefacts=[],
        onnx_package_details=ONNXPackageDetails(opset=17),
        trusted_source=True,
    )
    metadata = ModelMetadata(
        model_id="canonical/1",
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=[package],
    )
    monkeypatch.setattr(core, "get_model_from_provider", lambda **kwargs: metadata)
    model_class = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(core, "resolve_model_class", lambda **kwargs: model_class)

    first = core.AutoModel.from_pretrained("alias/1", backend="onnx", device="cpu")

    first_metadata = getattr(first, "resolved_model", None)
    assert isinstance(first_metadata, ResolvedModelMetadata)
    assert asdict(first_metadata) == {
        "model_id": "canonical/1",
        "model_package_id": "onnxpackage",
        "backend": "onnx",
        "quantization": "fp32",
    }

    refreshed_metadata = []

    def refresh_provider_metadata(**kwargs):
        refreshed_metadata.append(kwargs)
        return metadata

    monkeypatch.setattr(core, "get_model_from_provider", refresh_provider_metadata)

    class PartialMetadataCache(auto_resolution_cache.BaseAutoLoadMetadataCache):
        def retrieve(self, auto_negotiation_hash):
            entry = super().retrieve(auto_negotiation_hash)
            if entry is not None and missing_cache_field is not None:
                return entry.model_copy(update={missing_cache_field: None})
            return entry

    second = core.AutoModel.from_pretrained(
        "alias/1",
        backend="onnx",
        device="cpu",
        auto_resolution_cache=PartialMetadataCache(file_lock_acquire_timeout=1),
    )
    assert second is not first
    second_metadata = getattr(second, "resolved_model", None)
    assert second_metadata == first_metadata
    assert len(refreshed_metadata) == (0 if missing_cache_field is None else 1)

    third = core.AutoModel.from_pretrained("alias/1", backend="onnx", device="cpu")
    assert getattr(third, "resolved_model", None) == first_metadata
    assert len(refreshed_metadata) == (0 if missing_cache_field is None else 1)


def test_auto_model_reports_the_successful_fallback_package(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))
    monkeypatch.setattr(auto_resolution_cache, "INFERENCE_HOME", str(tmp_path))
    packages = [
        ModelPackageMetadata(
            package_id=package_id,
            backend=BackendType.ONNX,
            quantization=quantization,
            package_artefacts=[],
            onnx_package_details=ONNXPackageDetails(opset=17),
            trusted_source=True,
        )
        for package_id, quantization in [
            ("brokenfp16", Quantization.FP16),
            ("workingfp32", Quantization.FP32),
        ]
    ]
    metadata = ModelMetadata(
        model_id="canonical/1",
        model_architecture="yolov8",
        task_type="object-detection",
        model_packages=packages,
    )
    monkeypatch.setattr(core, "get_model_from_provider", lambda **kwargs: metadata)
    attempted_packages = []

    def load_package(package_path, **kwargs):
        package_id = package_path.rsplit("/", 1)[-1]
        attempted_packages.append(package_id)
        if package_id == "brokenfp16":
            raise RuntimeError("Unsupported package")
        return SimpleNamespace()

    model_class = SimpleNamespace(from_pretrained=load_package)
    monkeypatch.setattr(core, "resolve_model_class", lambda **kwargs: model_class)

    model = core.AutoModel.from_pretrained(
        "alias/1", backend="onnx", quantization=["fp16", "fp32"], device="cpu"
    )

    assert attempted_packages == ["brokenfp16", "workingfp32"]
    resolved_model = getattr(model, "resolved_model", None)
    assert isinstance(resolved_model, ResolvedModelMetadata)
    assert asdict(resolved_model) == {
        "model_id": "canonical/1",
        "model_package_id": "workingfp32",
        "backend": "onnx",
        "quantization": "fp32",
    }
