import importlib
from types import SimpleNamespace

import pytest

from inference.core.exceptions import ModelPackageSelectionError


@pytest.fixture
def adapters(monkeypatch):
    module = importlib.import_module("inference.core.models.inference_models_adapters")
    monkeypatch.setattr(module, "VALID_INFERENCE_MODELS_BACKENDS", {"onnx", "trt"})
    monkeypatch.setattr(module, "DISABLED_INFERENCE_MODELS_BACKENDS", set())
    return module


def test_pinned_package_requires_library_policy_validation(adapters, monkeypatch):
    calls = []

    def from_pretrained(model_id_or_path, validate_model_package=False, **kwargs):
        calls.append((model_id_or_path, validate_model_package, kwargs))
        return SimpleNamespace(pre_process=lambda x: x, class_names=[])

    monkeypatch.setattr(
        adapters, "AutoModel", SimpleNamespace(from_pretrained=from_pretrained)
    )

    adapters.InferenceModelsObjectDetectionAdapter(
        model_id="workspace/model/1", model_package_id="package-id"
    )

    assert len(calls) == 1
    model_id, validate_model_package, kwargs = calls[0]
    assert model_id == "workspace/model/1"
    assert validate_model_package is True
    assert kwargs["backend"] == ["onnx", "trt"]
    assert kwargs["model_package_id"] == "package-id"


def test_pinned_package_rejects_library_without_policy_validation(
    adapters, monkeypatch
):
    calls = []

    def from_pretrained(model_id_or_path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(pre_process=lambda x: x, class_names=[])

    monkeypatch.setattr(
        adapters, "AutoModel", SimpleNamespace(from_pretrained=from_pretrained)
    )

    with pytest.raises(ModelPackageSelectionError, match="cannot validate pinned"):
        adapters.InferenceModelsObjectDetectionAdapter(
            model_id="workspace/model/1", model_package_id="package-id"
        )

    assert calls == []


@pytest.mark.parametrize("kwargs", [{}, {"backend": "onnx"}])
def test_unpinned_selection_preserves_old_library_compatibility(
    adapters, monkeypatch, kwargs
):
    calls = []

    def from_pretrained(model_id_or_path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(pre_process=lambda x: x, class_names=[])

    monkeypatch.setattr(
        adapters, "AutoModel", SimpleNamespace(from_pretrained=from_pretrained)
    )

    adapters.InferenceModelsObjectDetectionAdapter(
        model_id="workspace/model/1", **kwargs
    )

    assert len(calls) == 1
    assert "validate_model_package" not in calls[0]
