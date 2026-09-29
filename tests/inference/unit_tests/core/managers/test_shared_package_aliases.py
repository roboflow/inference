from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inference.core.managers import base as base_module
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.models import base as model_module
from inference.core.models.base import Model
from inference_models import AutoModel


@pytest.mark.parametrize("first_selectors", [{}, {"backend": "onnx"}])
def test_aliases_share_one_package_and_retain_request_attribution(
    monkeypatch, first_selectors
):
    monkeypatch.setattr(base_module, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(model_module, "USE_INFERENCE_MODELS", True)
    metadata = SimpleNamespace(
        model_id="coco/41",
        model_package_id="onnx-package",
        backend="onnx",
        quantization="fp32",
    )
    resolution = Mock(return_value=[metadata])
    monkeypatch.setattr(AutoModel, "resolve_model_packages", resolution, raising=False)
    loaded = []

    class PackageModel(Model):
        supports_model_package_selection = True
        task_type = "object-detection"

        def __init__(self, **kwargs):
            self._model = SimpleNamespace(resolved_model=metadata)
            loaded.append(self)

    registry = Mock()
    registry.get_model.return_value = PackageModel
    manager = WithFixedSizeCache(
        ModelManager(registry, content_addressed_artifact_cache=Mock()), max_size=3
    )
    first_key = manager.load_model(
        "coco/41", "first-key", model_id_alias="yolo26n-640", **first_selectors
    )
    second_key = manager.load_model(
        "coco/41", "second-key", model_id_alias="yolov26n-640", backend="onnx"
    )
    third_key = manager.load_model("coco/41", "third-key")
    assert manager.add_model("yolo26n-640", "fourth-key") is None

    assert first_key == second_key == third_key
    for alias in ("yolo26n-640", "yolov26n-640", "coco/41"):
        assert alias in manager
        assert manager[alias] is loaded[0]
        assert manager.get_task_type(alias) == "object-detection"
    assert len(loaded) == len(manager.keys()) == 1
    assert all(
        call.kwargs["model_id"] == "coco/41" for call in resolution.call_args_list
    )
    assert {"yolo26n-640", "yolov26n-640"}.issubset(
        manager.describe_models()[0].request_aliases
    )

    manager.pin_model("yolo26n-640")
    manager.max_size = 1
    manager.add_model("another-project/1", "key")

    assert manager[first_key] is loaded[0]
    assert manager["coco/41"] is loaded[0]
    assert len(manager.keys()) == 2


def test_alias_lookup_without_package_resolution(monkeypatch):
    monkeypatch.setattr(base_module, "USE_INFERENCE_MODELS", True)
    monkeypatch.delattr(AutoModel, "resolve_model_packages", raising=False)
    loaded = []

    class LegacyPackageModel(Model):
        supports_model_package_selection = True
        task_type = "object-detection"

        def __init__(self, **kwargs):
            loaded.append(self)

    registry = Mock()
    registry.get_model.return_value = LegacyPackageModel
    manager = WithFixedSizeCache(
        ModelManager(registry, content_addressed_artifact_cache=Mock()), max_size=3
    )

    manager.add_model("yolo26n-640", "key")
    manager.add_model("coco/41", "key")

    assert len(loaded) == len(manager.keys()) == 1
    assert manager["yolo26n-640"] is manager["coco/41"]
    assert manager.get_task_type("yolo26n-640") == "object-detection"
