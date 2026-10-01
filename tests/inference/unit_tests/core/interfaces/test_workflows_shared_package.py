from types import SimpleNamespace
from unittest.mock import Mock

from inference.core.entities.requests.inference import InferenceRequest
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.managers import base as base_module
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.models import base as model_module
from inference.core.models.base import Model
from inference_models import AutoModel


def test_workflow_reuses_package_loaded_with_http_selection(monkeypatch):
    monkeypatch.setattr(base_module, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(base_module, "DISABLE_INFERENCE_CACHE", True)
    monkeypatch.setattr(model_module, "USE_INFERENCE_MODELS", True)
    metadata = SimpleNamespace(
        model_id="project/1",
        model_package_id="onnx-package",
        backend="onnx",
        quantization="fp32",
    )
    resolve = Mock(return_value=[metadata])
    monkeypatch.setattr(AutoModel, "resolve_model_packages", resolve)
    loaded = []

    class PackageModel(Model):
        supports_model_package_selection = True
        task_type = "object-detection"

        def __init__(self, **kwargs):
            self._model = SimpleNamespace(resolved_model=metadata)
            loaded.append(self)

        def infer_from_request(self, request):
            return {"instance": id(self), "api_key": request.api_key}

    registry = Mock()
    registry.get_model.return_value = PackageModel
    manager = WithFixedSizeCache(
        ModelManager(registry, content_addressed_artifact_cache=Mock()), max_size=3
    )
    selected_key = manager.load_model("project/1", "http-key", backend="onnx")
    provider = ModelManagerModelsProvider(manager)

    assert provider.add_model("project/1", "workflow-key") is None
    result = provider.infer_from_request_sync(
        model_id="project/1",
        request=InferenceRequest(
            id="request-id", model_id="project/1", api_key="workflow-key"
        ),
    )

    assert len(loaded) == len(manager.keys()) == 1
    assert result == {"instance": id(manager[selected_key]), "api_key": "workflow-key"}
    assert [call.kwargs["api_key"] for call in resolve.call_args_list] == [
        "http-key",
        "workflow-key",
    ]
