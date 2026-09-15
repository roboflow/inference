from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from inference.core import env
from inference.core.exceptions import ModelPackageSelectionError
from inference.core.managers import base as base_module
from inference.core.managers.active_learning import ActiveLearningManager
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference_models.entities import ResolvedModelMetadata


class PackageModel:
    task_type = "object-detection"
    batch_size = 1
    img_size_h = 32
    img_size_w = 32

    def __init__(self, model_id, api_key, backend=None, **kwargs):
        self.model_id = model_id
        self.resolved_model = ResolvedModelMetadata(
            model_id=model_id,
            model_package_id=f"{backend or 'onnx'}-package",
            backend=backend or "onnx",
            quantization="fp16" if backend == "trt" else "fp32",
        )

    def infer_from_request(self, request):
        return {"backend": self.resolved_model.backend}

    def clear_cache(self, delete_from_disk=True):
        pass


def test_package_variants_coexist_without_changing_automatic_model():
    registry = Mock()
    registry.get_model.return_value = PackageModel
    with patch.object(base_module, "USE_INFERENCE_MODELS", True):
        manager = WithFixedSizeCache(
            ModelManager(
                model_registry=registry, content_addressed_artifact_cache=Mock()
            ),
            max_size=3,
        )
        manager.add_model("project/1", "key")
        manager.add_model(
            "project/1", "key", backend="trt", model_cache_key="project/1:trt"
        )

    assert set(manager.keys()) == {"project/1", "project/1:trt"}
    assert manager.infer_from_request_sync("project/1", SimpleNamespace()) == {
        "backend": "onnx"
    }
    assert manager.infer_from_request_sync("project/1:trt", SimpleNamespace()) == {
        "backend": "trt"
    }
    assert all(
        call.args[0] == "project/1" for call in registry.get_model.call_args_list
    )


@pytest.fixture
def package_manager(monkeypatch):
    monkeypatch.setattr(base_module, "USE_INFERENCE_MODELS", True)
    registry = Mock()
    registry.get_model.return_value = PackageModel
    return WithFixedSizeCache(
        ModelManager(model_registry=registry, content_addressed_artifact_cache=Mock()),
        max_size=3,
    )


def test_explicit_first_load_does_not_bind_automatic_selection(package_manager):
    package_manager.add_model("project/1", "key", backend="trt", model_cache_key="trt")
    package_manager.add_model("project/1", "key")
    assert package_manager.infer_from_request_sync("project/1", SimpleNamespace()) == {
        "backend": "onnx"
    }


def test_cached_variant_is_validated_without_reloading(package_manager):
    package_manager.add_model("project/1", "key", backend="trt", model_cache_key="trt")
    package_manager.add_model("project/1", "key", backend="trt", model_cache_key="trt")
    with pytest.raises(ModelPackageSelectionError, match="requested backend"):
        package_manager.add_model(
            "project/1", "key", backend="onnx", model_cache_key="trt"
        )
    package_manager.model_manager.model_registry.get_model.assert_called_once()


def test_exact_package_id_cannot_bypass_disabled_backend(package_manager, monkeypatch):
    monkeypatch.setattr(env, "DISABLED_INFERENCE_MODELS_BACKENDS", {"onnx"})
    with pytest.raises(ModelPackageSelectionError, match="disabled"):
        package_manager.add_model(
            "project/1", "key", model_package_id="onnx-package", model_cache_key="exact"
        )
    assert not package_manager.keys()


def test_feature_flag_cannot_be_bypassed_by_warm_variant(package_manager, monkeypatch):
    package_manager.add_model("project/1", "key", backend="trt", model_cache_key="trt")
    monkeypatch.setattr(base_module, "USE_INFERENCE_MODELS", False)
    with pytest.raises(ModelPackageSelectionError, match="USE_INFERENCE_MODELS"):
        package_manager.add_model(
            "project/1", "key", backend="trt", model_cache_key="trt"
        )


def test_removing_variant_leaves_automatic_model(package_manager):
    package_manager.add_model("project/1", "key")
    package_manager.add_model("project/1", "key", backend="trt", model_cache_key="trt")
    package_manager.remove("trt")
    assert list(package_manager.keys()) == ["project/1"]


@pytest.mark.parametrize("as_alias", [False, True])
def test_cache_handle_cannot_bypass_selection_validation(package_manager, as_alias):
    handle = "project/1:package:secret"
    package_manager.add_model("project/1", "key", backend="trt", model_cache_key=handle)
    with pytest.raises(ModelPackageSelectionError, match="Cache handles"):
        package_manager.add_model(
            "project/1" if as_alias else handle,
            "different-key",
            model_id_alias=handle if as_alias else None,
        )


def test_active_learning_uses_public_identity_and_selected_model_for_task_type():
    manager = ActiveLearningManager(
        model_registry=Mock(), cache=Mock(), content_addressed_artifact_cache=Mock()
    )
    manager._models["project/1:package:variant"] = PackageModel(
        "project/1", "key", backend="trt"
    )
    request = SimpleNamespace(
        model_id="project/1",
        backend="trt",
        api_key="key",
        image=[],
        active_learning_target_dataset=None,
        id="request-id",
    )
    with patch(
        "inference.core.managers.active_learning.ActiveLearningMiddleware.init"
    ) as initialize:
        manager.register(
            prediction=Mock(), model_id="project/1:package:variant", request=request
        )
    assert initialize.call_args.kwargs["model_id"] == "project/1"
    assert (
        initialize.return_value.register_batch.call_args.kwargs["prediction_type"]
        == "object-detection"
    )


def test_active_request_finishes_on_its_instance_after_lru_eviction(package_manager):
    started, finish = Event(), Event()

    class BlockingPackageModel(PackageModel):
        def infer_from_request(self, request):
            if self.resolved_model.backend == "onnx":
                started.set()
                assert finish.wait(timeout=10)
            return super().infer_from_request(request)

    package_manager.model_manager.model_registry.get_model.return_value = (
        BlockingPackageModel
    )
    package_manager.max_size = 1
    package_manager.add_model("project/1", "key")
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(
            package_manager.infer_from_request_sync, "project/1", SimpleNamespace()
        )
        try:
            assert started.wait(timeout=10)
            package_manager.add_model(
                "project/1", "key", backend="trt", model_cache_key="trt"
            )
            assert set(package_manager.keys()) == {"trt"}
            assert package_manager.infer_from_request_sync(
                "trt", SimpleNamespace()
            ) == {"backend": "trt"}
        finally:
            finish.set()
        assert active.result(timeout=10) == {"backend": "onnx"}
