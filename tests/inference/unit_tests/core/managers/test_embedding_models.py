from unittest.mock import MagicMock

import pytest

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.exceptions import ModelDeploymentNotSupportedError
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.lazy import _LazyModelClass, _LazyModelRegistry
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.pretrained_image_embeddings import (
    pretrained_image_embeddings_inference_models,
)
from inference.usage_tracking.model_types import clear_recorded_model_descriptors


@pytest.mark.parametrize("decorated", [False, True])
def test_classifier_and_embedding_models_have_separate_cache_entries(decorated):
    registry = MagicMock()
    manager = ModelManager(registry)
    if decorated:
        manager = WithFixedSizeCache(manager, max_size=3)
    manager.add_model("classifiers/4", "key", model_id_alias="resnet101")
    manager.add_model(
        "classifiers/4",
        "key",
        model_id_alias="resnet101",
        required_capabilities=["image_embeddings"],
    )
    assert "resnet101" in manager
    assert "resnet101:capabilities=image_embeddings" in manager
    assert registry.get_model.call_count == 2

    assert all(
        call.args[0] == "resnet101" for call in registry.get_model.call_args_list
    )
    assert registry.get_model.return_value.call_args.kwargs[
        "required_capabilities"
    ] == ["image_embeddings"]
    manager.add_model(
        "classifiers/4",
        "key",
        model_id_alias="resnet101",
        required_capabilities=["image_embeddings"],
    )
    assert registry.get_model.call_count == 2

    manager.add_model(
        "classifiers/4",
        "key",
        model_id_alias="resnet101",
        required_capabilities=["image_embeddings"],
        output_type="logits",
    )
    assert "resnet101:capabilities=image_embeddings:output_type=logits" in manager
    assert registry.get_model.call_count == 3
    assert registry.get_model.return_value.call_args.kwargs["output_type"] == "logits"
    manager.add_model(
        "classifiers/4",
        "key",
        model_id_alias="resnet101",
        required_capabilities=["image_embeddings"],
        output_type="logits",
    )
    assert registry.get_model.call_count == 3


def test_embedding_request_dispatches_feature_method():
    model = MagicMock()
    manager = ModelManager(MagicMock(), models={"embedding": model})
    request = ImageEmbeddingRequest(
        model_id="project/1",
        image={"type": "url", "value": "https://example.com/image.jpg"},
    )
    assert (
        manager.model_infer_sync("embedding", request)
        is model.infer_embeddings_from_request.return_value
    )
    model.infer_from_request.assert_not_called()


def test_capability_check_resolves_lazily_registered_models(monkeypatch):
    registry = RoboflowModelRegistry(
        _LazyModelRegistry(
            {
                ("embed", "dinov3"): _LazyModelClass(
                    "inference.models.pretrained_image_embeddings."
                    "pretrained_image_embeddings_inference_models:"
                    "InferenceModelsPretrainedImageEmbeddingAdapter",
                    optional=True,
                ),
                ("embed", "sam2"): _LazyModelClass("inference.core.models.base:Model"),
            }
        )
    )
    load_model = MagicMock()
    monkeypatch.setattr(
        pretrained_image_embeddings_inference_models.AutoModel,
        "from_pretrained",
        load_model,
    )
    manager = ModelManager(registry, content_addressed_artifact_cache=MagicMock())

    try:
        manager.add_model(
            "dinov3/vits16", "key", required_capabilities=["image_embeddings"]
        )
        with pytest.raises(
            ModelDeploymentNotSupportedError, match="pretrained embedding model"
        ):
            manager.add_model(
                "sam2/hiera_tiny", "key", required_capabilities=["image_embeddings"]
            )
    finally:
        clear_recorded_model_descriptors()

    assert "dinov3/vits16:capabilities=image_embeddings" in manager
    assert load_model.call_args.kwargs["required_capabilities"] == ["image_embeddings"]
