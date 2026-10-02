from unittest.mock import MagicMock

import pytest

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache


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
