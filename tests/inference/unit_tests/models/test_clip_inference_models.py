from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.entities.responses.embeddings import ImageEmbeddingResponse
from inference.models.clip import clip_inference_models
from inference.models.clip.clip_inference_models import InferenceModelsClipAdapter
from inference_models.errors import ModelInputError
from inference_models.models.clip.clip_onnx import ClipOnnx


def embed(images, input_color_format=None):
    batch_size = len(images) if isinstance(images, list) else images.shape[0]
    return torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4)


def weightless_clip_onnx(image_size):
    model = ClipOnnx.__new__(ClipOnnx)
    model._image_size = image_size
    model.embed_images = MagicMock(side_effect=embed)
    return model


def load_adapter(monkeypatch, model_id="clip/ViT-B-32", image_size=224, **kwargs):
    backend = weightless_clip_onnx(image_size=image_size)
    load_model = MagicMock(return_value=backend)
    monkeypatch.setattr(clip_inference_models.AutoModel, "from_pretrained", load_model)
    adapter = InferenceModelsClipAdapter(model_id=model_id, api_key="key", **kwargs)
    return adapter, backend, load_model


def embedding_request(images, is_batch=True, **kwargs):
    request_images = [{"type": "numpy_object", "value": image} for image in images]
    return ImageEmbeddingRequest(
        model_id="clip/ViT-B-32",
        image=request_images if is_batch else request_images[0],
        **kwargs,
    )


@pytest.mark.parametrize("is_batch", [False, True])
def test_clip_serves_image_embeddings_without_negotiating_the_capability(
    monkeypatch, is_batch
) -> None:
    adapter, backend, load_model = load_adapter(
        monkeypatch,
        required_capabilities=["image_embeddings"],
        output_type="feature_vector",
        countinference=False,
    )
    images = [np.full((6, 9, 3), value, dtype=np.uint8) for value in (10, 90)]
    if not is_batch:
        images = images[:1]
    request = embedding_request(images, is_batch=is_batch)

    response = adapter.infer_embeddings_from_request(request)

    assert "required_capabilities" not in load_model.call_args.kwargs
    assert "output_type" not in load_model.call_args.kwargs
    assert load_model.call_args.kwargs["countinference"] is False
    assert isinstance(response, ImageEmbeddingResponse)
    assert response.embeddings == embed(images).tolist()
    assert response.inference_id == request.id
    info = response.embedding_info
    assert info.model_id == "clip/ViT-B-32"
    assert info.feature_definition == "clip-image-embeds@v1"
    assert info.output_type == "feature_vector"
    assert info.normalization == "none"
    assert info.dimension == 4
    assert info.preprocessing == backend.embedding_preprocessing
    assert info.preprocessing["image_size"] == 224
    assert info.backend == "ClipOnnx"
    passed_images = backend.embed_images.call_args.kwargs["images"]
    assert len(passed_images) == len(images)
    for actual, image in zip(passed_images, images):
        np.testing.assert_array_equal(actual, image)
    assert backend.embed_images.call_args.kwargs["input_color_format"] == "bgr"


def test_clip_space_id_changes_with_version_and_input_size(monkeypatch) -> None:
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]

    def space_id(model_id, image_size):
        adapter, _, _ = load_adapter(
            monkeypatch, model_id=model_id, image_size=image_size
        )
        response = adapter.infer_embeddings_from_request(embedding_request(images))
        return response.embedding_info.space_id

    reference = space_id("clip/ViT-B-32", 224)
    assert space_id("clip/ViT-B-32", 224) == reference
    assert space_id("clip/ViT-B-32", 336) != reference
    assert space_id("clip/ViT-B-16", 224) != reference


def test_clip_tensor_native_embeddings_keep_model_tensors(monkeypatch) -> None:
    adapter, backend, _ = load_adapter(monkeypatch)
    images = [torch.zeros((3, 4, 4), dtype=torch.uint8)]
    vectors = embed(images)
    backend.embed_images.side_effect = None
    backend.embed_images.return_value = vectors

    native = adapter.run_tensor_native_embeddings(
        images=images, input_color_format="rgb"
    )

    assert native["embeddings"].data_ptr() == vectors.data_ptr()
    assert native["embedding_info"]["feature_definition"] == "clip-image-embeds@v1"
    assert backend.embed_images.call_args.kwargs["images"] is images
    assert backend.embed_images.call_args.kwargs["input_color_format"] == "rgb"


def test_clip_rejects_logits_as_bad_input(monkeypatch) -> None:
    load_model = MagicMock()
    monkeypatch.setattr(clip_inference_models.AutoModel, "from_pretrained", load_model)

    with pytest.raises(ModelInputError):
        InferenceModelsClipAdapter(
            model_id="clip/ViT-B-32",
            api_key="key",
            required_capabilities=["image_embeddings"],
            output_type="logits",
        )

    load_model.assert_not_called()
    adapter, backend, _ = load_adapter(monkeypatch)
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]
    with pytest.raises(ModelInputError):
        adapter.infer_embeddings_from_request(
            embedding_request(images, output_type="logits")
        )
    with pytest.raises(ModelInputError):
        adapter.run_tensor_native_embeddings(images=images, output_type="logits")
    backend.embed_images.assert_not_called()
