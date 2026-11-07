from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.entities.responses.embeddings import ImageEmbeddingResponse
from inference.models.pretrained_image_embeddings import (
    pretrained_image_embeddings_inference_models,
)
from inference.models.pretrained_image_embeddings.pretrained_image_embeddings_inference_models import (
    InferenceModelsPretrainedImageEmbeddingAdapter,
)
from inference_models.errors import ModelInputError

DINOV3_PREPROCESSING = {
    "image_processor_type": "DINOv3ViTImageProcessor",
    "do_resize": True,
    "size": {"height": 224, "width": 224},
    "image_mean": [0.485, 0.456, 0.406],
}


class FakePretrainedEmbeddingModel:
    def __init__(self, preprocessing):
        self.embedding_preprocessing = preprocessing
        self.embed_images = MagicMock(side_effect=self._embed)

    @staticmethod
    def _embed(images, output_type=None, input_color_format=None):
        batch_size = len(images) if isinstance(images, list) else images.shape[0]
        vectors = torch.arange(batch_size * 3, dtype=torch.float32)
        return vectors.reshape(batch_size, 3) - 2.5

    def get_embedding_info(self, output_type=None):
        return {
            "feature_definition": "dinov3-cls-token@v1",
            "output_type": "feature_vector",
            "normalization": "none",
        }


def load_adapter(monkeypatch, model_id="dinov3/vits16", preprocessing=None, **kwargs):
    backend = FakePretrainedEmbeddingModel(preprocessing or DINOV3_PREPROCESSING)
    load_model = MagicMock(return_value=backend)
    monkeypatch.setattr(
        pretrained_image_embeddings_inference_models.AutoModel,
        "from_pretrained",
        load_model,
    )
    adapter = InferenceModelsPretrainedImageEmbeddingAdapter(
        model_id=model_id, api_key="key", **kwargs
    )
    return adapter, backend, load_model


def embedding_request(images, is_batch=True, **kwargs):
    request_images = [{"type": "numpy_object", "value": image} for image in images]
    return ImageEmbeddingRequest(
        model_id="dinov3/vits16",
        image=request_images if is_batch else request_images[0],
        **kwargs,
    )


@pytest.mark.parametrize("is_batch", [False, True])
def test_pretrained_embeddings_return_raw_vectors_with_model_preprocessing(
    monkeypatch, is_batch
) -> None:
    adapter, backend, load_model = load_adapter(
        monkeypatch,
        required_capabilities=["image_embeddings"],
        output_type="feature_vector",
    )
    images = [np.full((8, 6, 3), value, dtype=np.uint8) for value in (20, 50)]
    if not is_batch:
        images = images[:1]
    request = embedding_request(images, is_batch=is_batch)

    response = adapter.infer_embeddings_from_request(request)

    assert isinstance(response, ImageEmbeddingResponse)
    expected = backend._embed(images).tolist()
    assert response.embeddings == expected
    assert response.inference_id == request.id
    assert response.time >= 0
    info = response.embedding_info
    assert info.model_id == "dinov3/vits16"
    assert info.feature_definition == "dinov3-cls-token@v1"
    assert info.output_type == "feature_vector"
    assert info.normalization == "none"
    assert info.dimension == 3
    assert info.preprocessing == {
        **DINOV3_PREPROCESSING,
        "overrides": {"disable_preproc_auto_orient": False},
    }
    assert info.backend == "FakePretrainedEmbeddingModel"
    assert info.precision == "torch.float32"
    passed_images = backend.embed_images.call_args.args[0]
    assert len(passed_images) == len(images)
    for actual, image in zip(passed_images, images):
        np.testing.assert_array_equal(actual, image)
    assert backend.embed_images.call_args.kwargs["input_color_format"] == "bgr"
    assert load_model.call_args.kwargs["model_id_or_path"] == "dinov3/vits16"
    assert load_model.call_args.kwargs["required_capabilities"] == ["image_embeddings"]
    assert load_model.call_args.kwargs["output_type"] == "feature_vector"


def test_space_id_follows_model_and_preprocessing(monkeypatch) -> None:
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]

    def space_id(model_id, preprocessing):
        adapter, _, _ = load_adapter(
            monkeypatch, model_id=model_id, preprocessing=preprocessing
        )
        response = adapter.infer_embeddings_from_request(embedding_request(images))
        return response.embedding_info.space_id

    reference = space_id("dinov3/vits16", DINOV3_PREPROCESSING)
    assert space_id("dinov3/vits16", dict(DINOV3_PREPROCESSING)) == reference
    assert (
        space_id(
            "dinov3/vits16",
            {**DINOV3_PREPROCESSING, "size": {"height": 256, "width": 256}},
        )
        != reference
    )
    assert space_id("dinov2/small", DINOV3_PREPROCESSING) != reference


def test_tensor_native_embeddings_keep_model_tensors(monkeypatch) -> None:
    adapter, backend, _ = load_adapter(monkeypatch)
    images = [torch.zeros((3, 4, 4), dtype=torch.uint8) for _ in range(2)]
    http_response = adapter.infer_embeddings_from_request(
        embedding_request([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)
    )
    vectors = backend._embed(images)
    backend.embed_images.side_effect = None
    backend.embed_images.return_value = vectors

    def reject_conversion(*args, **kwargs):
        raise AssertionError("CPU/Python conversion")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", reject_conversion)
        guard.setattr(torch.Tensor, "tolist", reject_conversion)
        native = adapter.run_tensor_native_embeddings(
            images=images, input_color_format="rgb", disable_preproc_contrast=True
        )

    assert native["embeddings"].data_ptr() == vectors.data_ptr()
    assert native["embedding_info"] == http_response.embedding_info.model_dump(
        exclude_none=True
    )
    assert backend.embed_images.call_args.args[0] is images
    assert backend.embed_images.call_args.kwargs["input_color_format"] == "rgb"


def test_logits_are_rejected_as_bad_input_before_loading_weights(monkeypatch) -> None:
    load_model = MagicMock()
    monkeypatch.setattr(
        pretrained_image_embeddings_inference_models.AutoModel,
        "from_pretrained",
        load_model,
    )

    with pytest.raises(ModelInputError):
        InferenceModelsPretrainedImageEmbeddingAdapter(
            model_id="siglip2/base-patch16-224",
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
