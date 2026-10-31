import base64
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from inference.core.entities.requests.clip import ClipImageEmbeddingRequest
from inference.models.clip import clip_inference_models


@pytest.fixture
def clip_adapter(monkeypatch):
    backend = Mock(spec=["embed_images", "embed_text", "_image_size", "resolved_model"])
    backend._image_size = 224
    backend.resolved_model = None
    monkeypatch.setattr(clip_inference_models, "API_KEY", None)
    monkeypatch.setattr(
        clip_inference_models.AutoModel,
        "from_pretrained",
        Mock(return_value=backend),
    )
    adapter = clip_inference_models.InferenceModelsClipAdapter(model_id="clip/test")

    return adapter, backend


@pytest.mark.parametrize(
    "shapes,batched",
    [
        ([(4, 6, 3)], False),
        ([(4, 6, 3)], True),
        ([(4, 6, 3)] * 3, True),
        ([(4, 6, 3), (8, 6, 3), (2, 6, 3)], True),
        ([(4, 6, 3), (8, 5, 3), (2, 9, 3)], True),
    ],
)
def test_embed_image_preserves_images(clip_adapter, shapes, batched):
    adapter, backend = clip_adapter
    images = [np.full(shape, i + 1, dtype=np.uint8) for i, shape in enumerate(shapes)]
    expected_embeddings = torch.arange(len(images) * 2).reshape(len(images), 2)
    backend.embed_images.return_value = expected_embeddings

    result = adapter.embed_image(images if batched else images[0])

    backend.embed_images.assert_called_once()
    actual_images = backend.embed_images.call_args.kwargs["images"]
    if isinstance(actual_images, np.ndarray):
        actual_images = [actual_images]
    assert len(actual_images) == len(images)
    for actual, expected in zip(actual_images, images):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(result, expected_embeddings.numpy())


@pytest.mark.parametrize("as_mapping", [False, True])
def test_compare_preserves_image_prompts(clip_adapter, as_mapping):
    adapter, backend = clip_adapter
    images = [np.full((4, 6, 3), i + 1, dtype=np.uint8) for i in range(3)]
    prompts = dict(zip(["first", "second", "third"], images)) if as_mapping else images
    backend.embed_text.return_value = torch.tensor([[1.0, 0.0]])
    backend.embed_images.return_value = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )

    result = adapter.compare(
        "subject", prompts, subject_type="text", prompt_type="image"
    )

    actual_images = backend.embed_images.call_args.kwargs["images"]
    assert isinstance(actual_images, list)
    assert len(actual_images) == 3
    for actual, expected in zip(actual_images, images):
        np.testing.assert_array_equal(actual, expected)
    if as_mapping:
        assert list(result) == ["first", "second", "third"]
        result = list(result.values())
    np.testing.assert_allclose(np.asarray(result).reshape(-1), [1.0, 0.0, -1.0])


def test_image_request_preserves_base64_batch(clip_adapter):
    adapter, backend = clip_adapter
    images = [np.full((4, 6, 3), i + 1, dtype=np.uint8) for i in range(3)]
    request_images = []
    for image in images:
        success, encoded = cv2.imencode(".png", image)
        assert success
        request_images.append(
            {"type": "base64", "value": base64.b64encode(encoded).decode("ascii")}
        )
    backend.embed_images.return_value = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )

    response = adapter.infer_from_request(
        ClipImageEmbeddingRequest(image=request_images)
    )

    actual_images = backend.embed_images.call_args.kwargs["images"]
    assert isinstance(actual_images, list)
    assert len(actual_images) == 3
    for actual, expected in zip(actual_images, images):
        np.testing.assert_array_equal(actual, expected)
    assert response.embeddings == [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]


def test_embed_image_rejects_oversized_batch_before_loading(clip_adapter, monkeypatch):
    adapter, backend = clip_adapter
    monkeypatch.setattr(clip_inference_models, "CLIP_MAX_BATCH_SIZE", 2)
    loader = Mock(side_effect=AssertionError("Oversized inputs must not be loaded"))
    monkeypatch.setattr(clip_inference_models, "load_image_bgr", loader)

    with pytest.raises(ValueError, match="maximum number of images.*2"):
        adapter.embed_image([object(), object(), object()])

    loader.assert_not_called()
    backend.embed_images.assert_not_called()
