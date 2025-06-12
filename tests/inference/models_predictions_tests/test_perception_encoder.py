import base64
import os
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderCompareRequest,
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.models.perception_encoder import PerceptionEncoder


@pytest.fixture
def test_image():
    # Create a simple test image
    image = Image.new("RGB", (224, 224), color="red")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return InferenceRequestImage(type="base64", value=img_base64)


@pytest.fixture
def model():
    # Initialize model with CPU for testing
    return PerceptionEncoder(device="cpu")


def test_model_initialization(model):
    """Test that the model initializes correctly."""
    assert model.device == "cpu"
    assert model.model is not None
    assert model.preprocessor is not None
    assert model.tokenizer is not None
    assert model.task_type == "embedding"


def test_image_embedding(model, test_image):
    """Test image embedding functionality."""
    # Test single image embedding
    request = PerceptionEncoderImageEmbeddingRequest(image=test_image)
    response = model.infer_from_request(request)

    assert response.embeddings is not None
    assert isinstance(response.embeddings, list)
    assert len(response.embeddings) > 0
    assert isinstance(response.embeddings[0], list)
    assert isinstance(response.embeddings[0][0], float)


def test_text_embedding(model):
    """Test text embedding functionality."""
    # Test single text embedding
    request = PerceptionEncoderTextEmbeddingRequest(text="a red car")
    response = model.infer_from_request(request)

    assert response.embeddings is not None
    assert isinstance(response.embeddings, list)
    assert len(response.embeddings) > 0
    assert isinstance(response.embeddings[0], list)
    assert isinstance(response.embeddings[0][0], float)


def test_text_embedding_directly(model):
    prompt = "can"
    text_embed = model.embed_text(prompt)
    assert text_embed is not None


def test_batch_text_embedding(model):
    """Test batch text embedding functionality."""
    texts = ["a red car", "a blue truck"]
    request = PerceptionEncoderTextEmbeddingRequest(text=texts)
    response = model.infer_from_request(request)

    assert response.embeddings is not None
    assert isinstance(response.embeddings, list)
    assert len(response.embeddings) == len(texts)
    assert isinstance(response.embeddings[0], list)
    assert isinstance(response.embeddings[0][0], float)


def test_image_text_comparison(model, test_image):
    """Test image-text comparison functionality."""
    request = PerceptionEncoderCompareRequest(
        subject=test_image,
        prompt="a red car",
        subject_type="image",
        prompt_type="text",
    )
    response = model.infer_from_request(request)

    assert response.similarity is not None
    assert isinstance(response.similarity, list)
    assert len(response.similarity) == 1
    assert isinstance(response.similarity[0], float)


def test_text_text_comparison(model):
    """Test text-text comparison functionality."""
    request = PerceptionEncoderCompareRequest(
        subject="a red car",
        prompt="a blue truck",
        subject_type="text",
        prompt_type="text",
    )
    response = model.infer_from_request(request)

    assert response.similarity is not None
    assert isinstance(response.similarity, list)
    assert len(response.similarity) == 1
    assert isinstance(response.similarity[0], float)


def test_invalid_request_type(model):
    """Test handling of invalid request type."""
    with pytest.raises(ValueError):
        model.infer_from_request("invalid_request")


def test_invalid_subject_type(model, test_image):
    """Test handling of invalid subject type."""
    with pytest.raises(ValueError):
        request = PerceptionEncoderCompareRequest(
            subject=test_image,
            prompt="a red car",
            subject_type="invalid",
            prompt_type="text",
        )
        model.infer_from_request(request)


def test_invalid_prompt_type(model, test_image):
    """Test handling of invalid prompt type."""
    with pytest.raises(ValueError):
        request = PerceptionEncoderCompareRequest(
            subject=test_image,
            prompt="a red car",
            subject_type="image",
            prompt_type="invalid",
        )
        model.infer_from_request(request)
