import numpy as np
import pytest
import torch
from inference_exp import AutoModel


@pytest.mark.e2e_model_inference
def test_perception_encoder_text_embedding():
    # GIVEN
    model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")

    # WHEN
    embeddings = model.embed_text("hello world")

    # THEN
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)


@pytest.mark.e2e_model_inference
def test_perception_encoder_image_embedding_with_numpy_inputs():
    # GIVEN
    model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")

    # WHEN & THEN

    # --- Test with a single numpy.ndarray ---
    bgr_image_numpy = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    embeddings = model.embed_images(bgr_image_numpy)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)

    # --- Test with a list of numpy.ndarray ---
    bgr_images_numpy = [
        np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8) for _ in range(2)
    ]
    embeddings = model.embed_images(bgr_images_numpy)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)


@pytest.mark.e2e_model_inference
def test_perception_encoder_image_embedding_with_tensor_inputs():
    # GIVEN
    model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")

    # WHEN & THEN

    # --- Test with a single torch.Tensor (C, H, W) ---
    rgb_image_tensor = torch.randint(0, 256, size=(3, 224, 224), dtype=torch.uint8)
    embeddings = model.embed_images(rgb_image_tensor)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)

    # --- Test with a list of torch.Tensor (C, H, W) ---
    rgb_images_tensor = [
        torch.randint(0, 256, size=(3, 224, 224), dtype=torch.uint8) for _ in range(2)
    ]
    embeddings = model.embed_images(rgb_images_tensor)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)

    # --- Test with a batched torch.Tensor (N, C, H, W) ---
    batched_rgb_images_tensor = torch.randint(
        0, 256, size=(2, 3, 224, 224), dtype=torch.uint8
    )
    embeddings = model.embed_images(batched_rgb_images_tensor)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)
