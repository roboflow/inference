import clip
import numpy as np
import pytest
import torch
from PIL import Image

from inference_exp.models.clip.clip_pytorch import ClipTorch


@pytest.fixture(scope="module")
def clip_model_name() -> str:
    return "ViT-B/32"


@pytest.fixture(scope="module")
def baseline_clip_model(clip_model_name: str):
    model, _ = clip.load(clip_model_name, device="cpu")
    return model


@pytest.fixture(scope="module")
def clip_torch_wrapper(clip_model_name: str) -> ClipTorch:
    return ClipTorch.from_pretrained(model_name_or_path=clip_model_name, device="cpu")


@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (448, 448)])
def test_clip_torch_vs_baseline_for_image_embeddings(
    clip_torch_wrapper: ClipTorch,
    baseline_clip_model,
    image_shape: tuple,
):
    # given
    image = np.random.randint(
        0, 255, size=(image_shape[0], image_shape[1], 3), dtype=np.uint8
    )
    preprocessor = clip.clip._transform(baseline_clip_model.visual.input_resolution)
    baseline_input = preprocessor(Image.fromarray(image)).unsqueeze(0)

    # when
    with torch.no_grad():
        baseline_embeddings = baseline_clip_model.encode_image(baseline_input)
    wrapper_embeddings = clip_torch_wrapper.embed_images(
        images=[image], input_color_format="rgb"
    )

    # then
    assert np.allclose(
        baseline_embeddings.numpy(), wrapper_embeddings.numpy(), atol=1e-2
    )


def test_clip_torch_vs_baseline_for_text_embeddings(
    clip_torch_wrapper: ClipTorch,
    baseline_clip_model,
):
    # given
    text = "hello world"
    baseline_input = clip.tokenize([text])

    # when
    with torch.no_grad():
        baseline_embeddings = baseline_clip_model.encode_text(baseline_input)
    wrapper_embeddings = clip_torch_wrapper.embed_text(texts=[text])

    # then
    assert np.allclose(
        baseline_embeddings.numpy(), wrapper_embeddings.numpy(), atol=1e-4
    )


def test_embed_text(clip_torch_wrapper: ClipTorch):
    # given
    texts = ["hello world", "this is a test"]

    # when
    embeddings = clip_torch_wrapper.embed_text(texts)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 512)


@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_single_numpy_image(clip_torch_wrapper: ClipTorch, image_shape: tuple):
    # given
    image = np.random.randint(
        0, 255, size=(image_shape[0], image_shape[1], 3), dtype=np.uint8
    )

    # when
    embeddings = clip_torch_wrapper.embed_images(image)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 512)


@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_single_tensor_image(clip_torch_wrapper: ClipTorch, image_shape: tuple):
    # given
    image = torch.randint(
        0, 255, size=(3, image_shape[0], image_shape[1]), dtype=torch.uint8
    )

    # when
    embeddings = clip_torch_wrapper.embed_images(image)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 512)


@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_list_of_numpy_images(clip_torch_wrapper: ClipTorch, image_shape: tuple):
    # given
    images = [
        np.random.randint(
            0, 255, size=(image_shape[0], image_shape[1], 3), dtype=np.uint8
        ),
        np.random.randint(0, 255, size=(320, 240, 3), dtype=np.uint8),
    ]

    # when
    embeddings = clip_torch_wrapper.embed_images(images)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 512)


@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_list_of_tensor_images(clip_torch_wrapper: ClipTorch, image_shape: tuple):
    # given
    images = [
        torch.randint(
            0, 255, size=(3, image_shape[0], image_shape[1]), dtype=torch.uint8
        ),
        torch.randint(0, 255, size=(3, 320, 240), dtype=torch.uint8),
    ]

    # when
    embeddings = clip_torch_wrapper.embed_images(images)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 512)


@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_batch_of_tensor_images(
    clip_torch_wrapper: ClipTorch, image_shape: tuple
):
    # given
    images = torch.randint(
        0, 255, size=(2, 3, image_shape[0], image_shape[1]), dtype=torch.uint8
    )

    # when
    embeddings = clip_torch_wrapper.embed_images(images)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 512)
