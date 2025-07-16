import os


import clip
import numpy as np
import pytest
import torch
from PIL import Image

from inference_exp.models.clip.clip_onnx import ClipOnnx
from inference_exp.models.clip.clip_pytorch import ClipTorch
from inference_exp import AutoModel
from inference_exp.weights_providers.entities import BackendType


@pytest.fixture(scope="module")
def clip_model_name() -> str:
    return "RN50"


@pytest.fixture(scope="module")
def baseline_clip_model(clip_model_name: str):
    model, _ = clip.load(clip_model_name, device="cpu")
    return model


@pytest.fixture(scope="module")
def clip_torch_wrapper(clip_model_name: str) -> ClipTorch:
    return AutoModel.from_pretrained(
        model_name_or_path=f"clip/{clip_model_name}",
        device=torch.device("cpu"),
        backends=[BackendType.TORCH],
    )


@pytest.fixture(scope="module")
def clip_onnx_wrapper(clip_model_name: str) -> ClipOnnx:

    return AutoModel.from_pretrained(
        model_name_or_path=f"clip/{clip_model_name}",
        device=torch.device("cpu"),
        backends=[BackendType.ONNX],
    )


def _test_clip_wrapper_vs_baseline_for_image_embeddings(
    clip_wrapper,
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
    wrapper_embeddings = clip_wrapper.embed_images(
        images=[image], input_color_format="rgb"
    )

    # then
    similarity = torch.nn.functional.cosine_similarity(
        baseline_embeddings, wrapper_embeddings
    )
    assert similarity.item() > 0.99


@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (448, 448)])
def test_torch_clip_wrapper_vs_baseline_for_image_embeddings(
    clip_torch_wrapper: ClipTorch,
    baseline_clip_model,
    image_shape: tuple,
):
    _test_clip_wrapper_vs_baseline_for_image_embeddings(
        clip_wrapper=clip_torch_wrapper,
        baseline_clip_model=baseline_clip_model,
        image_shape=image_shape,
    )


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (448, 448)])
def test_onnx_clip_wrapper_vs_baseline_for_image_embeddings(
    clip_onnx_wrapper: ClipOnnx,
    baseline_clip_model,
    image_shape: tuple,
):
    _test_clip_wrapper_vs_baseline_for_image_embeddings(
        clip_wrapper=clip_onnx_wrapper,
        baseline_clip_model=baseline_clip_model,
        image_shape=image_shape,
    )


def _test_clip_wrapper_vs_baseline_for_text_embeddings(
    clip_wrapper,
    baseline_clip_model,
):
    # given
    text = "hello world"
    baseline_input = clip.tokenize([text])

    # when
    with torch.no_grad():
        baseline_embeddings = baseline_clip_model.encode_text(baseline_input)
    wrapper_embeddings = clip_wrapper.embed_text(texts=[text])

    # then
    similarity = torch.nn.functional.cosine_similarity(
        baseline_embeddings, wrapper_embeddings
    )
    assert similarity.item() > 0.999


@pytest.mark.e2e_model_inference
def test_torch_clip_wrapper_vs_baseline_for_text_embeddings(
    clip_torch_wrapper: ClipTorch,
    baseline_clip_model,
):
    _test_clip_wrapper_vs_baseline_for_text_embeddings(
        clip_wrapper=clip_torch_wrapper, baseline_clip_model=baseline_clip_model
    )


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
def test_onnx_clip_wrapper_vs_baseline_for_text_embeddings(
    clip_onnx_wrapper: ClipOnnx,
    baseline_clip_model,
):
    _test_clip_wrapper_vs_baseline_for_text_embeddings(
        clip_wrapper=clip_onnx_wrapper, baseline_clip_model=baseline_clip_model
    )


def _test_embed_text(clip_wrapper):
    # given
    texts = ["hello world", "this is a test"]

    # when
    embeddings = clip_wrapper.embed_text(texts)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)


@pytest.mark.e2e_model_inference
def test_torch_embed_text(clip_torch_wrapper: ClipTorch):
    _test_embed_text(clip_wrapper=clip_torch_wrapper)


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
def test_onnx_embed_text(clip_onnx_wrapper: ClipOnnx):
    _test_embed_text(clip_wrapper=clip_onnx_wrapper)


def _test_embed_single_numpy_image(clip_wrapper, image_shape: tuple):
    # given
    image = np.random.randint(
        0, 255, size=(image_shape[0], image_shape[1], 3), dtype=np.uint8
    )

    # when
    embeddings = clip_wrapper.embed_images(image)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)


@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_torch_embed_single_numpy_image(
    clip_torch_wrapper: ClipTorch, image_shape: tuple
):
    _test_embed_single_numpy_image(
        clip_wrapper=clip_torch_wrapper, image_shape=image_shape
    )


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_onnx_embed_single_numpy_image(clip_onnx_wrapper: ClipOnnx, image_shape: tuple):
    _test_embed_single_numpy_image(
        clip_wrapper=clip_onnx_wrapper, image_shape=image_shape
    )


def _test_embed_single_tensor_image(clip_wrapper, image_shape: tuple):
    # given
    image = torch.randint(
        0, 255, size=(3, image_shape[0], image_shape[1]), dtype=torch.uint8
    )

    # when
    embeddings = clip_wrapper.embed_images(image)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)


@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_torch_embed_single_tensor_image(
    clip_torch_wrapper: ClipTorch, image_shape: tuple
):
    _test_embed_single_tensor_image(
        clip_wrapper=clip_torch_wrapper, image_shape=image_shape
    )


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_onnx_embed_single_tensor_image(
    clip_onnx_wrapper: ClipOnnx, image_shape: tuple
):
    _test_embed_single_tensor_image(
        clip_wrapper=clip_onnx_wrapper, image_shape=image_shape
    )


def _test_embed_list_of_numpy_images(clip_wrapper, image_shape: tuple):
    # given
    images = [
        np.random.randint(
            0, 255, size=(image_shape[0], image_shape[1], 3), dtype=np.uint8
        ),
        np.random.randint(0, 255, size=(320, 240, 3), dtype=np.uint8),
    ]

    # when
    embeddings = clip_wrapper.embed_images(images)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)


@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_torch_embed_list_of_numpy_images(
    clip_torch_wrapper: ClipTorch, image_shape: tuple
):
    _test_embed_list_of_numpy_images(
        clip_wrapper=clip_torch_wrapper, image_shape=image_shape
    )


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_onnx_embed_list_of_numpy_images(
    clip_onnx_wrapper: ClipOnnx, image_shape: tuple
):
    _test_embed_list_of_numpy_images(
        clip_wrapper=clip_onnx_wrapper, image_shape=image_shape
    )


def _test_embed_list_of_tensor_images(clip_wrapper, image_shape: tuple):
    # given
    images = [
        torch.randint(
            0, 255, size=(3, image_shape[0], image_shape[1]), dtype=torch.uint8
        ),
        torch.randint(0, 255, size=(3, 320, 240), dtype=torch.uint8),
    ]

    # when
    embeddings = clip_wrapper.embed_images(images)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)


@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_torch_embed_list_of_tensor_images(
    clip_torch_wrapper: ClipTorch, image_shape: tuple
):
    _test_embed_list_of_tensor_images(
        clip_wrapper=clip_torch_wrapper, image_shape=image_shape
    )


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_onnx_embed_list_of_tensor_images(
    clip_onnx_wrapper: ClipOnnx, image_shape: tuple
):
    _test_embed_list_of_tensor_images(
        clip_wrapper=clip_onnx_wrapper, image_shape=image_shape
    )


def _test_embed_batch_of_tensor_images(clip_wrapper, image_shape: tuple):
    # given
    images = torch.randint(
        0, 255, size=(2, 3, image_shape[0], image_shape[1]), dtype=torch.uint8
    )

    # when
    embeddings = clip_wrapper.embed_images(images)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)


@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_torch_embed_batch_of_tensor_images(
    clip_torch_wrapper: ClipTorch, image_shape: tuple
):
    _test_embed_batch_of_tensor_images(
        clip_wrapper=clip_torch_wrapper, image_shape=image_shape
    )


@pytest.mark.onnx_extras
@pytest.mark.e2e_model_inference
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_onnx_embed_batch_of_tensor_images(
    clip_onnx_wrapper: ClipOnnx, image_shape: tuple
):
    _test_embed_batch_of_tensor_images(
        clip_wrapper=clip_onnx_wrapper, image_shape=image_shape
    )
