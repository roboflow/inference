import os

os.environ["ROBOFLOW_API_HOST"] = "https://api.roboflow.one"


import clip
import numpy as np
import pytest
import torch
from PIL import Image

from inference_exp.models.clip.clip_onnx import ClipOnnx
from inference_exp.models.clip.clip_pytorch import ClipTorch


@pytest.fixture(scope="module")
def clip_model_name() -> str:
    return "RN50"


@pytest.fixture(scope="module")
def baseline_clip_model(clip_model_name: str):
    model, _ = clip.load(clip_model_name, device="cpu")
    return model


@pytest.fixture(scope="module")
def clip_torch_wrapper(clip_model_name: str) -> ClipTorch:
    return ClipTorch.from_pretrained(model_name_or_path=clip_model_name, device="cpu")


@pytest.fixture(scope="module")
def clip_onnx_wrapper(clip_model_name: str) -> ClipOnnx:
    return ClipOnnx.from_pretrained(
        model_name_or_path=f"clip/{clip_model_name}",
        device=torch.device("cpu"),
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (448, 448)])
def test_clip_wrapper_vs_baseline_for_image_embeddings(
    wrapper_name: str,
    request,
    baseline_clip_model,
    image_shape: tuple,
):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
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


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
def test_clip_wrapper_vs_baseline_for_text_embeddings(
    wrapper_name: str,
    request,
    baseline_clip_model,
):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
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


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
def test_embed_text(wrapper_name: str, request):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
    texts = ["hello world", "this is a test"]

    # when
    embeddings = clip_wrapper.embed_text(texts)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_single_numpy_image(wrapper_name: str, request, image_shape: tuple):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
    image = np.random.randint(
        0, 255, size=(image_shape[0], image_shape[1], 3), dtype=np.uint8
    )

    # when
    embeddings = clip_wrapper.embed_images(image)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_single_tensor_image(wrapper_name: str, request, image_shape: tuple):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
    image = torch.randint(
        0, 255, size=(3, image_shape[0], image_shape[1]), dtype=torch.uint8
    )

    # when
    embeddings = clip_wrapper.embed_images(image)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_list_of_numpy_images(wrapper_name: str, request, image_shape: tuple):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
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


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_list_of_tensor_images(wrapper_name: str, request, image_shape: tuple):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
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


@pytest.mark.parametrize("wrapper_name", ["clip_torch_wrapper", "clip_onnx_wrapper"])
@pytest.mark.parametrize("image_shape", [(224, 224), (320, 240), (640, 480)])
def test_embed_batch_of_tensor_images(wrapper_name: str, request, image_shape: tuple):
    # given
    clip_wrapper = request.getfixturevalue(wrapper_name)
    images = torch.randint(
        0, 255, size=(2, 3, image_shape[0], image_shape[1]), dtype=torch.uint8
    )

    # when
    embeddings = clip_wrapper.embed_images(images)

    # then
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1024)
