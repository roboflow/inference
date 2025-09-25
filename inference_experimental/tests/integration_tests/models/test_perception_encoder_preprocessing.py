from typing import Callable, Union

import numpy as np
import pytest
import torch
from inference_exp.models.perception_encoder.perception_encoder_pytorch import (
    create_preprocessor,
)
from inference_exp.models.perception_encoder.vision_encoder import transforms
from PIL import Image


def create_pil_preprocessor(image_size: int) -> Callable:
    """Factory to create a preprocessor that mimics the original PIL-based implementation."""
    original_pil_transform = transforms.get_image_transform(image_size)

    def _pil_preprocessor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocesses an image using the original PIL-based pipeline."""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image_numpy = image.permute(1, 2, 0).cpu().numpy()
            if image_numpy.dtype in [np.float32, np.float64]:
                image_numpy = (image_numpy * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_numpy)
        else:
            raise TypeError("Unsupported image type for PIL preprocessor")

        preprocessed_image = original_pil_transform(pil_image)
        return preprocessed_image.unsqueeze(0)

    return _pil_preprocessor


def get_image_data(image_shape):
    generator = torch.Generator().manual_seed(42)
    rgb_image_tensor = torch.randint(
        0,
        256,
        size=(3, image_shape[0], image_shape[1]),
        dtype=torch.uint8,
        generator=generator,
    )
    bgr_image_numpy = rgb_image_tensor.permute(1, 2, 0).numpy()[:, :, ::-1]
    rgb_image_numpy = bgr_image_numpy[:, :, ::-1]
    return rgb_image_tensor, bgr_image_numpy, rgb_image_numpy


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_single_numpy_input(model_size, image_shape):
    # GIVEN
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)
    _, bgr_image_numpy, rgb_image_numpy = get_image_data(image_shape)
    # WHEN
    tensor_output = tensor_based_preprocessor(bgr_image_numpy.copy())
    pil_output = pil_based_preprocessor(rgb_image_numpy)
    # THEN
    assert torch.allclose(tensor_output, pil_output, atol=1e-2)


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_single_tensor_input(model_size, image_shape):
    # GIVEN
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)
    rgb_image_tensor, _, _ = get_image_data(image_shape)
    # WHEN
    tensor_output = tensor_based_preprocessor(rgb_image_tensor)
    pil_output = pil_based_preprocessor(rgb_image_tensor)
    # THEN
    assert torch.allclose(tensor_output, pil_output, atol=1e-2)


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_list_of_numpy_inputs(model_size, image_shape):
    # GIVEN
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)
    _, bgr_image_numpy, rgb_image_numpy = get_image_data(image_shape)
    bgr_images_numpy = [bgr_image_numpy.copy(), bgr_image_numpy.copy()]
    rgb_images_numpy = [rgb_image_numpy.copy(), rgb_image_numpy.copy()]
    # WHEN
    tensor_output = tensor_based_preprocessor(bgr_images_numpy)
    pil_output = torch.cat(
        [pil_based_preprocessor(img) for img in rgb_images_numpy], dim=0
    )
    # THEN
    assert torch.allclose(tensor_output, pil_output, atol=1e-2)


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_list_of_tensor_inputs(model_size, image_shape):
    # GIVEN
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)
    rgb_image_tensor, _, _ = get_image_data(image_shape)
    list_rgb_tensors = [rgb_image_tensor.clone(), rgb_image_tensor.clone()]
    # WHEN
    tensor_output = tensor_based_preprocessor(list_rgb_tensors)
    pil_output = torch.cat([pil_based_preprocessor(t) for t in list_rgb_tensors], dim=0)
    # THEN
    assert torch.allclose(tensor_output, pil_output, atol=1e-2)


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_batched_tensor_input(model_size, image_shape):
    # GIVEN
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)
    rgb_image_tensor, _, _ = get_image_data(image_shape)
    batched_tensor = torch.stack([rgb_image_tensor, rgb_image_tensor])
    # WHEN
    tensor_output = tensor_based_preprocessor(batched_tensor)
    pil_output = torch.cat(
        [pil_based_preprocessor(img) for img in batched_tensor], dim=0
    )
    # THEN
    assert torch.allclose(tensor_output, pil_output, atol=1e-2)


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_list_of_varied_size_numpy_inputs(model_size, image_shape):
    # GIVEN
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)
    _, bgr_image_numpy, rgb_image_numpy = get_image_data(image_shape)
    bgr_image_numpy_2 = np.random.randint(0, 256, size=(300, 300, 3), dtype=np.uint8)
    rgb_image_numpy_2 = bgr_image_numpy_2[:, :, ::-1]
    bgr_images_varied = [bgr_image_numpy.copy(), bgr_image_numpy_2]
    rgb_images_varied = [rgb_image_numpy.copy(), rgb_image_numpy_2]
    # WHEN
    tensor_output = tensor_based_preprocessor(bgr_images_varied)
    pil_output = torch.cat(
        [pil_based_preprocessor(img) for img in rgb_images_varied], dim=0
    )
    # THEN
    assert torch.allclose(tensor_output, pil_output, atol=1e-2)


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_internal_consistency_of_tensor_inputs(model_size, image_shape):
    # GIVEN
    tensor_based_preprocessor = create_preprocessor(model_size)
    rgb_image_tensor, _, _ = get_image_data(image_shape)
    list_rgb_tensors = [rgb_image_tensor.clone(), rgb_image_tensor.clone()]
    batched_tensor = torch.stack(list_rgb_tensors)
    # WHEN
    list_tensor_output = tensor_based_preprocessor(list_rgb_tensors)
    batched_tensor_output = tensor_based_preprocessor(batched_tensor)
    # THEN
    assert torch.allclose(list_tensor_output, batched_tensor_output, atol=1e-2)


@pytest.mark.torch_models
@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_internal_consistency_of_numpy_and_tensor_inputs(model_size, image_shape):
    # GIVEN
    tensor_based_preprocessor = create_preprocessor(model_size)
    rgb_image_tensor, bgr_image_numpy, _ = get_image_data(image_shape)
    # WHEN
    numpy_output = tensor_based_preprocessor(bgr_image_numpy.copy())
    tensor_output = tensor_based_preprocessor(rgb_image_tensor)
    # THEN
    assert torch.allclose(numpy_output, tensor_output, atol=1e-2)
