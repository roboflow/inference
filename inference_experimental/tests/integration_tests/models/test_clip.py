import torch
import numpy as np
import pytest
from PIL import Image
from typing import Callable, Union
import clip

from inference_exp.models.clip.clip_pytorch import ClipTorch, create_preprocessor


def create_pil_preprocessor(image_size: int) -> Callable:
    _, pil_based_transform = clip.load("ViT-B/32")

    def _pil_preprocessor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            # Assumes RGB numpy array
            pil_image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Assumes RGB tensor
            image_numpy = image.permute(1, 2, 0).cpu().numpy()
            if image_numpy.dtype in [np.float32, np.float64]:
                image_numpy = (image_numpy * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_numpy)
        else:
            raise TypeError("Unsupported image type for PIL preprocessor")

        preprocessed_image = pil_based_transform(pil_image)
        return preprocessed_image.unsqueeze(0)

    return _pil_preprocessor


@pytest.mark.parametrize("model_size", [224])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_preprocessing_consistency(model_size, image_shape):
    # given
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)

    # Create identical base data
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

    # when
    # --- Test with single NumPy input ---
    numpy_tensor_output = tensor_based_preprocessor(bgr_image_numpy.copy())
    numpy_pil_output = pil_based_preprocessor(rgb_image_numpy)

    # --- Test with single Tensor input ---
    tensor_tensor_output = tensor_based_preprocessor(rgb_image_tensor.clone())
    tensor_pil_output = pil_based_preprocessor(rgb_image_tensor.clone())

    # --- Test with a list of numpy.ndarray ---
    bgr_images_numpy = [bgr_image_numpy.copy(), bgr_image_numpy.copy()]
    rgb_images_numpy = [rgb_image_numpy.copy(), rgb_image_numpy.copy()]
    list_numpy_tensor_output = tensor_based_preprocessor(bgr_images_numpy)
    list_numpy_pil_output = torch.cat(
        [pil_based_preprocessor(img) for img in rgb_images_numpy], dim=0
    )

    # --- Test with a list of torch.Tensor ---
    list_rgb_tensors = [rgb_image_tensor.clone(), rgb_image_tensor.clone()]
    list_tensor_tensor_output = tensor_based_preprocessor(list_rgb_tensors)
    list_tensor_pil_output = torch.cat(
        [pil_based_preprocessor(t) for t in list_rgb_tensors], dim=0
    )

    # --- Test with a batched torch.Tensor (N, C, H, W) ---
    batched_tensor = torch.stack([rgb_image_tensor, rgb_image_tensor])
    batch_tensor_output = tensor_based_preprocessor(batched_tensor)
    batched_pil_output = torch.cat(
        [pil_based_preprocessor(img) for img in batched_tensor], dim=0
    )

    # --- Test with a list of numpy.ndarray of different sizes ---
    bgr_image_numpy_2 = np.random.randint(0, 256, size=(300, 300, 3), dtype=np.uint8)
    rgb_image_numpy_2 = bgr_image_numpy_2[:, :, ::-1]
    bgr_images_varied = [bgr_image_numpy.copy(), bgr_image_numpy_2]
    rgb_images_varied = [rgb_image_numpy.copy(), rgb_image_numpy_2]
    list_varied_numpy_tensor_output = tensor_based_preprocessor(bgr_images_varied)
    list_varied_numpy_pil_output = torch.cat(
        [pil_based_preprocessor(img) for img in rgb_images_varied], dim=0
    )

    # then
    # --- Assert single inputs ---
    assert torch.allclose(numpy_tensor_output, numpy_pil_output, atol=1e-5)
    assert torch.allclose(tensor_tensor_output, tensor_pil_output, atol=1e-5)
    assert torch.allclose(numpy_tensor_output, tensor_tensor_output, atol=1e-5)

    # --- Assert list inputs ---
    assert torch.allclose(list_numpy_tensor_output, list_numpy_pil_output, atol=1e-5)
    assert torch.allclose(list_tensor_tensor_output, list_tensor_pil_output, atol=1e-5)

    # --- Assert batched inputs ---
    assert torch.allclose(batch_tensor_output, batched_pil_output, atol=1e-5)
    assert torch.allclose(list_tensor_tensor_output, batch_tensor_output, atol=1e-5)

    # --- Assert varied size list inputs ---
    assert torch.allclose(
        list_varied_numpy_tensor_output, list_varied_numpy_pil_output, atol=1e-5
    )
