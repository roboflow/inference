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

        return pil_based_transform(pil_image)

    return _pil_preprocessor


@pytest.mark.parametrize("model_size", [224])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_preprocessing_consistency(model_size, image_shape):
    # given
    pil_based_preprocessor = create_pil_preprocessor(model_size)
    tensor_based_preprocessor = create_preprocessor(model_size)

    generator = torch.Generator().manual_seed(42)
    bgr_image_numpy = np.random.randint(
        0, 255, size=(image_shape[0], image_shape[1], 3), dtype=np.uint8
    )
    rgb_image_numpy = bgr_image_numpy[:, :, ::-1]

    # when
    # Test with single numpy input
    numpy_tensor_output = tensor_based_preprocessor(bgr_image_numpy)
    numpy_pil_output = pil_based_preprocessor(rgb_image_numpy)

    diff = torch.abs(numpy_tensor_output - numpy_pil_output.unsqueeze(0))
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")
    print(f"Number of elements exceeding tolerance: {(diff > 0.1).sum()}")
    exceeds_tol = diff > 0.1
    if exceeds_tol.any():
        indices = torch.where(exceeds_tol)
        print(f"Indices with large differences: {indices}")
        print(f"Values at those indices:")
        for i in range(min(5, len(indices[0]))):  # Show first 5
            idx = tuple(ind[i].item() for ind in indices)
            print(
                f"  Index {idx}: {numpy_tensor_output[idx].item():.4f} vs {numpy_pil_output.unsqueeze(0)[idx].item():.4f}"
            )

    # then
    assert torch.allclose(numpy_tensor_output, numpy_pil_output.unsqueeze(0), atol=1e-1)
