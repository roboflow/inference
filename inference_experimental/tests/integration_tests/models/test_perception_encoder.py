import torch
import numpy as np
import pytest
from PIL import Image
from typing import Callable, Union

from inference_exp.models.perception_encoder.perception_encoder import (
    PerceptionEncoder,
    get_tensor_image_transform,
    create_preprocessor,
)
from inference_exp.models.perception_encoder.vision_encoder import transforms


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



@pytest.mark.parametrize("model_size", [224, 336, 448])
@pytest.mark.parametrize("image_shape", [(224, 224), (336, 448), (640, 640)])
def test_preprocessing_consistency(model_size, image_shape):
    
    # Create a PIL-based preprocessor to compare against the tensor-based one
    # this is the original preprocessor used in the original implementation
    pil_based_preprocessor = create_pil_preprocessor(model_size)

    # this is the new preprocessor used in the new implementation since we dont want to convert to PIL image
    tensor_based_preprocessor = create_preprocessor(model_size)
    

    # Create identical base data
    generator = torch.Generator().manual_seed(42)
    rgb_image_tensor = torch.randint(0, 256, size=(3, image_shape[0], image_shape[1]), dtype=torch.uint8, generator=generator)
    bgr_image_numpy = rgb_image_tensor.permute(1, 2, 0).numpy()[:, :, ::-1]
    rgb_image_numpy = bgr_image_numpy[:, :, ::-1]

    # WHEN
    # --- Test with NumPy input ---
    numpy_tensor_output = tensor_based_preprocessor(bgr_image_numpy.copy())
    numpy_pil_output = pil_based_preprocessor(rgb_image_numpy)
    
    # --- Test with Tensor input ---
    tensor_tensor_output = tensor_based_preprocessor(rgb_image_tensor)
    tensor_pil_output = pil_based_preprocessor(rgb_image_tensor)

    # THEN
    assert torch.allclose(numpy_tensor_output, numpy_pil_output, atol=1e-2)
    assert torch.allclose(tensor_tensor_output, tensor_pil_output, atol=1e-2)
    # Also ensure that processing a numpy array vs a tensor gives the same result
    assert torch.allclose(numpy_tensor_output, tensor_tensor_output, atol=1e-2)


@pytest.mark.e2e_model_inference
def test_perception_encoder_text_embedding():
    # GIVEN
    model = PerceptionEncoder.from_pretrained(
        "/tmp/cache/perception_encoder/PE-Core-B16-224"
    )

    # WHEN
    embeddings = model.embed_text("hello world")

    # THEN
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024)


@pytest.mark.e2e_model_inference
def test_perception_encoder_image_embedding():
    # GIVEN
    model = PerceptionEncoder.from_pretrained(
        "/tmp/cache/perception_encoder/PE-Core-B16-224"
    )
    # Create a BGR numpy image, simulating a cv2.imread() output
    bgr_image_numpy = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

    # WHEN
    embeddings = model.embed_images(bgr_image_numpy)

    # THEN
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (1, 1024) 