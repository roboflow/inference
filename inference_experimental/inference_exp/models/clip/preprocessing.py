from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

from inference_exp.entities import ColorFormat
from inference_exp.errors import ModelRuntimeError

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def create_clip_preprocessor(image_size: int, device: torch.device) -> Callable:
    """
    Creates a preprocessor for CLIP models that operates on tensors.

    This implementation replicates the logic of the original CLIP preprocessing pipeline
    but is designed to work directly with torch.Tensors and np.ndarrays, avoiding
    the need to convert to and from PIL.Image objects.

    Note: Due to differences in the underlying resizing algorithms (torchvision vs. PIL),
    the output of this preprocessor may have minor numerical differences compared to
    the original. These differences have been tested and are known to produce
    embeddings with very high cosine similarity, making them functionally equivalent.

    Args:
        image_size (int): The target size for the input images.
        device (torch.device): The device to move the tensors to.

    Returns:
        A callable function that preprocesses images.
    """
    # This pre-processing pipeline matches the original CLIP implementation.
    # 1. Resize to `image_size`
    # 2. Center crop to `image_size`
    # 3. Scale pixel values to [0, 1]
    # 4. Normalize with CLIP's specific mean and standard deviation.
    transforms = Compose(
        [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            lambda x: x.to(torch.float32) / 255.0,
            Normalize(MEAN, STD),
        ]
    )
    return partial(preprocess_clip_image, transforms=transforms, device=device)


def preprocess_clip_image(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    transforms: Compose,
    device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
) -> torch.Tensor:
    if not isinstance(images, (list, np.ndarray, torch.Tensor)):
        raise ModelRuntimeError(
            f"Unsupported input type: {type(images)}. Must be one of list, np.ndarray, or torch.Tensor."
        )

    def _to_tensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            # HWC -> CHW
            tensor_image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            tensor_image = image

        # Default to BGR for numpy and RGB for tensor
        effective_color_format = input_color_format
        if effective_color_format is None:
            effective_color_format = "bgr" if is_numpy else "rgb"

        if effective_color_format == "bgr":
            # BGR -> RGB
            tensor_image = tensor_image[[2, 1, 0], :, :]

        return tensor_image.to(device)

    if isinstance(images, list):
        if not images:
            raise ModelRuntimeError("Input image list cannot be empty.")
        # Handle lists of varied-size images by applying transforms to each
        # before stacking. This is less efficient than batching but necessary.
        processed_images = [transforms(_to_tensor(img).unsqueeze(0)) for img in images]
        return torch.cat(processed_images, dim=0)
    else:
        # Handle single image or a batch of images thats already tensor
        tensor_batch = _to_tensor(images)
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)  # Add batch dimension

        return transforms(tensor_batch)
