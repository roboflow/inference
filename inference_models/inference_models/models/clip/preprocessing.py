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

from inference_models.entities import ColorFormat
from inference_models.errors import ModelInputError

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

PreprocessorFun = Callable[
    [
        Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        Optional[ColorFormat],
        torch.device,
    ],
    torch.Tensor,
]


def create_clip_preprocessor(image_size: int) -> PreprocessorFun:
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
        image_size (int): The target size for the input images.`
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
    return partial(pre_process_image, transforms=transforms)


def pre_process_image(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat],
    device: torch.device,
    transforms: Compose,
) -> torch.Tensor:
    images = inputs_to_tensor(
        images=images, device=device, input_color_format=input_color_format
    )
    if isinstance(images, torch.Tensor):
        return transforms(images)
    return torch.cat([transforms(i) for i in images], dim=0).contiguous()


def inputs_to_tensor(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if not isinstance(images, (list, np.ndarray, torch.Tensor)):
        raise ModelInputError(
            message=f"Unsupported input type: {type(images)}. Must be one of list, np.ndarray, or torch.Tensor.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if isinstance(images, list):
        if not images:
            raise ModelInputError(
                message="Input image list cannot be empty.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        return [
            input_to_tensor(
                image=image,
                device=device,
                input_color_format=input_color_format,
                batched_tensors_allowed=False,
            )
            for image in images
        ]
    return input_to_tensor(
        image=images,
        device=device,
        input_color_format=input_color_format,
    ).contiguous()


def input_to_tensor(
    image: Union[torch.Tensor, np.ndarray],
    device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    batched_tensors_allowed: bool = True,
) -> torch.Tensor:
    if not isinstance(image, (np.ndarray, torch.Tensor)):
        raise ModelInputError(
            message=f"Unsupported input type: {type(image)}. Each element must be one of np.ndarray, or torch.Tensor.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    is_numpy = isinstance(image, np.ndarray)
    if is_numpy:
        if len(image.shape) != 3:
            raise ModelInputError(
                message=f"Unsupported input type: detected np.ndarray image of shape {image.shape} which has "
                f"number of dimensions different than 3. This input is invalid.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if image.shape[-1] != 3:
            raise ModelInputError(
                message="Unsupported input type: detected np.ndarray image of shape {image.shape} which has "
                f"incorrect number of color channels (expected: 3).",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        # HWC -> CHW
        tensor_image = torch.from_numpy(image).to(device).permute(2, 0, 1).unsqueeze(0)
    else:
        expected_dimensions_str = (
            "expected: 3 or 4" if batched_tensors_allowed else "expected: 3"
        )
        if len(image.shape) == 4 and not batched_tensors_allowed:
            raise ModelInputError(
                message="Unsupported input type: detected torch.Tensor image of shape {image.shape} which has "
                f"incorrect number of dimensions ({expected_dimensions_str}).",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if len(image.shape) != 3 and len(image.shape) != 4:
            raise ModelInputError(
                message=f"Unsupported input type: detected torch.Tensor image of shape {image.shape} which has "
                f"incorrect number of dimensions ({expected_dimensions_str}).",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if (len(image.shape) == 3 and image.shape[0] != 3) or (
            len(image.shape) == 4 and image.shape[1] != 3
        ):
            raise ModelInputError(
                message=f"Unsupported input type: detected torch.Tensor image of shape {image.shape} which has "
                f"incorrect number of color channels (expected: 3).",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        tensor_image = image.to(device)
    effective_color_format = input_color_format
    if effective_color_format is None:
        effective_color_format = "bgr" if is_numpy else "rgb"
    if effective_color_format == "bgr":
        # BGR -> RGB
        tensor_image = tensor_image[:, [2, 1, 0], :, :]
    return tensor_image
