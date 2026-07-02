from typing import List, Optional, Union

import cv2
import numpy as np
import torch

from inference_models.entities import ColorFormat
from inference_models.errors import ModelInputError

FLOAT_UNIT_RANGE_MAX = 1.0 + 1e-6


def normalize_input_images(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat] = None,
) -> List[np.ndarray]:
    """Normalize supported image inputs into a list of BGR ``uint8`` arrays.

    Accepted pixel scales: integer arrays / tensors are assumed to be in
    ``[0, 255]``; floating-point arrays / tensors are assumed to be in
    ``[0, 1]`` and are rescaled, unless values above ``1.0`` are present, in
    which case they are treated as already being in ``[0, 255]``.
    """
    if isinstance(images, np.ndarray):
        input_color_format = input_color_format or "bgr"
        return [
            convert_numpy_image_to_bgr(image=images, color_format=input_color_format)
        ]
    if isinstance(images, torch.Tensor):
        input_color_format = input_color_format or "rgb"
        return [
            convert_numpy_image_to_bgr(
                image=image,
                color_format=input_color_format,
            )
            for image in torch_images_to_numpy_list(images=images)
        ]
    if not isinstance(images, list):
        raise ModelInputError(
            message="PP-OCRv6 models support np.ndarray, torch.Tensor, or lists of those inputs.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if not images:
        raise ModelInputError(
            message="Detected empty input to PP-OCRv6 model.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if isinstance(images[0], np.ndarray):
        input_color_format = input_color_format or "bgr"
        return [
            convert_numpy_image_to_bgr(image=image, color_format=input_color_format)
            for image in images
        ]
    if isinstance(images[0], torch.Tensor):
        input_color_format = input_color_format or "rgb"
        result = []
        for image in images:
            result.extend(
                convert_numpy_image_to_bgr(
                    image=numpy_image, color_format=input_color_format
                )
                for numpy_image in torch_images_to_numpy_list(images=image)
            )
        return result
    raise ModelInputError(
        message=f"Detected unsupported PP-OCRv6 model input type: {type(images[0])}",
        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
    )


def torch_images_to_numpy_list(images: torch.Tensor) -> List[np.ndarray]:
    if len(images.shape) == 3:
        images = torch.unsqueeze(images, dim=0)
    if len(images.shape) != 4:
        raise ModelInputError(
            message="PP-OCRv6 models expect torch images in CHW or BCHW format.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    return [image.permute(1, 2, 0).detach().cpu().numpy() for image in images]


def convert_numpy_image_to_bgr(
    image: np.ndarray, color_format: ColorFormat
) -> np.ndarray:
    image = rescale_image_to_uint8(image=image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ModelInputError(
            message="PP-OCRv6 models expect images with 3 color channels.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if color_format == "rgb":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def rescale_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Bring an image to ``uint8`` in ``[0, 255]``, making the scale explicit.

    Float images in ``[0, 1]`` (the common normalized ``torch.Tensor`` case)
    are rescaled by 255; float images with values above ``1.0`` are treated as
    already being on the ``[0, 255]`` scale. Without this, float inputs would
    silently be interpreted as near-black images.
    """
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(image.max()) if image.size else 0.0
        if max_value <= FLOAT_UNIT_RANGE_MAX:
            image = image * 255.0
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)
    raise ModelInputError(
        message=f"PP-OCRv6 models do not support images of dtype {image.dtype}.",
        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
    )


def is_torch_input(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
) -> bool:
    """Whether ``images`` is a ``torch.Tensor`` (or a non-empty list of them).

    Torch inputs are pre-processed on their device without a numpy round-trip;
    numpy inputs keep the cv2 pipeline. See ``normalize_torch_images_to_device``.
    """
    if isinstance(images, torch.Tensor):
        return True
    return isinstance(images, list) and bool(images) and isinstance(images[0], torch.Tensor)


def normalize_torch_images_to_device(
    images: Union[torch.Tensor, List[torch.Tensor]],
    input_color_format: Optional[ColorFormat],
    device: torch.device,
) -> List[torch.Tensor]:
    """Normalize torch image inputs into a list of ``CHW`` BGR ``float32`` tensors.

    The tensors are kept on ``device`` (no host round-trip). Channel order and
    pixel-scale handling match the numpy path (``normalize_input_images``):
    torch images are assumed ``rgb`` and flipped to ``bgr``; float images in
    ``[0, 1]`` are rescaled ×255 while float images with values above ``1.0`` are
    treated as already ``[0, 255]``. Values stay ``float`` (no ``uint8`` round)
    so no precision is lost before the model's own normalization.
    """
    color_format = input_color_format or "rgb"
    if isinstance(images, torch.Tensor):
        tensors = _split_torch_batch(images)
    elif isinstance(images, list):
        tensors = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                raise ModelInputError(
                    message="PP-OCRv6 models expect a homogeneous list of torch.Tensor images.",
                    help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
                )
            tensors.extend(_split_torch_batch(image))
    else:
        raise ModelInputError(
            message="PP-OCRv6 models support np.ndarray, torch.Tensor, or lists of those inputs.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    return [_torch_image_to_chw_bgr(image, color_format, device) for image in tensors]


def _split_torch_batch(images: torch.Tensor) -> List[torch.Tensor]:
    if images.ndim == 3:
        return [images]
    if images.ndim == 4:
        return [image for image in images]
    raise ModelInputError(
        message="PP-OCRv6 models expect torch images in CHW or BCHW format.",
        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
    )


def _torch_image_to_chw_bgr(
    image: torch.Tensor, color_format: ColorFormat, device: torch.device
) -> torch.Tensor:
    image = image.to(device)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ModelInputError(
            message="PP-OCRv6 models expect torch images in CHW format.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if image.shape[0] != 3:
        raise ModelInputError(
            message="PP-OCRv6 models expect images with 3 color channels.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    image = _rescale_torch_image_to_255(image)
    if color_format == "rgb":
        image = image.flip(0)
    return image.contiguous()


def _rescale_torch_image_to_255(image: torch.Tensor) -> torch.Tensor:
    """Bring a torch image to ``float32`` in ``[0, 255]``, making the scale explicit."""
    if image.dtype == torch.uint8:
        return image.to(torch.float32)
    if image.is_floating_point():
        max_value = float(image.max()) if image.numel() else 0.0
        image = image.to(torch.float32)
        if max_value <= FLOAT_UNIT_RANGE_MAX:
            image = image * 255.0
        return image.clamp(0.0, 255.0)
    return image.clamp(0, 255).to(torch.float32)
