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
