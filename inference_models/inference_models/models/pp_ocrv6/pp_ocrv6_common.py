from typing import List, NamedTuple, Optional, Union

import cv2
import numpy as np
import torch

from inference_models.entities import ColorFormat
from inference_models.errors import ModelInputError


class PreProcessingMetadata(NamedTuple):
    source_height: int
    source_width: int


def normalize_input_images(
    images: Union[np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat] = None,
    target_color_format: ColorFormat = "bgr",
) -> List[np.ndarray]:
    """Normalize numpy image inputs into a list of ``uint8`` arrays.

    Numpy-only: ``torch.Tensor`` inputs are pre-processed on-device by the
    models (see ``normalize_torch_images_to_device``) and never reach this path.
    Integer images are read as ``[0, 255]`` (clipped and cast); floating-point
    images are assumed to already be on the ``[0, 255]`` scale (clipped, rounded
    and cast). Channels are emitted in ``target_color_format`` order via a single
    net flip against ``input_color_format`` (numpy inputs default to ``bgr``).
    """
    if isinstance(images, np.ndarray):
        input_color_format = input_color_format or "bgr"
        return [
            convert_numpy_image(
                image=images,
                source_color_format=input_color_format,
                target_color_format=target_color_format,
            )
        ]
    if isinstance(images, torch.Tensor) or (
        isinstance(images, list) and images and isinstance(images[0], torch.Tensor)
    ):
        raise ModelInputError(
            message=(
                "normalize_input_images accepts numpy images only; torch.Tensor "
                "inputs are pre-processed on-device by the PP-OCRv6 models."
            ),
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
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
            convert_numpy_image(
                image=image,
                source_color_format=input_color_format,
                target_color_format=target_color_format,
            )
            for image in images
        ]
    raise ModelInputError(
        message=f"Detected unsupported PP-OCRv6 model input type: {type(images[0])}",
        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
    )


def images_to_numpy_bgr_for_cropping(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat] = None,
) -> List[np.ndarray]:
    """Convert supported image inputs into a list of BGR ``uint8`` arrays.

    Used only by ``PPOCRv6Pipeline`` for its CPU perspective-crop path: torch
    tensors are copied out to numpy here (the models themselves never do this),
    numpy inputs go straight through ``normalize_input_images``.
    """
    if isinstance(images, torch.Tensor):
        input_color_format = input_color_format or "rgb"
        return [
            convert_numpy_image(image=image, source_color_format=input_color_format)
            for image in torch_images_to_numpy_list(images=images)
        ]
    if isinstance(images, list) and images and isinstance(images[0], torch.Tensor):
        input_color_format = input_color_format or "rgb"
        result = []
        for image in images:
            result.extend(
                convert_numpy_image(
                    image=numpy_image, source_color_format=input_color_format
                )
                for numpy_image in torch_images_to_numpy_list(images=image)
            )
        return result
    return normalize_input_images(
        images=images, input_color_format=input_color_format
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


def convert_numpy_image(
    image: np.ndarray,
    source_color_format: ColorFormat,
    target_color_format: ColorFormat = "bgr",
) -> np.ndarray:
    image = rescale_image_to_uint8(image=image)
    if len(image.shape) == 2:
        # Grayscale expands to identical channels, so channel order is irrelevant.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(image)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ModelInputError(
            message="PP-OCRv6 models expect images with 3 color channels.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if source_color_format != target_color_format:
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def rescale_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Bring an image to ``uint8`` in ``[0, 255]``, making the scale explicit.

    Integer images are clipped to ``[0, 255]`` and cast; floating-point images
    are assumed to already be on the ``[0, 255]`` scale and are clipped, rounded
    and cast. This mirrors the sibling ONNX convention (see RF-DETR's
    ``_ensure_hwc_uint8``): callers pass images on the ``[0, 255]`` scale.
    """
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
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
    target_color_format: ColorFormat = "bgr",
) -> List[torch.Tensor]:
    """Normalize torch image inputs into a list of ``CHW`` ``float32`` tensors.

    The tensors are kept on ``device`` (no host round-trip). Integer images are
    read as ``[0, 255]`` and float images are assumed already on the ``[0, 255]``
    scale; values stay ``float`` (no ``uint8`` round) so no precision is lost
    before the model's own normalization. Channels are emitted in
    ``target_color_format`` order via a single net flip against
    ``input_color_format`` (torch inputs default to ``rgb``), so no intermediate
    channel-flipped copy is materialized.
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
    return [
        _torch_image_to_chw(image, color_format, target_color_format, device)
        for image in tensors
    ]


def _split_torch_batch(images: torch.Tensor) -> List[torch.Tensor]:
    if images.ndim == 3:
        return [images]
    if images.ndim == 4:
        return [image for image in images]
    raise ModelInputError(
        message="PP-OCRv6 models expect torch images in CHW or BCHW format.",
        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
    )


def _torch_image_to_chw(
    image: torch.Tensor,
    source_color_format: ColorFormat,
    target_color_format: ColorFormat,
    device: torch.device,
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
    if source_color_format != target_color_format:
        image = image.flip(0)
    return image.contiguous()


def _rescale_torch_image_to_255(image: torch.Tensor) -> torch.Tensor:
    """Bring a torch image to ``float32`` in ``[0, 255]``, making the scale explicit."""
    if image.dtype == torch.uint8:
        return image.to(torch.float32)
    if image.is_floating_point():
        return image.to(torch.float32).clamp(0.0, 255.0)
    return image.clamp(0, 255).to(torch.float32)
