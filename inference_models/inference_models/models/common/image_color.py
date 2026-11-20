from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from inference_models.entities import ColorFormat

ImageInput = Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]


def images_to_rgb(
    images: ImageInput,
    input_color_format: Optional[ColorFormat],
) -> ImageInput:
    """Flip BGR inputs to the RGB the processor expects.

    An explicit "bgr" flips every input, tensors included (rank 3 and batched
    rank 4, CHW/BCHW and HWC/BHWC), like the sibling VLMs. When no format is
    declared, numpy inputs are assumed BGR (the cv2 default) and flipped;
    tensors are passed through unchanged.
    """
    if input_color_format == "rgb":
        return images
    flip_tensors = input_color_format == "bgr"
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
        return [_flip_to_rgb(image, flip_tensors=flip_tensors) for image in images]
    return _flip_to_rgb(images, flip_tensors=flip_tensors)


def _flip_to_rgb(image, flip_tensors: bool):
    if isinstance(image, np.ndarray):
        return image[..., ::-1].copy()
    if flip_tensors and isinstance(image, torch.Tensor) and image.ndim in (3, 4):
        channel_dim = image.ndim - 3  # 0 for CHW, 1 for BCHW
        if image.shape[channel_dim] == 3:  # CHW / BCHW
            return image.index_select(
                channel_dim, torch.tensor([2, 1, 0], device=image.device)
            )
        if image.shape[-1] == 3:  # HWC / BHWC
            return image[..., [2, 1, 0]]
    return image
