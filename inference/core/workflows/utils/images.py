"""Workflows-local copy of the numpy image-resizing helpers.

`_resize_image_keeping_aspect_ratio` is a copy of
`inference.core.utils.preprocess.resize_image_keeping_aspect_ratio` with only the
`isinstance(image, np.ndarray)` branches kept. The torch branch (guarded by
`USE_PYTORCH_FOR_PREPROCESSING` on the server) is deliberately not carried over:
every workflows caller passes a numpy image.
"""

from typing import Tuple

import cv2
import numpy as np


def downscale_image_keeping_aspect_ratio(
    image: np.ndarray,
    desired_size: Tuple[int, int],
) -> np.ndarray:
    if image.shape[0] <= desired_size[1] and image.shape[1] <= desired_size[0]:
        return image
    return _resize_image_keeping_aspect_ratio(image=image, desired_size=desired_size)


def _resize_image_keeping_aspect_ratio(
    image: np.ndarray,
    desired_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize reserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    """
    if isinstance(image, np.ndarray):
        img_ratio = image.shape[1] / image.shape[0]
    else:
        raise ValueError(
            f"Received an image of unknown type, {type(image)}; "
            "This is most likely a bug. Contact Roboflow team through github issues "
            "(https://github.com/roboflow/inference/issues) providing full context of the problem"
        )
    desired_ratio = desired_size[0] / desired_size[1]

    # Determine the new dimensions
    if img_ratio >= desired_ratio:
        # Resize by width
        new_width = desired_size[0]
        new_height = int(desired_size[0] / img_ratio)
    else:
        # Resize by height
        new_height = desired_size[1]
        new_width = int(desired_size[1] * img_ratio)

    # Resize the image to new dimensions
    if isinstance(image, np.ndarray):
        return cv2.resize(image, (new_width, new_height))
    else:
        raise ValueError(
            f"Received an image of unknown type, {type(image)}; "
            "This is most likely a bug. Contact Roboflow team through github issues "
            "(https://github.com/roboflow/inference/issues) providing full context of the problem"
        )
