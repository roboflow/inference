from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def resize_opencv_image(
    image: np.ndarray,
    max_height: Optional[int],
    max_width: Optional[int],
) -> Tuple[np.ndarray, Optional[float]]:
    """Downsize an OpenCV image unless a dimension would round to zero.

    Args:
        image: The image to resize.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The resized image and the scaling factor, or the original image and
        None when no resize is needed or a dimension would round to zero.
        In the latter case, the image may exceed the requested bounds to avoid
        distorting geometry represented by a single uniform scaling factor.
    """
    if max_width is None or max_height is None:
        return image, None
    height, width = image.shape[:2]
    scaling_ratio = determine_scaling_aspect_ratio(
        image_height=height,
        image_width=width,
        max_height=max_height,
        max_width=max_width,
    )
    if scaling_ratio is None:
        return image, None

    new_width = round(scaling_ratio * width)
    new_height = round(scaling_ratio * height)
    if scaling_ratio > 0 and (new_width == 0 or new_height == 0):
        return image, None

    # Preserve the original sampling scale when neither dimension rounds to zero.
    resized_image = cv2.resize(
        src=image, dsize=None, fx=scaling_ratio, fy=scaling_ratio
    )
    return resized_image, scaling_ratio


def resize_pillow_image(
    image: Image.Image,
    max_height: Optional[int],
    max_width: Optional[int],
) -> Tuple[Image.Image, Optional[float]]:
    """Downsize a Pillow image unless a dimension would round to zero.

    Args:
        image: The image to resize.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The resized image and the scaling factor, or the original image and
        None when no resize is needed or a dimension would round to zero.
        In the latter case, the image may exceed the requested bounds to avoid
        distorting geometry represented by a single uniform scaling factor.
    """
    if max_width is None or max_height is None:
        return image, None
    width, height = image.size
    scaling_ratio = determine_scaling_aspect_ratio(
        image_height=height,
        image_width=width,
        max_height=max_height,
        max_width=max_width,
    )
    if scaling_ratio is None:
        return image, None

    new_width = round(scaling_ratio * width)
    new_height = round(scaling_ratio * height)
    if scaling_ratio > 0 and (new_width == 0 or new_height == 0):
        return image, None

    resized_image = image.resize(size=(new_width, new_height))
    return resized_image, scaling_ratio


def determine_scaling_aspect_ratio(
    image_height: int,
    image_width: int,
    max_height: int,
    max_width: int,
) -> Optional[float]:
    """Determine the scaling aspect ratio.

    Args:
        image_height: The height of the image.
        image_width: The width of the image.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The scaling aspect ratio.
    """
    height_scaling_ratio = max_height / image_height
    width_scaling_ratio = max_width / image_width
    min_scaling_ratio = min(height_scaling_ratio, width_scaling_ratio)
    return min_scaling_ratio if min_scaling_ratio < 1.0 else None
