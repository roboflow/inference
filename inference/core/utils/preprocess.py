from enum import Enum
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.exposure import rescale_intensity

from inference.core.env import (
    DISABLE_PREPROC_CONTRAST,
    DISABLE_PREPROC_GRAYSCALE,
    DISABLE_PREPROC_STATIC_CROP,
    USE_PYTORCH_FOR_PREPROCESSING,
)

if USE_PYTORCH_FOR_PREPROCESSING:
    import torch


from inference.core.exceptions import PreProcessingError
from inference.core.utils.onnx import ImageMetaType

STATIC_CROP_KEY = "static-crop"
CONTRAST_KEY = "contrast"
GRAYSCALE_KEY = "grayscale"
ENABLED_KEY = "enabled"
TYPE_KEY = "type"


class ContrastAdjustmentType(Enum):
    CONTRAST_STRETCHING = "Contrast Stretching"
    HISTOGRAM_EQUALISATION = "Histogram Equalization"
    ADAPTIVE_EQUALISATION = "Adaptive Equalization"


def prepare(
    image: np.ndarray,
    preproc,
    disable_preproc_contrast: bool = False,
    disable_preproc_grayscale: bool = False,
    disable_preproc_static_crop: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Prepares an image by applying a series of preprocessing steps defined in the `preproc` dictionary.

    Args:
        image (PIL.Image.Image): The input PIL image object.
        preproc (dict): Dictionary containing preprocessing steps. Example:
            {
                "resize": {"enabled": true, "width": 416, "height": 416, "format": "Stretch to"},
                "static-crop": {"y_min": 25, "x_max": 75, "y_max": 75, "enabled": true, "x_min": 25},
                "auto-orient": {"enabled": true},
                "grayscale": {"enabled": true},
                "contrast": {"enabled": true, "type": "Adaptive Equalization"}
            }
        disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
        disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
        disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

    Returns:
        PIL.Image.Image: The preprocessed image object.
        tuple: The dimensions of the image.

    Note:
        The function uses global flags like `DISABLE_PREPROC_AUTO_ORIENT`, `DISABLE_PREPROC_STATIC_CROP`, etc.
        to conditionally enable or disable certain preprocessing steps.
    """
    try:
        if isinstance(image, np.ndarray):
            h, w = image.shape[0:2]
        elif USE_PYTORCH_FOR_PREPROCESSING:
            h, w = image.shape[-2:]
        else:
            raise ValueError(
                f"Received an image of unknown type, {type(image)}; "
                "This is most likely a bug. Contact Roboflow team through github issues "
                "(https://github.com/roboflow/inference/issues) providing full context of the problem"
            )

        img_dims = (h, w)
        if static_crop_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_static_crop=disable_preproc_static_crop,
        ):
            image = take_static_crop(
                image=image, crop_parameters=preproc[STATIC_CROP_KEY]
            )
        if contrast_adjustments_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_contrast=disable_preproc_contrast,
        ):
            adjustment_type = ContrastAdjustmentType(preproc[CONTRAST_KEY][TYPE_KEY])
            image = apply_contrast_adjustment(
                image=image, adjustment_type=adjustment_type
            )
        if grayscale_conversion_should_be_applied(
            preprocessing_config=preproc,
            disable_preproc_grayscale=disable_preproc_grayscale,
        ):
            image = apply_grayscale_conversion(image=image)
        return image, img_dims
    except KeyError as error:
        raise PreProcessingError(
            f"Pre-processing of image failed due to misconfiguration. Missing key: {error}."
        ) from error


def static_crop_should_be_applied(
    preprocessing_config: dict,
    disable_preproc_static_crop: bool,
) -> bool:
    return (
        STATIC_CROP_KEY in preprocessing_config.keys()
        and not DISABLE_PREPROC_STATIC_CROP
        and not disable_preproc_static_crop
        and preprocessing_config[STATIC_CROP_KEY][ENABLED_KEY]
    )


def take_static_crop(image: np.ndarray, crop_parameters: Dict[str, int]) -> np.ndarray:
    height, width = image.shape[0:2]
    x_min = int(crop_parameters["x_min"] / 100 * width)
    y_min = int(crop_parameters["y_min"] / 100 * height)
    x_max = int(crop_parameters["x_max"] / 100 * width)
    y_max = int(crop_parameters["y_max"] / 100 * height)
    return image[y_min:y_max, x_min:x_max, :]


def contrast_adjustments_should_be_applied(
    preprocessing_config: dict,
    disable_preproc_contrast: bool,
) -> bool:
    return (
        CONTRAST_KEY in preprocessing_config.keys()
        and not DISABLE_PREPROC_CONTRAST
        and not disable_preproc_contrast
        and preprocessing_config[CONTRAST_KEY][ENABLED_KEY]
    )


def apply_contrast_adjustment(
    image: np.ndarray,
    adjustment_type: ContrastAdjustmentType,
) -> np.ndarray:
    adjustment = CONTRAST_ADJUSTMENTS_METHODS[adjustment_type]
    return adjustment(image)


def apply_contrast_stretching(image: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(image, (2, 98))
    return rescale_intensity(image, in_range=(p2, p98))  # type: ignore


def apply_histogram_equalisation(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def apply_adaptive_equalisation(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.03, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


CONTRAST_ADJUSTMENTS_METHODS = {
    ContrastAdjustmentType.CONTRAST_STRETCHING: apply_contrast_stretching,
    ContrastAdjustmentType.HISTOGRAM_EQUALISATION: apply_histogram_equalisation,
    ContrastAdjustmentType.ADAPTIVE_EQUALISATION: apply_adaptive_equalisation,
}


def grayscale_conversion_should_be_applied(
    preprocessing_config: dict,
    disable_preproc_grayscale: bool,
) -> bool:
    return (
        GRAYSCALE_KEY in preprocessing_config.keys()
        and not DISABLE_PREPROC_GRAYSCALE
        and not disable_preproc_grayscale
        and preprocessing_config[GRAYSCALE_KEY][ENABLED_KEY]
    )


def apply_grayscale_conversion(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def letterbox_image(
    image: ImageMetaType,
    desired_size: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
) -> ImageMetaType:
    """
    Resize and pad image to fit the desired size, preserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    - color: tuple (B, G, R) representing the color to pad with.

    Returns:
    - letterboxed image.
    """
    resized_img = resize_image_keeping_aspect_ratio(
        image=image,
        desired_size=desired_size,
    )
    new_height, new_width = (
        resized_img.shape[:2]
        if isinstance(resized_img, np.ndarray)
        else resized_img.shape[-2:]
    )
    top_padding = (desired_size[1] - new_height) // 2
    bottom_padding = desired_size[1] - new_height - top_padding
    left_padding = (desired_size[0] - new_width) // 2
    right_padding = desired_size[0] - new_width - left_padding
    if isinstance(resized_img, np.ndarray):
        return cv2.copyMakeBorder(
            resized_img,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=color,
        )
    elif USE_PYTORCH_FOR_PREPROCESSING:
        return torch.nn.functional.pad(
            resized_img,
            (left_padding, right_padding, top_padding, bottom_padding),
            "constant",
            color[0],
        )
    else:
        raise ValueError(
            f"Received an image of unknown type, {type(resized_img)}; "
            "This is most likely a bug. Contact Roboflow team through github issues "
            "(https://github.com/roboflow/inference/issues) providing full context of the problem"
        )


def downscale_image_keeping_aspect_ratio(
    image: ImageMetaType,
    desired_size: Tuple[int, int],
) -> ImageMetaType:
    if image.shape[0] <= desired_size[1] and image.shape[1] <= desired_size[0]:
        return image
    return resize_image_keeping_aspect_ratio(image=image, desired_size=desired_size)


def resize_image_keeping_aspect_ratio(
    image: ImageMetaType,
    desired_size: Tuple[int, int],
) -> ImageMetaType:
    """
    Resize reserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    """
    if isinstance(image, np.ndarray):
        img_ratio = image.shape[1] / image.shape[0]
    elif USE_PYTORCH_FOR_PREPROCESSING:
        img_ratio = image.shape[-1] / image.shape[-2]
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
    elif USE_PYTORCH_FOR_PREPROCESSING:
        return torch.nn.functional.interpolate(
            image, size=(new_height, new_width), mode="bilinear"
        )
    else:
        raise ValueError(
            f"Received an image of unknown type, {type(image)}; "
            "This is most likely a bug. Contact Roboflow team through github issues "
            "(https://github.com/roboflow/inference/issues) providing full context of the problem"
        )
