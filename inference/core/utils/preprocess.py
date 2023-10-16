import functools
from enum import Enum
from typing import Dict, Tuple

import cv2
import numpy as np

from inference.core.env import (
    DISABLE_PREPROC_CONTRAST,
    DISABLE_PREPROC_GRAYSCALE,
    DISABLE_PREPROC_STATIC_CROP,
)
from inference.core.exceptions import PreProcessingError

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
    disable_preproc_auto_orient: bool = False,
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
        disable_preproc_auto_orient (bool, optional): NOT USED AND DEPRECATED.
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
    h, w = image.shape[0:2]
    img_dims = (h, w)
    if static_crop_should_be_applied(
        preprocessing_config=preproc,
        disable_preproc_static_crop=disable_preproc_static_crop,
    ):
        image = take_static_crop(image=image, crop_parameters=preproc[STATIC_CROP_KEY])
    if contrast_adjustments_should_be_applied(
        preprocessing_config=preproc,
        disable_preproc_contrast=disable_preproc_contrast,
    ):
        adjustment_type = ContrastAdjustmentType(preproc[CONTRAST_KEY][TYPE_KEY])
        image = apply_contrast_adjustment(image=image, adjustment_type=adjustment_type)
    if grayscale_conversion_should_be_applied(
        preprocessing_config=preproc,
        disable_preproc_grayscale=disable_preproc_grayscale,
    ):
        image = apply_grayscale_conversion(image=image)
    return image, img_dims


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
    return rescale_intensity(image, in_range=(p2, p98))


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


### The following functions are included from scikit-image (https://github.com/scikit-image/scikit-image) ###
### View copyright and license information here: https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt ###


def rescale_intensity(
    image: np.ndarray, in_range="image", out_range="dtype"
) -> np.ndarray:
    """
    Return image after stretching or shrinking its intensity levels.

    Args:
        image (array): Image array.
        in_range (str or 2-tuple, optional): Min and max intensity values of input image. Defaults to 'image'.
        out_range (str or 2-tuple, optional): Min and max intensity values of output image. Defaults to 'dtype'.

    Returns:
        array: The rescaled image.

    Note:
        The possible values for `in_range` and `out_range` are:
            - 'image': Use image min/max as the intensity range.
            - 'dtype': Use min/max of the image's dtype as the intensity range.
            - dtype-name: Use intensity range based on desired `dtype`. Must be valid key in `DTYPE_RANGE`.
            - 2-tuple: Use `range_values` as explicit min/max intensities.
    """

    dtype = image.dtype.type

    imin, imax = intensity_range(image, in_range)
    omin, omax = intensity_range(image, out_range, clip_negative=(imin >= 0))

    image = np.clip(image, imin, imax)

    if imin != imax:
        image = (image - imin) / float(imax - imin)
    return np.asarray(image * (omax - omin) + omin, dtype=dtype)


def intensity_range(image: np.ndarray, range_values="image", clip_negative=False):
    """
    Return image intensity range (min, max) based on the desired value type.

    Args:
        image (array): Input image.
        range_values (str or 2-tuple, optional): The image intensity range configuration. Defaults to 'image'.
            - 'image': Return image min/max as the range.
            - 'dtype': Return min/max of the image's dtype as the range.
            - dtype-name: Return intensity range based on desired `dtype`. Must be a valid key in `DTYPE_RANGE`.
            - 2-tuple: Return `range_values` as min/max intensities.
        clip_negative (bool, optional): If True, clip the negative range (i.e., return 0 for min intensity)
                                        even if the image dtype allows negative values. Defaults to False.

    Returns:
        tuple: The minimum and maximum intensity of the image.
    """
    if range_values == "dtype":
        range_values = image.dtype.type

    if range_values == "image":
        i_min = np.min(image)
        i_max = np.max(image)
    elif range_values in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[range_values]
        if clip_negative:
            i_min = 0
    else:
        i_min, i_max = range_values
    return i_min, i_max
