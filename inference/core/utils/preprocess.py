import functools
from typing import Tuple

import cv2
import numpy as np

from inference.core.env import (
    DISABLE_PREPROC_AUTO_ORIENT,
    DISABLE_PREPROC_CONTRAST,
    DISABLE_PREPROC_GRAYSCALE,
    DISABLE_PREPROC_STATIC_CROP,
)

# def auto_orient(image):
#     """
#     Automatically adjusts the orientation of an image based on its EXIF data.
#     The orientation is corrected by rotating or flipping the image as needed.

#     Args:
#         image (PIL.Image.Image): The input image object that may contain EXIF orientation data.

#     Returns:
#         PIL.Image.Image: The image object with corrected orientation. If no EXIF orientation data is found,
#                          the original image is returned unmodified.

#     Raises:
#         None: Any exceptions raised during processing are caught and ignored, so the original image is returned.

#     Example:
#         corrected_image = auto_orient(original_image)
#     """
#     info = image.info
#     if "exif" in info:
#         exif_dict = piexif.load(info["exif"])

#         try:
#             if piexif.ImageIFD.Orientation in exif_dict["0th"]:
#                 orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
#                 # exif_bytes = piexif.dump(exif_dict)

#                 if orientation == 2:
#                     image = image.transpose(Image.FLIP_LEFT_RIGHT)
#                 elif orientation == 3:
#                     image = image.rotate(180)
#                 elif orientation == 4:
#                     image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
#                 elif orientation == 5:
#                     image = image.rotate(-90, expand=True).transpose(
#                         Image.FLIP_LEFT_RIGHT
#                     )
#                 elif orientation == 6:
#                     image = image.rotate(-90, expand=True)
#                 elif orientation == 7:
#                     image = image.rotate(90, expand=True).transpose(
#                         Image.FLIP_LEFT_RIGHT
#                     )
#                 elif orientation == 8:
#                     image = image.rotate(90, expand=True)
#         except:
#             pass
#     return image


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
        disable_preproc_auto_orient (bool, optional): If true, the auto-orient preprocessing step is disabled for this call. Default is False.
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
    if (
        "static-crop" in preproc.keys()
        and not DISABLE_PREPROC_STATIC_CROP
        and not disable_preproc_static_crop
    ):
        if preproc["static-crop"]["enabled"] == True:
            x_min = int(preproc["static-crop"]["x_min"] / 100 * w)
            y_min = int(preproc["static-crop"]["y_min"] / 100 * h)
            x_max = int(preproc["static-crop"]["x_max"] / 100 * w)
            y_max = int(preproc["static-crop"]["y_max"] / 100 * h)

            image = image[y_min:y_max, x_min:x_max, :]
    if (
        "contrast" in preproc.keys()
        and not DISABLE_PREPROC_CONTRAST
        and not disable_preproc_contrast
    ):
        if preproc["contrast"]["enabled"]:
            how = preproc["contrast"]["type"]

            if how == "Contrast Stretching":
                p2, p98 = np.percentile(image, (2, 98))
                image = rescale_intensity(image, in_range=(p2, p98))

            elif how == "Histogram Equalization":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.equalizeHist(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            elif how == "Adaptive Equalization":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=0.03, tileGridSize=(8, 8))
                image = clahe.apply(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if (
        "grayscale" in preproc.keys()
        and not DISABLE_PREPROC_GRAYSCALE
        and not disable_preproc_grayscale
    ):
        if preproc["grayscale"]["enabled"]:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, img_dims


### The following functions are included from scikit-image (https://github.com/scikit-image/scikit-image) ###
### View copyright and license information here: https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt ###


def intensity_range(image, range_values="image", clip_negative=False):
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


def rescale_intensity(image, in_range="image", out_range="dtype"):
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


def histogram(image, nbins=256, source_range="image", normalize=False):
    """
    Return histogram of image.

    Args:
        image (array): Input image.
        nbins (int, optional): Number of bins used to calculate histogram. This value is ignored for integer arrays. Defaults to 256.
        source_range (string, optional): 'image' (default) determines the range from the input image. 'dtype' determines the range from the expected range of the images of that data type.
        normalize (bool, optional): If True, normalize the histogram by the sum of its values. Defaults to False.

    Returns:
        hist (array): The values of the histogram.
        bin_centers (array): The values at the center of the bins.
    """
    sh = image.shape

    image = image.flatten()
    if source_range == "image":
        hist_range = None
    else:
        ValueError("Wrong value for the `source_range` argument")
    hist, bin_edges = np.histogram(image, bins=nbins, range=hist_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers


def cumulative_distribution(image, nbins=256):
    """
    Return cumulative distribution function (cdf) for the given image.

    Args:
        image (array): Image array.
        nbins (int, optional): Number of bins for image histogram. Defaults to 256.

    Returns:
        img_cdf (array): Values of cumulative distribution function.
        bin_centers (array): Centers of bins.
    """
    hist, bin_centers = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    return img_cdf, bin_centers


def equalize_hist(image, nbins=256, mask=None):
    """
    Equalizes the histogram of an image.

    Args:
        image (array): Image array.
        nbins (int, optional): Number of bins for image histogram. Defaults to 256.
        mask (array, optional): A binary mask. If provided, only the pixels selected by the mask are included in the analysis. Maybe a 1-D sequence of length equal to the number of channels, or a single scalar (in which case all channels are identically masked). Defaults to None.

    Returns:
        array: The equalized image.
    """
    cdf, bin_centers = cumulative_distribution(image, nbins)
    out = np.interp(image.flat, bin_centers, cdf)
    return out.reshape(image.shape)


def _prepare_colorarray(arr, force_copy=False):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    # Drop alpha channel if present
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(
            "Input array must have a shape == (..., 3)), " f"got {arr.shape}"
        )

    return img_as_float(arr, force_copy=force_copy)


def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale.

    Args:
        rgb (array): The input RGB image.

    Returns:
        array: The grayscale image.

    Note:
        The conversion uses the following formula: 0.2125 * R + 0.7154 * G + 0.0721 * B
    """
    rgb = _prepare_colorarray(rgb)
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=rgb.dtype)
    return rgb @ coeffs


###Adaptive Equalization###
_integer_types = (
    np.byte,
    np.ubyte,  # 8 bits
    np.short,
    np.ushort,  # 16 bits
    np.intc,
    np.uintc,  # 16 or 32 or 64 bits
    np.int_,
    np.uint,  # 32 or 64 bits
    np.longlong,
    np.ulonglong,
)  # 64 bits
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max) for t in _integer_types}
dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}
dtype_range.update(_integer_ranges)

_supported_types = list(dtype_range.keys())

DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update((d.__name__, limits) for d, limits in dtype_range.items())
DTYPE_RANGE.update(
    {
        "uint10": (0, 2**10 - 1),
        "uint12": (0, 2**12 - 1),
        "uint14": (0, 2**14 - 1),
        "bool": dtype_range[np.bool_],
        "float": dtype_range[np.float64],
    }
)


def _dtype_itemsize(itemsize, *dtypes):
    """Return first of `dtypes` with itemsize greater than `itemsize`.

    Args:
        itemsize (int): The data type object element size.
        *dtypes: Any Object accepted by `np.dtype` to be converted to a data type object.

    Returns:
        dtype: data type object, First of `dtypes` with itemsize greater than `itemsize`.
    """
    return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)


def convert(image, dtype, force_copy=False, uniform=False):
    """Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).

    Args:
        image (ndarray): Input image.
        dtype (dtype): Target data-type.
        force_copy (bool, optional): Force a copy of the data, irrespective of its current dtype.
        uniform (bool, optional): Uniformly quantize the floating point range to the integer range.

    Returns:
        ndarray: The converted image.

    References:
        .. [1] DirectX data conversion rules.
           https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
        .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
        .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
        .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.
    """
    image = np.asarray(image)
    dtypeobj_in = image.dtype
    dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    # Below, we do an `issubdtype` check.  Its purpose is to find out
    # whether we can get away without doing any image conversion.  This happens
    # when:
    #
    # - the output and input dtypes are the same or
    # - when the output is specified as a type, and the input dtype
    #   is a subclass of that type (e.g. `np.floating` will allow
    #   `float32` and `float64` arrays through)

    if np.issubdtype(dtype_in, np.obj2sctype(dtype)):
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError(
            "Can not convert from {} to {}.".format(dtypeobj_in, dtypeobj_out)
        )

    if kind_in in "ui":
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in "ui":
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    # any -> binary
    if kind_out == "b":
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == "b":
        result = image.astype(dtype_out)
        if kind_out != "f":
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == "f":
        if kind_out == "f":
            # float -> float
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(
            itemsize_out, dtype_in, np.float32, np.float64
        )

        if not uniform:
            if kind_out == "u":
                image_out = np.multiply(image, imax_out, dtype=computation_type)
            else:
                image_out = np.multiply(
                    image, (imax_out - imin_out) / 2, dtype=computation_type
                )
                image_out -= 1.0 / 2.0
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == "u":
            image_out = np.multiply(image, imax_out + 1, dtype=computation_type)
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = np.multiply(
                image, (imax_out - imin_out + 1.0) / 2.0, dtype=computation_type
            )
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == "f":
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(
            itemsize_in, dtype_out, np.float32, np.float64
        )

        if kind_in == "u":
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)

        return np.asarray(image, dtype_out)


def img_as_float(image, force_copy=False):
    """Convert an image to floating point format.

    Args:
        image (ndarray): Input image.
        force_copy (bool, optional): Force a copy of the data, irrespective of its current dtype.

    Returns:
        ndarray of float: Output image.

    Notes:
        The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
        converting from unsigned or signed datatypes, respectively.
        If the input image has a float type, intensity values are not modified
        and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].
    """
    return convert(image, np.floating, force_copy)


def img_as_uint(image, force_copy=False):
    """Convert an image to 16-bit unsigned integer format.

    Args:
        image (ndarray): Input image.
        force_copy (bool, optional): Force a copy of the data, irrespective of its current dtype.

    Returns:
        ndarray of uint16: Output image.

    Notes:
        Negative input values will be clipped.
        Positive values are scaled between 0 and 65535.
    """
    return convert(image, np.uint16, force_copy)


def img_as_int(image, force_copy=False):
    """Convert an image to 16-bit signed integer format.

    Args:
        image (ndarray): Input image.
        force_copy (bool, optional): Force a copy of the data, irrespective of its current dtype.

    Returns:
        ndarray of uint16: Output image.

    Notes:
        The values are scaled between -32768 and 32767.
        If the input data-type is positive-only (e.g., uint8), then
        the output image will still only have positive values.
    """
    return convert(image, np.int16, force_copy)


def rgb2hsv(rgb):
    """RGB to HSV color space conversion.

    Args:
        rgb (array_like): The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.

    Returns:
        ndarray: The image in HSV format, in a 3-D array of shape ``(.., .., 3)``.

    Raises:
        ValueError: If `rgb` is not a 3-D array of shape ``(.., .., 3)``.

    Notes:
        Conversion between RGB and HSV color spaces results in some loss of
        precision, due to integer arithmetic and rounding [1]_.

    References:
        .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV
    """
    arr = _prepare_colorarray(rgb)
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = arr.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid="ignore")
    out_s = delta / out_v
    out_s[delta == 0.0] = 0.0

    # -- H channel
    # red is max
    idx = arr[:, :, 0] == out_v
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = arr[:, :, 1] == out_v
    out[idx, 0] = 2.0 + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = arr[:, :, 2] == out_v
    out[idx, 0] = 4.0 + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[:, :, 0] / 6.0) % 1.0
    out_h[delta == 0.0] = 0.0

    np.seterr(**old_settings)

    # -- output
    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v

    # remove NaN
    out[np.isnan(out)] = 0

    return out


def hsv2rgb(hsv):
    """HSV to RGB color space conversion.

    Args:
        hsv (array_like): The image in HSV format, in a 3-D array of shape ``(.., .., 3)``.

    Returns:
        ndarray: The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.

    Raises:
        ValueError: If `hsv` is not a 3-D array of shape ``(.., .., 3)``.

    Notes:
        Conversion between RGB and HSV color spaces results in some loss of
        precision, due to integer arithmetic and rounding [1]_.

    References:
        .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV
    """
    arr = _prepare_colorarray(hsv)

    hi = np.floor(arr[:, :, 0] * 6)
    f = arr[:, :, 0] * 6 - hi
    p = arr[:, :, 2] * (1 - arr[:, :, 1])
    q = arr[:, :, 2] * (1 - f * arr[:, :, 1])
    t = arr[:, :, 2] * (1 - (1 - f) * arr[:, :, 1])
    v = arr[:, :, 2]

    hi = np.dstack([hi, hi, hi]).astype(np.uint8) % 6
    out = np.choose(
        hi,
        [
            np.dstack((v, t, p)),
            np.dstack((q, v, p)),
            np.dstack((p, v, t)),
            np.dstack((p, q, v)),
            np.dstack((t, p, v)),
            np.dstack((v, p, q)),
        ],
    )

    return out


def is_rgb_like(image):
    """Return True if the image *looks* like it's RGB.

    This function should not be public because it is only intended to be used
    for functions that don't accept volumes as input, since checking an image's
    shape is fragile.

    Args:
        image (ndarray): The input image.

    Returns:
        bool: True if the image appears to be RGB, False otherwise.
    """
    return (image.ndim == 3) and (image.shape[2] in (3, 4))


def adapt_rgb(apply_to_rgb):
    """Return decorator that adapts to RGB images to a gray-scale filter.

    This function is only intended to be used for functions that don't accept
    volumes as input, since checking an image's shape is fragile.

    Args:
        apply_to_rgb (function): Function that returns a filtered image from an image-filter and RGB image. This will only be called if the image is RGB-like.

    Returns:
        function: A decorator function.
    """

    def decorator(image_filter):
        @functools.wraps(image_filter)
        def image_filter_adapted(image, *args, **kwargs):
            if is_rgb_like(image):
                return apply_to_rgb(image_filter, image, *args, **kwargs)
            else:
                return image_filter(image, *args, **kwargs)

        return image_filter_adapted

    return decorator


def hsv_value(image_filter, image, *args, **kwargs):
    """Return color image by applying `image_filter` on HSV-value of `image`.

    Note that this function is intended for use with `adapt_rgb`.

    Args:
        image_filter (function): Function that filters a gray-scale image.
        image (array): Input image. Note that RGBA images are treated as RGB.

    Returns:
        ndarray: The filtered image.
    """
    # Slice the first three channels so that we remove any alpha channels.
    hsv = rgb2hsv(image[:, :, :3])
    value = hsv[:, :, 2].copy()
    value = image_filter(value, *args, **kwargs)
    hsv[:, :, 2] = convert(value, hsv.dtype)
    return hsv2rgb(hsv)


def each_channel(image_filter, image, *args, **kwargs):
    """Return color image by applying `image_filter` on channels of `image`.

    Note that this function is intended for use with `adapt_rgb`.

    Args:
        image_filter (function): Function that filters a gray-scale image.
        image (array): Input image.

    Returns:
        ndarray: The filtered image.
    """
    c_new = [image_filter(c, *args, **kwargs) for c in np.moveaxis(image, -1, 0)]
    return np.moveaxis(np.array(c_new), 0, -1)


NR_OF_GREY = 2**14  # number of grayscale levels to use in CLAHE algorithm


@adapt_rgb(hsv_value)
def equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image (ndarray): Input image of shape (M, N[, C]).
        kernel_size (int or list-like, optional): Defines the shape of contextual regions used in the algorithm. Defaults to 1/8 of image height by 1/8 of its width.
        clip_limit (float, optional): Clipping limit, normalized between 0 and 1. Defaults to 0.01.
        nbins (int, optional): Number of gray bins for histogram. Defaults to 256.

    Returns:
        ndarray: Equalized image.

    See Also:
        equalize_hist, rescale_intensity

    Notes:
        For color images, the image is converted to HSV color space, the CLAHE algorithm is run on the V (Value) channel, and the image is converted back to RGB space.
        For RGBA images, the original alpha channel is removed.

    References:
        .. [1] http://tog.acm.org/resources/GraphicsGems/
        .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    image = img_as_uint(image)
    image = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))

    if kernel_size is None:
        kernel_size = (image.shape[0] // 8, image.shape[1] // 8)
    elif len(kernel_size) != image.ndim:
        ValueError("Incorrect value of `kernel_size`: {}".format(kernel_size))

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit * nbins, nbins)
    image = img_as_float(image)

    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins=128):
    """Contrast Limited Adaptive Histogram Equalization.

    Args:
        image (ndarray): Input image of shape (M, N).
        kernel_size (2-tuple of int): Defines the shape of contextual regions used in the algorithm.
        clip_limit (float): Normalized clipping limit.
        nbins (int, optional): Number of gray bins for histogram. Defaults to 128.

    Returns:
        ndarray: Equalized image.
    """

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    nr = int(np.ceil(image.shape[0] / kernel_size[0]))
    nc = int(np.ceil(image.shape[1] / kernel_size[1]))

    row_step = int(np.floor(image.shape[0] / nr))
    col_step = int(np.floor(image.shape[1] / nc))

    bin_size = 1 + NR_OF_GREY // nbins
    lut = np.arange(NR_OF_GREY)
    lut //= bin_size

    map_array = np.zeros((nr, nc, nbins), dtype=int)

    # Calculate greylevel mappings for each contextual region
    for r in range(nr):
        for c in range(nc):
            sub_img = image[
                r * row_step : (r + 1) * row_step, c * col_step : (c + 1) * col_step
            ]

            if clip_limit > 0.0:  # Calculate actual cliplimit
                clim = int(clip_limit * sub_img.size / nbins)
                if clim < 1:
                    clim = 1
            else:
                clim = NR_OF_GREY  # Large value, do not clip (AHE)

            hist = lut[sub_img.ravel()]
            hist = np.bincount(hist)
            hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
            hist = clip_histogram(hist, clim)
            hist = map_histogram(hist, 0, NR_OF_GREY - 1, sub_img.size)
            map_array[r, c] = hist

    # Interpolate greylevel mappings to get CLAHE image
    rstart = 0
    for r in range(nr + 1):
        cstart = 0
        if r == 0:  # special case: top row
            r_offset = row_step / 2.0
            rU = 0
            rB = 0
        elif r == nr:  # special case: bottom row
            r_offset = row_step / 2.0
            rU = nr - 1
            rB = rU
        else:  # default values
            r_offset = row_step
            rU = r - 1
            rB = rB + 1

        for c in range(nc + 1):
            if c == 0:  # special case: left column
                c_offset = col_step / 2.0
                cL = 0
                cR = 0
            elif c == nc:  # special case: right column
                c_offset = col_step / 2.0
                cL = nc - 1
                cR = cL
            else:  # default values
                c_offset = col_step
                cL = c - 1
                cR = cL + 1

            mapLU = map_array[rU, cL]
            mapRU = map_array[rU, cR]
            mapLB = map_array[rB, cL]
            mapRB = map_array[rB, cR]

            cslice = np.arange(cstart, cstart + c_offset)
            rslice = np.arange(rstart, rstart + r_offset)

            interpolate(image, cslice, rslice, mapLU, mapRU, mapLB, mapRB, lut)

            cstart += c_offset  # set pointer on next matrix */

        rstart += r_offset

    return image


def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.

    Args:
        hist (ndarray): Histogram array.
        clip_limit (int): Maximum allowed bin count.

    Returns:
        ndarray: Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = int(n_excess / hist.size)  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    hist[excess_mask] = clip_limit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = (hist >= upper) & (hist < clip_limit)
    mid = hist[mid_mask]
    n_excess -= mid.size * clip_limit - mid.sum()
    hist[mid_mask] = clip_limit

    prev_n_excess = n_excess

    while n_excess > 0:  # Redistribute remaining excess
        index = 0
        while n_excess > 0 and index < hist.size:
            under_mask = hist < 0
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under_mask[indices] = True
            under_mask = (under_mask) & (hist < clip_limit)
            hist[under_mask] += 1
            n_excess -= under_mask.sum()
            index += 1
        # bail if we have not distributed any excess
        if prev_n_excess == n_excess:
            break
        prev_n_excess = n_excess

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).

    Args:
        hist (ndarray): Clipped histogram.
        min_val (int): Minimum value for mapping.
        max_val (int): Maximum value for mapping.
        n_pixels (int): Number of pixels in the region.

    Returns:
        ndarray: Mapped intensity LUT.
    """
    out = np.cumsum(hist).astype(float)
    scale = ((float)(max_val - min_val)) / n_pixels
    out *= scale
    out += min_val
    out[out > max_val] = max_val
    return out.astype(int)


def interpolate(image, xslice, yslice, mapLU, mapRU, mapLB, mapRB, lut):
    """Find the new grayscale level for a region using bilinear interpolation.

    Args:
        image (ndarray): Full image.
        xslice, yslice (array-like): Indices of the region.
        mapLU, mapRU, mapLB, mapRB (ndarray): Mappings of greylevels from histograms.
        lut (ndarray): Maps grayscale levels in image to histogram levels.

    Returns:
        ndarray: Original image with the subregion replaced.
    """
    norm = xslice.size * yslice.size  # Normalization factor
    # interpolation weight matrices
    x_coef, y_coef = np.meshgrid(np.arange(xslice.size), np.arange(yslice.size))
    x_inv_coef, y_inv_coef = x_coef[:, ::-1] + 1, y_coef[::-1] + 1

    view = image[
        int(yslice[0]) : int(yslice[-1] + 1), int(xslice[0]) : int(xslice[-1] + 1)
    ]
    im_slice = lut[view]
    new = (
        y_inv_coef * (x_inv_coef * mapLU[im_slice] + x_coef * mapRU[im_slice])
        + y_coef * (x_inv_coef * mapLB[im_slice] + x_coef * mapRB[im_slice])
    ) / norm
    view[:, :] = new
    return image
