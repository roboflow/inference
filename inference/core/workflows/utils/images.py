"""Workflows-local copy of the numpy image-resizing helpers.

`_resize_image_keeping_aspect_ratio` is a copy of
`inference.core.utils.preprocess.resize_image_keeping_aspect_ratio` with only the
`isinstance(image, np.ndarray)` branches kept. The torch branch (guarded by
`USE_PYTORCH_FOR_PREPROCESSING` on the server) is deliberately not carried over:
every workflows caller passes a numpy image.
"""

from typing import Any, Tuple, Union

import cv2
import numpy as np
from _io import _IOBase

from inference.core.workflows.prototypes.image_codec import get_image_codec
from inference.core.workflows.utils.image_encoding import encode_image_to_jpeg_bytes

__all__ = [
    "attempt_loading_image_from_string",
    "downscale_image_keeping_aspect_ratio",
    "encode_image_to_jpeg_bytes",
    "ensure_local_image_load_allowed",
    "load_image",
    "load_image_from_url",
]


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


def load_image(
    value: Any,
    disable_preproc_auto_orient: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Load an `inference`-format image through the installed `ImageCodec`.

    Same name, signature and `(image, is_bgr)` return as
    `inference.core.utils.image_utils.load_image`, deliberately: the 24 blocks
    calling it were repointed by changing only the module path in their import
    line. The URL, local-file and pickle policy this used to apply directly now
    comes from the host's codec.
    """
    return get_image_codec().load_image(
        value, disable_preproc_auto_orient=disable_preproc_auto_orient
    )


def load_image_from_url(
    value: str, cv_imread_flags: int = cv2.IMREAD_COLOR
) -> np.ndarray:
    """Fetch an image over http(s) through the installed `ImageCodec`.

    The keyword name `value` is part of the contract: `entities/base.py` calls it
    by keyword and `visual_search/test_v1.py:117` asserts on that call.
    """
    return get_image_codec().fetch_url(value, cv_imread_flags=cv_imread_flags)


def attempt_loading_image_from_string(
    value: Union[str, bytes, bytearray, _IOBase],
    cv_imread_flags: int = cv2.IMREAD_COLOR,
) -> Tuple[np.ndarray, bool]:
    """Decode an in-memory image payload through the installed `ImageCodec`.

    Routed rather than vendored because the host implementation's last fallback
    is a pickled-numpy payload behind `ALLOW_NUMPY_INPUT` - a gate Workflows must
    not own.
    """
    return get_image_codec().decode_string(value, cv_imread_flags=cv_imread_flags)


def ensure_local_image_load_allowed(path: str) -> None:
    """Ask the installed `ImageCodec` whether reading `path` off disk is allowed.

    Called before the two local-file decoders Workflows keeps for itself:
    `cv2.imread` in `WorkflowImageData.numpy_image` and
    `torchvision.io.read_file` + `decode_image` in
    `WorkflowImageData._decode_source_to_tensor`.
    """
    get_image_codec().ensure_local_file_load_allowed(path)
