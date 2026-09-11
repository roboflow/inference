"""Pure image encode/decode helpers used by Workflows.

Copied from `inference.core.utils.image_utils` so that `inference/core/workflows`
does not import the server package for arithmetic. Every function here is a pure
transformation of bytes and arrays: no network, no filesystem, no `pickle`, no
environment reads. Everything needing one of those lives behind the `ImageCodec`
port (`inference.core.workflows.prototypes.image_codec`).

`encode_image_to_jpeg_bytes`, `choose_image_decoding_flags` and
`convert_gray_image_to_bgr` are byte-for-byte copies. `decode_base64_image`,
`decode_encoded_image_bytes` and `ensure_valid_numpy_image` copy
`load_image_base64`, `load_image_from_encoded_bytes` and `validate_numpy_image`
with `inference.core.exceptions.InputImageLoadError` / `InvalidNumpyInput`
replaced by `WorkflowImageLoadError`. Do not "improve" any of them:
`tests/workflows/unit_tests/utils/test_image_encoding.py` pins them to the
originals by encoded bytes, decoded pixels, EXIF behaviour and signature.
"""

import binascii
import re
from typing import Union

import cv2
import numpy as np
import pybase64

from inference.core.workflows.errors import WorkflowImageLoadError

BASE64_DATA_TYPE_PATTERN = re.compile(r"^data:image\/[a-z]+;base64,")


def encode_image_to_jpeg_bytes(image: np.ndarray, jpeg_quality: int = 90) -> bytes:
    """Encode a BGR numpy image to JPEG bytes."""
    encoding_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, img_encoded = cv2.imencode(".jpg", image, encoding_param)
    return np.array(img_encoded).tobytes()


def choose_image_decoding_flags(disable_preproc_auto_orient: bool) -> int:
    """Pick the OpenCV decoding flags for the requested auto-orient policy."""
    cv_imread_flags = cv2.IMREAD_COLOR
    if disable_preproc_auto_orient:
        cv_imread_flags = cv_imread_flags | cv2.IMREAD_IGNORE_ORIENTATION
    return cv_imread_flags


def convert_gray_image_to_bgr(image: np.ndarray) -> np.ndarray:
    """Expand a single-channel image to 3-channel BGR; pass BGR through."""
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def decode_base64_image(
    value: Union[str, bytes], cv_imread_flags=cv2.IMREAD_COLOR
) -> np.ndarray:
    """Decode a base64 payload (optionally data-URL prefixed) into a BGR image."""
    if not isinstance(value, str):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WorkflowImageLoadError(
                public_message="Invalid base64 input: the image payload contains raw "
                "bytes instead of a base64-encoded string.",
                context="workflow_execution | image_decoding",
                inner_error=error,
            ) from error
    value = BASE64_DATA_TYPE_PATTERN.sub("", value)
    try:
        decoded = pybase64.b64decode(value)
    except binascii.Error as error:
        raise WorkflowImageLoadError(
            public_message="Malformed base64 input image.",
            context="workflow_execution | image_decoding",
            inner_error=error,
        ) from error
    if len(decoded) == 0:
        raise WorkflowImageLoadError(
            public_message="Empty image payload.",
            context="workflow_execution | image_decoding",
        )
    image_np = np.frombuffer(decoded, np.uint8)
    result = cv2.imdecode(image_np, cv_imread_flags)
    if result is None:
        raise WorkflowImageLoadError(
            public_message="Malformed base64 input image.",
            context="workflow_execution | image_decoding",
        )
    return result


def decode_encoded_image_bytes(
    value: bytes, cv_imread_flags: int = cv2.IMREAD_COLOR
) -> np.ndarray:
    """Decode raw encoded image bytes (JPEG/PNG/...) into a BGR image."""
    image_np = np.asarray(bytearray(value), dtype=np.uint8)
    image = cv2.imdecode(image_np, cv_imread_flags)
    if image is None:
        raise WorkflowImageLoadError(
            public_message="Data is not image.",
            context="workflow_execution | image_decoding",
        )
    return image


def ensure_valid_numpy_image(data: np.ndarray) -> np.ndarray:
    """Validate an array is a usable image and return it unchanged."""
    if not issubclass(type(data), np.ndarray):
        raise WorkflowImageLoadError(
            public_message="Data provided as input could not be decoded into "
            "np.ndarray object.",
            context="workflow_execution | image_decoding",
        )
    if len(data.shape) != 3 and len(data.shape) != 2:
        raise WorkflowImageLoadError(
            public_message="For image given as np.ndarray expected 2 or 3 dimensions.",
            context="workflow_execution | image_decoding",
        )
    if len(data.shape) == 3 and data.shape[-1] != 3 and data.shape[-1] != 1:
        raise WorkflowImageLoadError(
            public_message="For image given as np.ndarray expected 1 or 3 channels.",
            context="workflow_execution | image_decoding",
        )
    return data
