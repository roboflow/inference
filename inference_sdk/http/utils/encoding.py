import base64
from io import BytesIO
from typing import Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from inference_sdk.http.errors import EncodingError


def numpy_array_to_base64_jpeg(
    image: np.ndarray,
) -> Union[str]:
    """Encode a numpy array to a base64 JPEG string.

    Args:
        image: The numpy array to encode.

    Returns:
        The base64 JPEG string.
    """
    _, img_encoded = cv2.imencode(".jpg", image)
    image_bytes = np.array(img_encoded).tobytes()
    return encode_base_64(payload=image_bytes)


def pillow_image_to_base64_jpeg(image: Image.Image) -> str:
    """Encode a PIL image to a base64 JPEG string.

    Args:
        image: The PIL image to encode.

    Returns:
        The base64 JPEG string.
    """
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        return encode_base_64(payload=buffer.getvalue())


def encode_base_64(payload: bytes) -> str:
    """Encode a bytes object to a base64 string.

    Args:
        payload: The bytes object to encode.

    Returns:
        The base64 string.
    """
    return base64.b64encode(payload).decode("utf-8")


def bytes_to_opencv_image(
    payload: bytes, array_type: np.number = np.uint8
) -> np.ndarray:
    """Decode a bytes object to an OpenCV image.

    Args:
        payload: The bytes object to decode.
        array_type: The type of the array.

    Returns:
        The OpenCV image.
    """
    bytes_array = np.frombuffer(payload, dtype=array_type)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    if decoding_result is None:
        raise EncodingError("Could not encode bytes to OpenCV image.")
    return decoding_result


def bytes_to_pillow_image(payload: bytes) -> Image.Image:
    """Decode a bytes object to a PIL image.

    Args:
        payload: The bytes object to decode.

    Returns:
        The PIL image.
    """
    buffer = BytesIO(payload)
    try:
        return Image.open(buffer)
    except UnidentifiedImageError as error:
        raise EncodingError("Could not encode bytes to PIL image.") from error
