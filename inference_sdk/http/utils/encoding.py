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
    _, img_encoded = cv2.imencode(".jpg", image)
    image_bytes = np.array(img_encoded).tobytes()
    return encode_base_64(payload=image_bytes)


def pillow_image_to_base64_jpeg(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        return encode_base_64(payload=buffer.getvalue())


def encode_base_64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("utf-8")


def bytes_to_opencv_image(
    payload: bytes, array_type: np.number = np.uint8
) -> np.ndarray:
    bytes_array = np.frombuffer(payload, dtype=array_type)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    if decoding_result is None:
        raise EncodingError("Could not encode bytes to OpenCV image.")
    return decoding_result


def bytes_to_pillow_image(payload: bytes) -> Image.Image:
    buffer = BytesIO(payload)
    try:
        return Image.open(buffer)
    except UnidentifiedImageError as error:
        raise EncodingError("Could not encode bytes to PIL image.") from error
