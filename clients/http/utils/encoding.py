import base64
import pickle
from io import BytesIO
from typing import Union

import cv2
import numpy as np
from PIL import Image


def encode_numpy_array(
    array: np.ndarray,
    as_jpeg: bool = True,
) -> Union[str, bytes]:
    if as_jpeg:
        _, img_encoded = cv2.imencode(".jpg", array)
        image_bytes = np.array(img_encoded).tobytes()
        return encode_base_64(payload=image_bytes)
    return pickle.dumps(array)


def encode_pillow_image(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        return encode_base_64(payload=buffer.getvalue())


def encode_base_64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("utf-8")


def bytes_to_opencv_image(
    payload: bytes, array_type: np.number = np.uint8
) -> np.ndarray:
    bytes_array = np.frombuffer(payload, dtype=array_type)
    return cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)


def bytes_to_pillow_image(payload: bytes) -> Image.Image:
    buffer = BytesIO(payload)
    return Image.open(buffer)
