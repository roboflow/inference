import base64
import pickle
from io import BytesIO
from typing import Union

import cv2
import numpy as np
from PIL.Image import Image


def encode_numpy_array(
    array: np.ndarray,
    as_jpeg: bool = False,
) -> Union[str, bytes]:
    if as_jpeg:
        _, img_encoded = cv2.imencode(".jpg", array)
        image_bytes = np.array(img_encoded).tobytes()
        return encode_base_64(payload=image_bytes)
    return pickle.dumps(array)


def encode_pillow_image(image: Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        return encode_base_64(payload=buffer.getvalue())


def encode_base_64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("utf-8")
