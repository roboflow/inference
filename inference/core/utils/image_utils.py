import base64
import os
import pickle
import re
from io import BytesIO
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

from inference.core.data_models import InferenceRequestImage
from inference.core.env import ALLOW_NUMPY_INPUT
from inference.core.exceptions import InvalidNumpyInput


def load_image(value: Any) -> Image.Image:
    """Loads an image based on the specified type and value.

    Args:
        value (Any): Image value which could be an instance of InferenceRequestImage,
            a dict with 'type' and 'value' keys, or inferred based on the value's content.

    Returns:
        Image.Image: The loaded PIL image, converted to RGB.

    Raises:
        NotImplementedError: If the specified image type is not supported.
        InvalidNumpyInput: If the numpy input method is used and the input data is invalid.
    """
    type = None
    if isinstance(value, InferenceRequestImage):
        type = value.type
        value = value.value
    elif isinstance(value, dict):
        type = value.get("type")
        value = value.get("value")

    if type is not None:
        if type == "base64":
            pil_image = load_image_base64(value)
        elif type == "file":
            pil_image = Image.open(value)
        elif type == "multipart":
            pil_image = load_image_multipart(value)
        elif type == "numpy" and ALLOW_NUMPY_INPUT:
            pil_image = load_image_numpy_str(value)
        elif type == "pil":
            pil_image = value
        elif type == "url":
            pil_image = load_image_url(value)
        else:
            raise NotImplementedError(f"Image type '{type}' is not supported.")
    else:
        pil_image = load_image_inferred(value)

    return pil_image.convert("RGB")


def load_image_inferred(value: Any) -> Image.Image:
    """Tries to infer the image type from the value and loads it.

    Args:
        value (Any): Image value to infer and load.

    Returns:
        Image.Image: The loaded PIL image.

    Raises:
        NotImplementedError: If the image type could not be inferred.
    """
    if isinstance(value, Image.Image):
        return value
    elif isinstance(value, (np.ndarray, np.generic)):
        return Image.fromarray(value)
    elif isinstance(value, str) and (value.startswith("http")):
        return load_image_url(value)
    elif isinstance(value, str) and os.path.exists(value):
        return Image.open(value)
    elif isinstance(value, str):
        try:
            return load_image_base64(value)
        except Exception:
            pass
        try:
            return load_image_multipart(value)
        except Exception:
            pass
        try:
            return load_image_numpy_str(value)
        except Exception:
            pass
    raise NotImplementedError(
        f"Could not infer image type from value of type {type(value)}."
    )


pattern = re.compile(r"^data:image\/[a-z]+;base64,")


def load_image_base64(value):
    """Loads an image from a base64 encoded string.

    Args:
        value (str): Base64 encoded string representing the image.

    Returns:
        Image.Image: The loaded PIL image.
    """
    # New routes accept images via json body (str), legacy routes accept bytes which need to be decoded as strings
    if not isinstance(value, str):
        value = value.decode("utf-8")
    try:
        value = base64.b64decode(value)
        return Image.open(BytesIO(value))
    except Exception:
        # Sometimes base64 strings that were encoded by a browser are padded with extra characters, so we need to remove them
        value = pattern.sub("", value)
        value = base64.b64decode(value)
        return Image.open(BytesIO(value))


def load_image_multipart(value):
    """Loads an image from a multipart-encoded input.

    Args:
        value (Any): Multipart-encoded input representing the image.

    Returns:
        Image.Image: The loaded PIL image.
    """
    return Image.open(value)


def load_image_numpy_str(value):
    """Loads an image from a numpy array string.

    Args:
        value (str): String representing the numpy array of the image.

    Returns:
        Image.Image: The loaded PIL image.

    Raises:
        InvalidNumpyInput: If the numpy data is invalid.
    """
    data = pickle.loads(value)
    try:
        return Image.fromarray(data)
    except Exception as e:
        if len(data.shape) != 3 and len(data.shape) != 2:
            raise InvalidNumpyInput(
                f"Expected 2 or 3 dimensions, got {len(data.shape)} dimensions."
            )
        elif data.shape[-1] != 3 and data.shape[-1] != 1:
            raise InvalidNumpyInput(
                f"Expected 1 or 3 channels, got {data.shape[-1]} channels."
            )
        elif max(data) > 255 or min(data) < 0:
            raise InvalidNumpyInput(
                f"Expected values between 0 and 255, got values between {min(data)} and {max(data)}."
            )
        else:
            raise e


def load_image_url(value):
    """Loads an image from a given URL.

    Args:
        value (str): URL of the image.

    Returns:
        Image.Image: The loaded PIL image.
    """
    return Image.open(requests.get(value, stream=True).raw)
