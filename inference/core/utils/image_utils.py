import os
import pickle
import re
import traceback
from typing import Any

import cv2
import numpy as np
import pybase64
import requests
from PIL import Image

from inference.core.data_models import InferenceRequestImage
from inference.core.env import ALLOW_NUMPY_INPUT
from inference.core.exceptions import InvalidNumpyInput


def load_image(value: Any, disable_preproc_auto_orient=False) -> np.ndarray:
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
    cv_imread_flags = cv2.IMREAD_COLOR
    if disable_preproc_auto_orient:
        cv_imread_flags = cv_imread_flags | cv2.IMREAD_IGNORE_ORIENTATION
    type = None
    if isinstance(value, InferenceRequestImage):
        type = value.type
        value = value.value
    elif isinstance(value, dict):
        type = value.get("type")
        value = value.get("value")
    is_bgr = True
    if type is not None:
        if type == "base64":
            np_image = load_image_base64(value, cv_imread_flags=cv_imread_flags)
        elif type == "file":
            np_image = cv2.imread(value, cv_imread_flags=cv_imread_flags)
        elif type == "multipart":
            np_image = load_image_multipart(value, cv_imread_flags=cv_imread_flags)
        elif type == "numpy" and ALLOW_NUMPY_INPUT:
            np_image = load_image_numpy_str(value)
        elif type == "pil":
            np_image = np.asarray(value.convert("RGB"))
            is_bgr = False
        elif type == "url":
            np_image = load_image_url(value, cv_imread_flags=cv_imread_flags)
        else:
            raise NotImplementedError(f"Image type '{type}' is not supported.")
    else:
        np_image, is_bgr = load_image_inferred(value, cv_imread_flags=cv_imread_flags)

    if len(np_image.shape) == 2 or np_image.shape[2] == 1:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

    return np_image, is_bgr


def load_image_rgb(value: Any, disable_preproc_auto_orient=False) -> np.ndarray:
    np_image, is_bgr = load_image(
        value, disable_preproc_auto_orient=disable_preproc_auto_orient
    )
    if is_bgr:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return np_image


def load_image_inferred(value: Any, cv_imread_flags=cv2.IMREAD_COLOR) -> Image.Image:
    """Tries to infer the image type from the value and loads it.

    Args:
        value (Any): Image value to infer and load.

    Returns:
        Image.Image: The loaded PIL image.

    Raises:
        NotImplementedError: If the image type could not be inferred.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        return value, True
    elif isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB")), False
    elif isinstance(value, str) and (value.startswith("http")):
        return load_image_url(value, cv_imread_flags=cv_imread_flags), True
    elif isinstance(value, str) and os.path.exists(value):
        return cv2.imread(value, cv_imread_flags), True
    elif isinstance(value, str):
        try:
            return load_image_base64(value, cv_imread_flags=cv_imread_flags), True
        except Exception:
            pass
        try:
            return load_image_multipart(value, cv_imread_flags=cv_imread_flags), True
        except Exception:
            pass
        try:
            return load_image_numpy_str(value), True
        except Exception:
            pass
    raise NotImplementedError(
        f"Could not infer image type from value of type {type(value)}."
    )


pattern = re.compile(r"^data:image\/[a-z]+;base64,")


def load_image_base64(value: str, cv_imread_flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """Loads an image from a base64 encoded string using OpenCV.

    Args:
        value (str): Base64 encoded string representing the image.

    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    # New routes accept images via json body (str), legacy routes accept bytes which need to be decoded as strings
    if not isinstance(value, str):
        value = value.decode("utf-8")

    try:
        value = pybase64.b64decode(value)
        image_np = np.frombuffer(value, np.uint8)
        return cv2.imdecode(image_np, cv_imread_flags)
    except Exception as e:
        # The variable "pattern" isn't defined in the original function. Assuming it exists somewhere in your code.
        # Sometimes base64 strings that were encoded by a browser are padded with extra characters, so we need to remove them
        # print traceback
        traceback.print_exc()
        value = pattern.sub("", value)
        value = pybase64.b64decode(value)
        image_np = np.frombuffer(value, np.uint8)
        return cv2.imdecode(image_np, cv_imread_flags)


def load_image_multipart(value, cv_imread_flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """Loads an image from a multipart-encoded input.

    Args:
        value (Any): Multipart-encoded input representing the image.

    Returns:
        Image.Image: The loaded PIL image.
    """
    value.seek(0)
    image_np = np.frombuffer(value.read(), np.uint8)
    return cv2.imdecode(image_np, cv_imread_flags)


def load_image_numpy_str(value: str) -> np.ndarray:
    """Loads an image from a numpy array string.

    Args:
        value (str): String representing the numpy array of the image.

    Returns:
        Image.Image: The loaded PIL image.

    Raises:
        InvalidNumpyInput: If the numpy data is invalid.
    """
    data = pickle.loads(value)
    assert isinstance(data, np.ndarray)
    assert len(data.shape) == 3 or len(data.shape) == 2
    assert data.shape[-1] == 3 or data.shape[-1] == 1
    assert data.max() <= 255 and data.min() >= 0
    try:
        return data
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


def load_image_url(value: str, cv_imread_flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """Loads an image from a given URL.

    Args:
        value (str): URL of the image.

    Returns:
        Image.Image: The loaded PIL image.
    """
    response = requests.get(value, stream=True)
    image_np = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_np, cv_imread_flags)
