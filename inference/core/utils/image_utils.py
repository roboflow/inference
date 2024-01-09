import binascii
import os
import pickle
import re
from enum import Enum
from io import BytesIO
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
import pybase64
import requests
from _io import _IOBase
from PIL import Image
from requests import RequestException

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.env import ALLOW_NUMPY_INPUT
from inference.core.exceptions import (
    InputFormatInferenceFailed,
    InputImageLoadError,
    InvalidImageTypeDeclared,
    InvalidNumpyInput,
)
from inference.core.utils.requests import api_key_safe_raise_for_status

BASE64_DATA_TYPE_PATTERN = re.compile(r"^data:image\/[a-z]+;base64,")


class ImageType(Enum):
    BASE64 = "base64"
    FILE = "file"
    MULTIPART = "multipart"
    NUMPY = "numpy"
    PILLOW = "pil"
    URL = "url"


def load_image_rgb(value: Any, disable_preproc_auto_orient: bool = False) -> np.ndarray:
    np_image, is_bgr = load_image(
        value=value, disable_preproc_auto_orient=disable_preproc_auto_orient
    )
    if is_bgr:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return np_image


def load_image(
    value: Any,
    disable_preproc_auto_orient: bool = False,
) -> Tuple[np.ndarray, bool]:
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
    cv_imread_flags = choose_image_decoding_flags(
        disable_preproc_auto_orient=disable_preproc_auto_orient
    )
    value, image_type = extract_image_payload_and_type(value=value)
    if image_type is not None:
        np_image, is_bgr = load_image_with_known_type(
            value=value,
            image_type=image_type,
            cv_imread_flags=cv_imread_flags,
        )
    else:
        np_image, is_bgr = load_image_with_inferred_type(
            value, cv_imread_flags=cv_imread_flags
        )
    np_image = convert_gray_image_to_bgr(image=np_image)
    return np_image, is_bgr


def choose_image_decoding_flags(disable_preproc_auto_orient: bool) -> int:
    """Choose the appropriate OpenCV image decoding flags.

    Args:
        disable_preproc_auto_orient (bool): Flag to disable preprocessing auto-orientation.

    Returns:
        int: OpenCV image decoding flags.
    """
    cv_imread_flags = cv2.IMREAD_COLOR
    if disable_preproc_auto_orient:
        cv_imread_flags = cv_imread_flags | cv2.IMREAD_IGNORE_ORIENTATION
    return cv_imread_flags


def extract_image_payload_and_type(value: Any) -> Tuple[Any, Optional[ImageType]]:
    """Extract the image payload and type from the given value.

    This function supports different types of image inputs (e.g., InferenceRequestImage, dict, etc.)
    and extracts the relevant data and image type for further processing.

    Args:
        value (Any): The input value which can be an image or information to derive the image.

    Returns:
        Tuple[Any, Optional[ImageType]]: A tuple containing the extracted image data and the corresponding image type.
    """
    image_type = None
    if issubclass(type(value), InferenceRequestImage):
        image_type = value.type
        value = value.value
    elif issubclass(type(value), dict):
        image_type = value.get("type")
        value = value.get("value")
    allowed_payload_types = {e.value for e in ImageType}
    if image_type is None:
        return value, image_type
    if image_type.lower() not in allowed_payload_types:
        raise InvalidImageTypeDeclared(
            f"Declared image type: {value} which is not in allowed types: {allowed_payload_types}."
        )
    return value, ImageType(image_type.lower())


def load_image_with_known_type(
    value: Any,
    image_type: ImageType,
    cv_imread_flags: int = cv2.IMREAD_COLOR,
) -> Tuple[np.ndarray, bool]:
    """Load an image using the known image type.

    Supports various image types (e.g., NUMPY, PILLOW, etc.) and loads them into a numpy array format.

    Args:
        value (Any): The image data.
        image_type (ImageType): The type of the image.
        cv_imread_flags (int): Flags used for OpenCV's imread function.

    Returns:
        Tuple[np.ndarray, bool]: A tuple of the loaded image as a numpy array and a boolean indicating if the image is in BGR format.
    """
    if image_type is ImageType.NUMPY and not ALLOW_NUMPY_INPUT:
        raise InvalidImageTypeDeclared(
            f"NumPy image type is not supported in this configuration of `inference`."
        )
    loader = IMAGE_LOADERS[image_type]
    is_bgr = True if image_type is not ImageType.PILLOW else False
    image = loader(value, cv_imread_flags)
    return image, is_bgr


def load_image_with_inferred_type(
    value: Any,
    cv_imread_flags: int = cv2.IMREAD_COLOR,
) -> Tuple[np.ndarray, bool]:
    """Load an image by inferring its type.

    Args:
        value (Any): The image data.
        cv_imread_flags (int): Flags used for OpenCV's imread function.

    Returns:
        Tuple[np.ndarray, bool]: Loaded image as a numpy array and a boolean indicating if the image is in BGR format.

    Raises:
        NotImplementedError: If the image type could not be inferred.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        validate_numpy_image(data=value)
        return value, True
    elif isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB")), False
    elif isinstance(value, str) and (value.startswith("http")):
        return load_image_from_url(value=value, cv_imread_flags=cv_imread_flags), True
    elif isinstance(value, str) and os.path.isfile(value):
        return cv2.imread(value, cv_imread_flags), True
    else:
        return attempt_loading_image_from_string(
            value=value, cv_imread_flags=cv_imread_flags
        )


def attempt_loading_image_from_string(
    value: Union[str, bytes, bytearray, _IOBase],
    cv_imread_flags: int = cv2.IMREAD_COLOR,
) -> Tuple[np.ndarray, bool]:
    """
    Attempt to load an image from a string.

    Args:
        value (Union[str, bytes, bytearray, _IOBase]): The image data in string format.
        cv_imread_flags (int): OpenCV flags used for image reading.

    Returns:
        Tuple[np.ndarray, bool]: A tuple of the loaded image in numpy array format and a boolean flag indicating if the image is in BGR format.
    """
    try:
        return load_image_base64(value=value, cv_imread_flags=cv_imread_flags), True
    except:
        pass
    try:
        return (
            load_image_from_encoded_bytes(value=value, cv_imread_flags=cv_imread_flags),
            True,
        )
    except:
        pass
    try:
        return (
            load_image_from_buffer(value=value, cv_imread_flags=cv_imread_flags),
            True,
        )
    except:
        pass
    try:
        return load_image_from_numpy_str(value=value), True
    except InvalidNumpyInput as error:
        raise InputFormatInferenceFailed(
            "Input image format could not be inferred from string."
        ) from error


def load_image_base64(
    value: Union[str, bytes], cv_imread_flags=cv2.IMREAD_COLOR
) -> np.ndarray:
    """Loads an image from a base64 encoded string using OpenCV.

    Args:
        value (str): Base64 encoded string representing the image.

    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    # New routes accept images via json body (str), legacy routes accept bytes which need to be decoded as strings
    if not isinstance(value, str):
        value = value.decode("utf-8")
    value = BASE64_DATA_TYPE_PATTERN.sub("", value)
    value = pybase64.b64decode(value)
    image_np = np.frombuffer(value, np.uint8)
    result = cv2.imdecode(image_np, cv_imread_flags)
    if result is None:
        raise InputImageLoadError("Could not load valid image from base64 string.")
    return result


def load_image_from_buffer(
    value: _IOBase,
    cv_imread_flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    """Loads an image from a multipart-encoded input.

    Args:
        value (Any): Multipart-encoded input representing the image.

    Returns:
        Image.Image: The loaded PIL image.
    """
    value.seek(0)
    image_np = np.frombuffer(value.read(), np.uint8)
    result = cv2.imdecode(image_np, cv_imread_flags)
    if result is None:
        raise InputImageLoadError("Could not load valid image from buffer.")
    return result


def load_image_from_numpy_str(value: Union[bytes, str]) -> np.ndarray:
    """Loads an image from a numpy array string.

    Args:
        value (Union[bytes, str]): Base64 string or byte sequence representing the pickled numpy array of the image.

    Returns:
        Image.Image: The loaded PIL image.

    Raises:
        InvalidNumpyInput: If the numpy data is invalid.
    """
    try:
        if isinstance(value, str):
            value = pybase64.b64decode(value)
        data = pickle.loads(value)
    except (EOFError, TypeError, pickle.UnpicklingError, binascii.Error) as error:
        raise InvalidNumpyInput(
            f"Could not unpickle image data. Cause: {error}"
        ) from error
    validate_numpy_image(data=data)
    return data


def validate_numpy_image(data: np.ndarray) -> None:
    """
    Validate if the provided data is a valid numpy image.

    Args:
        data (np.ndarray): The numpy array representing an image.

    Raises:
        InvalidNumpyInput: If the provided data is not a valid numpy image.
    """
    if not issubclass(type(data), np.ndarray):
        raise InvalidNumpyInput(
            f"Data provided as input could not be decoded into np.ndarray object."
        )
    if len(data.shape) != 3 and len(data.shape) != 2:
        raise InvalidNumpyInput(
            f"For image given as np.ndarray expected 2 or 3 dimensions, got {len(data.shape)} dimensions."
        )
    if data.shape[-1] != 3 and data.shape[-1] != 1:
        raise InvalidNumpyInput(
            f"For image given as np.ndarray expected 1 or 3 channels, got {data.shape[-1]} channels."
        )


def load_image_from_url(
    value: str, cv_imread_flags: int = cv2.IMREAD_COLOR
) -> np.ndarray:
    """Loads an image from a given URL.

    Args:
        value (str): URL of the image.

    Returns:
        Image.Image: The loaded PIL image.
    """
    try:
        response = requests.get(value, stream=True)
        api_key_safe_raise_for_status(response=response)
        return load_image_from_encoded_bytes(
            value=response.content, cv_imread_flags=cv_imread_flags
        )
    except (RequestException, ConnectionError) as error:
        raise InputImageLoadError(
            f"Error while loading image from url: {value}. Details: {error}"
        )


def load_image_from_encoded_bytes(
    value: bytes, cv_imread_flags: int = cv2.IMREAD_COLOR
) -> np.ndarray:
    """
    Load an image from encoded bytes.

    Args:
        value (bytes): The byte sequence representing the image.
        cv_imread_flags (int): OpenCV flags used for image reading.

    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    image_np = np.asarray(bytearray(value), dtype=np.uint8)
    image = cv2.imdecode(image_np, cv_imread_flags)
    if image is None:
        raise InputImageLoadError(
            f"Could not parse response content from url {value} into image."
        )
    return image


IMAGE_LOADERS = {
    ImageType.BASE64: load_image_base64,
    ImageType.FILE: cv2.imread,
    ImageType.MULTIPART: load_image_from_buffer,
    ImageType.NUMPY: lambda v, _: load_image_from_numpy_str(v),
    ImageType.PILLOW: lambda v, _: np.asarray(v.convert("RGB")),
    ImageType.URL: load_image_from_url,
}


def convert_gray_image_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to BGR format.

    Args:
        image (np.ndarray): The grayscale image.

    Returns:
        np.ndarray: The converted BGR image.
    """

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def np_image_to_base64(image: np.ndarray) -> bytes:
    """
    Convert a numpy image to a base64 encoded byte string.

    Args:
        image (np.ndarray): The numpy array representing an image.

    Returns:
        bytes: The base64 encoded image.
    """
    image = Image.fromarray(image)
    with BytesIO() as buffer:
        image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer.getvalue()


def xyxy_to_xywh(xyxy):
    """
    Convert bounding box format from (xmin, ymin, xmax, ymax) to (xcenter, ycenter, width, height).

    Args:
        xyxy (List[int]): List containing the coordinates in (xmin, ymin, xmax, ymax) format.

    Returns:
        List[int]: List containing the converted coordinates in (xcenter, ycenter, width, height) format.
    """
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])

    return [int(x_temp), int(y_temp), int(w_temp), int(h_temp)]


def encode_image_to_jpeg_bytes(image: np.ndarray, jpeg_quality: int = 90) -> bytes:
    """
    Encode a numpy image to JPEG format in bytes.

    Args:
        image (np.ndarray): The numpy array representing an image.
        jpeg_quality (int): Quality of the JPEG image.

    Returns:
        bytes: The JPEG encoded image.
    """
    encoding_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, img_encoded = cv2.imencode(".jpg", image, encoding_param)
    return np.array(img_encoded).tobytes()
