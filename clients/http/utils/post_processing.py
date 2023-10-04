import base64
from typing import Union

import numpy as np
from PIL import Image
from requests import Response

from clients.http.entities import VisualisationResponseFormat
from clients.http.utils.encoding import (
    encode_base_64,
    bytes_to_np_array,
    bytes_to_pillow_image,
)

CONTENT_TYPE_HEADERS = ["content-type", "Content-Type"]
IMAGES_TRANSCODING_METHODS = {
    VisualisationResponseFormat.BASE64: encode_base_64,
    VisualisationResponseFormat.NUMPY: bytes_to_np_array,
    VisualisationResponseFormat.PILLOW: bytes_to_pillow_image,
}


def response_contains_jpeg_image(response: Response) -> bool:
    content_type = None
    for header_name in CONTENT_TYPE_HEADERS:
        if header_name in response.headers:
            content_type = response.headers[header_name]
            break
    if content_type is None:
        return False
    return "image/jpeg" in content_type


def transform_base64_visualisation(
    visualisation: str,
    expected_format: VisualisationResponseFormat,
) -> Union[str, np.ndarray, Image.Image]:
    visualisation_bytes = base64.b64decode(visualisation)
    return transform_visualisation_bytes(
        visualisation=visualisation_bytes, expected_format=expected_format
    )


def transform_visualisation_bytes(
    visualisation: bytes,
    expected_format: VisualisationResponseFormat,
) -> Union[str, np.ndarray, Image.Image]:
    if expected_format not in IMAGES_TRANSCODING_METHODS:
        raise NotImplementedError(
            f"Expected format: {expected_format} is not supported in terms of visualisations transcoding."
        )
    transcoding_method = IMAGES_TRANSCODING_METHODS[expected_format]
    return transcoding_method(visualisation)
