from typing import Union, List

import cv2
import numpy as np
import requests
from PIL.Image import Image

from clients.http.entities import ImagesReference
from clients.http.errors import InvalidInputFormatError
from clients.http.utils.encoding import encode_base_64, encode_numpy_array, encode_pillow_image


def load_static_inference_input(
    inference_input: Union[ImagesReference, List[ImagesReference]]
) -> List[str]:
    if issubclass(type(inference_input), list):
        results = []
        for element in inference_input:
            results.extend(load_static_inference_input(inference_input=element))
        return results
    if issubclass(type(inference_input), str):
        return [load_image_from_uri(uri=inference_input)]
    if issubclass(type(inference_input), np.ndarray):
        return [encode_numpy_array(array=inference_input)]
    if issubclass(type(inference_input), Image):
        return [encode_pillow_image(image=inference_input)]
    raise InvalidInputFormatError(
        f"Unknown type of input ({inference_input.__class__.__name__}) submitted."
    )


def load_image_from_uri(uri: str) -> str:
    if "http://" in uri or "https://" in uri:
        return load_file_from_url(url=uri)
    local_image = cv2.imread(uri)
    return encode_numpy_array(array=local_image, as_jpeg=True)


def load_file_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return encode_base_64(response.content)

