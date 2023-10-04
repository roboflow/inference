import os
from typing import Union, List, Generator, Optional

import cv2
import numpy as np
import requests
from PIL import Image
import supervision as sv

from clients.http.entities import ImagesReference
from clients.http.errors import InvalidInputFormatError
from clients.http.utils.encoding import (
    encode_base_64,
    encode_numpy_array,
    encode_pillow_image,
)


def load_stream_inference_input(
    input_uri: str,
    image_extensions: Optional[List[str]],
) -> Generator[np.ndarray, None, None]:
    if os.path.isdir(input_uri):
        yield from load_directory_inference_input(
            directory_path=input_uri, image_extensions=image_extensions
        )
    else:
        yield from sv.get_video_frames_generator(source_path=input_uri)


def load_directory_inference_input(
    directory_path: str,
    image_extensions: Optional[List[str]],
) -> Generator[np.ndarray, None, None]:
    for path in sv.list_files_with_extensions(
        directory=directory_path,
        extensions=image_extensions,
    ):
        yield cv2.imread(path.as_posix())


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
    if issubclass(type(inference_input), Image.Image):
        return [encode_pillow_image(image=inference_input)]
    raise InvalidInputFormatError(
        f"Unknown type of input ({inference_input.__class__.__name__}) submitted."
    )


def load_image_from_uri(uri: str) -> str:
    if uri_is_http_link(uri=uri):
        return load_file_from_url(url=uri)
    local_image = cv2.imread(uri)
    return encode_numpy_array(array=local_image)


def load_file_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return encode_base_64(response.content)


def uri_is_http_link(uri: str) -> bool:
    return "http://" in uri or "https://" in uri
