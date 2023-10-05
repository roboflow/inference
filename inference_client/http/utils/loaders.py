import os
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import supervision as sv
from PIL import Image

from inference_client.http.entities import ImagesReference
from inference_client.http.errors import EncodingError, InvalidInputFormatError
from inference_client.http.utils.encoding import (
    bytes_to_opencv_image,
    encode_base_64,
    numpy_array_to_base64_jpeg,
    pillow_image_to_base64_jpeg,
)
from inference_client.http.utils.pre_processing import (
    resize_opencv_image,
    resize_pillow_image,
)


def load_stream_inference_input(
    input_uri: str,
    image_extensions: Optional[List[str]],
) -> Generator[Tuple[Union[str, int], np.ndarray], None, None]:
    if os.path.isdir(input_uri):
        yield from load_directory_inference_input(
            directory_path=input_uri, image_extensions=image_extensions
        )
    else:
        yield from enumerate(sv.get_video_frames_generator(source_path=input_uri))


def load_directory_inference_input(
    directory_path: str,
    image_extensions: Optional[List[str]],
) -> Generator[Tuple[Union[str, int], np.ndarray], None, None]:
    for path in sv.list_files_with_extensions(
        directory=directory_path,
        extensions=image_extensions,
    ):
        yield path, cv2.imread(path.as_posix())


def load_static_inference_input(
    inference_input: Union[ImagesReference, List[ImagesReference]],
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> List[Tuple[str, Optional[float]]]:
    if issubclass(type(inference_input), list):
        results = []
        for element in inference_input:
            results.extend(
                load_static_inference_input(
                    inference_input=element,
                    max_height=max_height,
                    max_width=max_width,
                )
            )
        return results
    if issubclass(type(inference_input), str):
        return [
            load_image_from_uri(
                uri=inference_input, max_height=max_height, max_width=max_width
            )
        ]
    if issubclass(type(inference_input), np.ndarray):
        image, scaling_factor = resize_opencv_image(
            image=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        return [(numpy_array_to_base64_jpeg(image=image), scaling_factor)]
    if issubclass(type(inference_input), Image.Image):
        image, scaling_factor = resize_pillow_image(
            image=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        return [(pillow_image_to_base64_jpeg(image=image), scaling_factor)]
    raise InvalidInputFormatError(
        f"Unknown type of input ({inference_input.__class__.__name__}) submitted."
    )


def load_image_from_uri(
    uri: str,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    if uri_is_http_link(uri=uri):
        return load_image_from_url(url=uri, max_height=max_height, max_width=max_width)
    local_image = cv2.imread(uri)
    if local_image is None:
        raise EncodingError(f"Could not load image from {uri}")
    local_image, scaling_factor = resize_opencv_image(
        image=local_image,
        max_height=max_height,
        max_width=max_width,
    )
    return numpy_array_to_base64_jpeg(image=local_image), scaling_factor


def load_image_from_url(
    url: str,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    response = requests.get(url)
    response.raise_for_status()
    image = bytes_to_opencv_image(payload=response.content)
    if max_height is None or max_width is None:
        return encode_base_64(response.content), None
    resized_image, scaling_factor = resize_opencv_image(
        image=image,
        max_height=max_height,
        max_width=max_width,
    )
    serialised_image = numpy_array_to_base64_jpeg(image=resized_image)
    return serialised_image, scaling_factor


def uri_is_http_link(uri: str) -> bool:
    return "http://" in uri or "https://" in uri
