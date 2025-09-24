import base64
import os
from typing import Generator, List, Optional, Tuple, Union

import aiohttp
import cv2
import numpy as np
import requests
import supervision as sv
from PIL import Image

from inference_sdk.http.entities import ImagesReference
from inference_sdk.http.errors import EncodingError, InvalidInputFormatError
from inference_sdk.http.utils.encoding import (
    bytes_to_opencv_image,
    encode_base_64,
    numpy_array_to_base64_jpeg,
    pillow_image_to_base64_jpeg,
)
from inference_sdk.http.utils.pre_processing import (
    resize_opencv_image,
    resize_pillow_image,
)


def load_stream_inference_input(
    input_uri: str,
    image_extensions: Optional[List[str]],
) -> Generator[Tuple[Union[str, int], np.ndarray], None, None]:
    """Load an inference input from a stream.

    Args:
        input_uri: The URI of the input.
        image_extensions: The extensions of the images.

    Returns:
        The generator of the inference input.
    """
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
    """Load an inference input from a directory.

    Args:
        directory_path: The path to the directory.
        image_extensions: The extensions of the images.

    Returns:
        The generator of the inference input.
    """
    paths = {
        path.as_posix().lower()
        for path in sv.list_files_with_extensions(
            directory=directory_path,
            extensions=image_extensions,
        )
    }
    # making a set due to case-insensitive behaviour of Windows
    # see: https://stackoverflow.com/questions/7199039/file-paths-in-windows-environment-not-case-sensitive
    for path in paths:
        yield path, cv2.imread(path)


def load_nested_batches_of_inference_input(
    inference_input: Union[list, ImagesReference],
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Union[Tuple[str, Optional[float]], list]:
    """Load a nested batch of inference input.

    Args:
        inference_input: The inference input.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The nested batch of inference input.
    """
    if not isinstance(inference_input, list):
        return load_static_inference_input(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )[0]
    result = []
    for element in inference_input:
        result.append(
            load_nested_batches_of_inference_input(
                inference_input=element,
                max_height=max_height,
                max_width=max_width,
            )
        )
    return result


def load_static_inference_input(
    inference_input: Union[ImagesReference, List[ImagesReference]],
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> List[Tuple[str, Optional[float]]]:
    """Load a static inference input.

    Args:
        inference_input: The inference input.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The list of the inference input.
    """
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
            load_image_from_string(
                reference=inference_input, max_height=max_height, max_width=max_width
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


async def load_static_inference_input_async(
    inference_input: Union[ImagesReference, List[ImagesReference]],
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> List[Tuple[str, Optional[float]]]:
    """Load a static inference input asynchronously.

    Args:
        inference_input: The inference input.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The list of the inference input.
    """
    if issubclass(type(inference_input), list):
        results = []
        for element in inference_input:
            results.extend(
                await load_static_inference_input_async(
                    inference_input=element,
                    max_height=max_height,
                    max_width=max_width,
                )
            )
        return results
    if issubclass(type(inference_input), str):
        return [
            await load_image_from_string_async(
                reference=inference_input, max_height=max_height, max_width=max_width
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


def load_image_from_string(
    reference: str,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    """Load an image from a string.

    Args:
        reference: The reference to the image.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The image and the scaling factor.
    """
    if uri_is_http_link(uri=reference):
        return load_image_from_url(
            url=reference, max_height=max_height, max_width=max_width
        )
    if os.path.exists(reference):
        if max_height is None or max_width is None:
            with open(reference, "rb") as f:
                img_bytes = f.read()
            img_base64_str = encode_base_64(payload=img_bytes)
            return img_base64_str, None
        local_image = cv2.imread(reference)
        if local_image is None:
            raise EncodingError(f"Could not load image from {reference}")
        local_image, scaling_factor = resize_opencv_image(
            image=local_image,
            max_height=max_height,
            max_width=max_width,
        )
        return numpy_array_to_base64_jpeg(image=local_image), scaling_factor
    if max_height is not None and max_width is not None:
        image_bytes = base64.b64decode(reference)
        image = bytes_to_opencv_image(payload=image_bytes)
        image, scaling_factor = resize_opencv_image(
            image=image,
            max_height=max_height,
            max_width=max_width,
        )
        return numpy_array_to_base64_jpeg(image=image), scaling_factor
    return reference, None


async def load_image_from_string_async(
    reference: str,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    """Load an image from a string asynchronously.

    Args:
        reference: The reference to the image.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The image and the scaling factor.
    """
    if uri_is_http_link(uri=reference):
        return await load_image_from_url_async(
            url=reference, max_height=max_height, max_width=max_width
        )
    if os.path.exists(reference):
        local_image = cv2.imread(reference)
        if local_image is None:
            raise EncodingError(f"Could not load image from {reference}")
        local_image, scaling_factor = resize_opencv_image(
            image=local_image,
            max_height=max_height,
            max_width=max_width,
        )
        return numpy_array_to_base64_jpeg(image=local_image), scaling_factor
    if max_height is not None and max_width is not None:
        image_bytes = base64.b64decode(reference)
        image = bytes_to_opencv_image(payload=image_bytes)
        image, scaling_factor = resize_opencv_image(
            image=image,
            max_height=max_height,
            max_width=max_width,
        )
        return numpy_array_to_base64_jpeg(image=image), scaling_factor
    return reference, None


def load_image_from_url(
    url: str,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    """Load an image from a URL.

    Args:
        url: The URL of the image.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The image and the scaling factor.
    """
    response = requests.get(url)
    response.raise_for_status()
    if max_height is None or max_width is None:
        return encode_base_64(response.content), None
    image = bytes_to_opencv_image(payload=response.content)
    resized_image, scaling_factor = resize_opencv_image(
        image=image,
        max_height=max_height,
        max_width=max_width,
    )
    serialised_image = numpy_array_to_base64_jpeg(image=resized_image)
    return serialised_image, scaling_factor


async def load_image_from_url_async(
    url: str,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    """Load an image from a URL asynchronously.

    Args:
        url: The URL of the image.
        max_height: The maximum height of the image.
        max_width: The maximum width of the image.

    Returns:
        The image and the scaling factor.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            response_payload = await response.read()
    if max_height is None or max_width is None:
        return encode_base_64(response_payload), None
    image = bytes_to_opencv_image(payload=response_payload)
    resized_image, scaling_factor = resize_opencv_image(
        image=image,
        max_height=max_height,
        max_width=max_width,
    )
    serialised_image = numpy_array_to_base64_jpeg(image=resized_image)
    return serialised_image, scaling_factor


def uri_is_http_link(uri: str) -> bool:
    """Check if the URI is an HTTP link.

    Args:
        uri: The URI to check.

    Returns:
        True if the URI is an HTTP link, False otherwise.
    """
    return uri.startswith("http://") or uri.startswith("https://")
