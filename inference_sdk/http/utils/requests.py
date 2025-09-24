import re
from typing import List, Optional, Tuple, Union

from requests import Response

API_KEY_PATTERN = re.compile(r"api_key=(.[^&]*)")
KEY_VALUE_GROUP = 1
MIN_KEY_LENGTH_TO_REVEAL_PREFIX = 8


def api_key_safe_raise_for_status(response: Response) -> None:
    """Raise an exception if the request is not successful.

    Args:
        response: The response of the request.
    """
    request_is_successful = response.status_code < 400
    if request_is_successful:
        return None
    response.url = deduct_api_key_from_string(value=response.url)
    response.raise_for_status()


def deduct_api_key_from_string(value: str) -> str:
    """Deduct the API key from the string.

    Args:
        value: The string to deduct the API key from.

    Returns:
        The string with the API key deducted.
    """
    return API_KEY_PATTERN.sub(deduct_api_key, value)


def deduct_api_key(match: re.Match) -> str:
    """Deduct the API key from the string.

    Args:
        match: The match of the API key.

    Returns:
        The string with the API key deducted.
    """
    key_value = match.group(KEY_VALUE_GROUP)
    if len(key_value) < MIN_KEY_LENGTH_TO_REVEAL_PREFIX:
        return f"api_key=***"
    key_prefix = key_value[:2]
    key_postfix = key_value[-2:]
    return f"api_key={key_prefix}***{key_postfix}"


def inject_images_into_payload(
    payload: dict,
    encoded_images: List[Tuple[str, Optional[float]]],
    key: str = "image",
) -> dict:
    """Inject images into the payload.

    Args:
        payload: The payload to inject the images into.
        encoded_images: The encoded images.
        key: The key of the images.

    Returns:
        The payload with the images injected.
    """
    if len(encoded_images) == 0:
        return payload
    if len(encoded_images) > 1:
        images_payload = [
            {"type": "base64", "value": image} for image, _ in encoded_images
        ]
        payload[key] = images_payload
    else:
        payload[key] = {"type": "base64", "value": encoded_images[0][0]}
    return payload


def inject_nested_batches_of_images_into_payload(
    payload: dict,
    encoded_images: Union[list, Tuple[str, Optional[float]]],
    key: str = "image",
) -> dict:
    """Inject nested batches of images into the payload.

    Args:
        payload: The payload to inject the images into.
        encoded_images: The encoded images.
        key: The key of the images.

    Returns:
        The payload with the images injected.
    """
    payload_value = _batch_of_images_into_inference_format(
        encoded_images=encoded_images,
    )
    payload[key] = payload_value
    return payload


def _batch_of_images_into_inference_format(
    encoded_images: Union[list, Tuple[str, Optional[float]]],
) -> Union[dict, list]:
    """Batch of images into inference format.

    Args:
        encoded_images: The encoded images.

    Returns:
        The images in inference format.
    """
    if not isinstance(encoded_images, list):
        return {"type": "base64", "value": encoded_images[0]}
    result = []
    for element in encoded_images:
        result.append(
            _batch_of_images_into_inference_format(
                encoded_images=element,
            )
        )
    return result
