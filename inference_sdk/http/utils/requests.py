import re
from typing import List, Optional, Tuple

from requests import Response

API_KEY_PATTERN = re.compile(r"api_key=(.[^&]*)")
KEY_VALUE_GROUP = 1
MIN_KEY_LENGTH_TO_REVEAL_PREFIX = 8


def api_key_safe_raise_for_status(response: Response) -> None:
    request_is_successful = response.status_code < 400
    if request_is_successful:
        return None
    response.url = deduct_api_key_from_string(value=response.url)
    response.raise_for_status()


def deduct_api_key_from_string(value: str) -> str:
    return API_KEY_PATTERN.sub(deduct_api_key, value)


def deduct_api_key(match: re.Match) -> str:
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
