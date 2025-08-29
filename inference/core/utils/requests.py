import re

import aiohttp
from requests import Response

API_KEY_PATTERN = re.compile(r"api_key=(.[^&]*)")
KEY_VALUE_GROUP = 1
MIN_KEY_LENGTH_TO_REVEAL_PREFIX = 8


def api_key_safe_raise_for_status(response: Response) -> None:
    request_is_successful = response.status_code < 400
    if request_is_successful:
        return None
    response.url = API_KEY_PATTERN.sub(deduct_api_key, response.url)
    response.raise_for_status()


def api_key_safe_raise_for_status_aiohttp(response: aiohttp.ClientResponse) -> None:
    request_is_successful = response.status < 400
    if request_is_successful:
        return None
    response.request_info.real_url._query = API_KEY_PATTERN.sub(
        deduct_api_key,
        response.request_info.real_url._query,
    )
    response.raise_for_status()


def deduct_api_key(match: re.Match) -> str:
    key_value = match.group(KEY_VALUE_GROUP)
    if len(key_value) < MIN_KEY_LENGTH_TO_REVEAL_PREFIX:
        return f"api_key=***"
    key_prefix = key_value[:2]
    key_postfix = key_value[-2:]
    return f"api_key={key_prefix}***{key_postfix}"
