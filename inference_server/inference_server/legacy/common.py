from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import ParseResult, SplitResult, urlparse, urlsplit

import numpy as np
import orjson
import requests
from fastapi import Request, Response
from inference_model_manager.backends.decode import decoded_dims
from PIL import Image
from pydantic import BaseModel

from inference_server import configuration
from inference_server.auth import extract_bearer
from inference_server.configuration import (
    ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM,
    ALLOW_URL_INPUT,
    DEFAULT_API_KEY,
    OFFLINE_MODE,
)
from inference_server.framework.input_parsers.image_limits import too_many_images
from inference_server.framework.input_parsers.url_fetch import fetch_images_from_urls
from inference_server.legacy.errors import LegacyHTTPError

_NPY_MAGIC = b"\x93NUMPY"
_BASE64_DATA_TYPE_PATTERN = re.compile(rb"^data:image/[a-zA-Z]+;base64,")
_IMAGE_ERROR = "Could not load valid image from request."
_IMAGE_LOAD_ERROR_PREFIX = "Could not load input image. Cause: "
_UNKNOWN_IMAGE_TYPE_ERROR = "Image declaration contains not recognised image type."
_LOCAL_FILE_DISABLED_ERROR = "Loading images from local filesystem is disabled."
_RAW_BYTES_ERROR = (
    "Invalid base64 input: the image payload contains raw bytes instead of a "
    "base64-encoded string. Please base64-encode the image before sending."
)
_MALFORMED_BASE64_ERROR = "Malformed base64 input image."
_EMPTY_PAYLOAD_ERROR = "Empty image payload."
_NUMPY_UNSUPPORTED_ERROR = (
    "NumPy image type is not supported in this configuration of `inference`."
)
_NOT_NDARRAY_ERROR = (
    "Data provided as input could not be decoded into np.ndarray object."
)
_NDARRAY_DIMENSIONS_ERROR = "For image given as np.ndarray expected 2 or 3 dimensions."
_URL_OFFLINE_ERROR = "Cannot load an image from URL while OFFLINE_MODE is enabled."
_URL_INPUT_DISABLED_ERROR = (
    "Providing images via URL is not supported in this configuration of `inference`."
)
_URL_INVALID_ERROR = "Provided image URL is invalid"
_URL_NON_HTTPS_ERROR = (
    "Providing images via non https:// URL is not supported in this configuration "
    "of `inference`."
)
_URL_WHITELIST_ERROR = (
    "It is not allowed to reach image URL - prohibited by whitelisted destinations."
)
_URL_BLACKLIST_ERROR = (
    "It is not allowed to reach image URL - prohibited by blacklisted destinations."
)
_URL_DESTINATION_ERROR = "URL points to a network destination that is not allowed."
_URL_FETCH_ERROR = "Data pointed by URL could not be decoded into image."
_URL_NOT_IMAGE_ERROR = "Data is not image."


@dataclass(slots=True)
class ImagePayload:
    data: Union[bytes, np.ndarray]
    width: int
    height: int


def resolve_api_key(
    request: Request, query_value: Optional[str], body_value: Optional[str]
) -> Optional[str]:
    if query_value:
        return query_value
    header_value = extract_bearer(request.headers.get("authorization", ""))
    if header_value:
        return header_value
    if body_value:
        return body_value
    return DEFAULT_API_KEY


def image_load_error(public_message: str) -> LegacyHTTPError:
    return LegacyHTTPError(400, f"{_IMAGE_LOAD_ERROR_PREFIX}{public_message}")


def image_dims(
    data: bytes, *, not_image_message: str = _URL_NOT_IMAGE_ERROR
) -> tuple[int, int]:
    if data[:6] == _NPY_MAGIC:
        buffer = io.BytesIO(data)
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(buffer)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(buffer)
        if len(shape) < 2:
            raise image_load_error(not_image_message)
        return int(shape[1]), int(shape[0])
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as error:
        raise image_load_error(not_image_message) from error
    # The decoder applies EXIF orientation; report the size of what it yields.
    width, height = decoded_dims(data, width, height)
    return int(width), int(height)


def decode_inline_image(image: Any, *, ndarray_ok: bool) -> ImagePayload:
    image_type = _image_attribute(image, "type")
    value = _image_attribute(image, "value")
    if image_type == "url":
        raise ValueError("url")
    if image_type == "numpy":
        raise image_load_error(_NUMPY_UNSUPPORTED_ERROR)
    if image_type == "numpy_object":
        return _numpy_object_payload(value, ndarray_ok=ndarray_ok)
    if image_type == "base64":
        data = _decode_base64(value)
        width, height = image_dims(data, not_image_message=_MALFORMED_BASE64_ERROR)
        return ImagePayload(data, width, height)
    if image_type == "file" and not ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM:
        raise image_load_error(_LOCAL_FILE_DISABLED_ERROR)
    raise image_load_error(_UNKNOWN_IMAGE_TYPE_ERROR)


async def load_request_images(images: list, *, ndarray_ok: bool) -> list[ImagePayload]:
    limit_error = too_many_images(len(images))
    if limit_error is not None:
        raise _error_from_response(limit_error)
    payloads: list[Optional[ImagePayload]] = [None] * len(images)
    url_positions: list[int] = []
    urls: list[str] = []
    for position, image in enumerate(images):
        if _image_attribute(image, "type") == "url":
            if OFFLINE_MODE or not ALLOW_URL_INPUT:
                raise image_load_error(_url_input_refused_message())
            url_positions.append(position)
            urls.append(_image_attribute(image, "value"))
        else:
            payloads[position] = decode_inline_image(image, ndarray_ok=ndarray_ok)
    if urls:
        unparsable = next(
            (index for index, url in enumerate(urls) if _split_url(url) is None),
            len(urls),
        )
        fetched, fetch_error = [], None
        if unparsable > 0:
            fetched, fetch_error = await fetch_images_from_urls(urls[:unparsable])
        if fetch_error is not None:
            raise _url_fetch_error(fetch_error)
        if unparsable < len(urls):
            raise image_load_error(_URL_INVALID_ERROR)

        for position, data in zip(url_positions, fetched):
            width, height = image_dims(data)
            payloads[position] = ImagePayload(data, width, height)
    return payloads


def as_image_list(value: Any) -> tuple[list, bool]:
    if isinstance(value, list):
        return value, True
    return [value], False


def orjson_response(
    obj: Union[BaseModel, list[BaseModel]], keep_parent_id: bool = False
) -> Response:
    if isinstance(obj, list):
        content: Any = [_dump_model(item, keep_parent_id) for item in obj]
    else:
        content = _dump_model(obj, keep_parent_id)
    return Response(
        content=orjson.dumps(
            content,
            default=_orjson_default,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        ),
        media_type="application/json",
    )


def _dump_model(model: BaseModel, keep_parent_id: bool) -> dict:
    dumped = model.model_dump(by_alias=True, exclude_none=True)
    if keep_parent_id and "parent_id" not in dumped:
        dumped["parent_id"] = None
    return dumped


def _orjson_default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    return obj


def _image_attribute(image: Any, name: str) -> Any:
    if isinstance(image, dict):
        return image.get(name)
    return getattr(image, name, None)


def _numpy_object_payload(value: Any, *, ndarray_ok: bool) -> ImagePayload:
    array = np.asarray(value)
    if array.ndim < 2:
        if not isinstance(value, np.ndarray):
            raise image_load_error(_NOT_NDARRAY_ERROR)
        raise image_load_error(_NDARRAY_DIMENSIONS_ERROR)
    width, height = int(array.shape[1]), int(array.shape[0])
    if ndarray_ok:
        return ImagePayload(value, width, height)
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return ImagePayload(buffer.getvalue(), width, height)


def _decode_base64(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        try:
            encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise image_load_error(_RAW_BYTES_ERROR) from error
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        raise image_load_error(_MALFORMED_BASE64_ERROR)
    encoded = _BASE64_DATA_TYPE_PATTERN.sub(b"", encoded)
    try:
        data = base64.b64decode(encoded, validate=False)
    except Exception as error:
        raise image_load_error(_MALFORMED_BASE64_ERROR) from error
    if not data:
        raise image_load_error(_EMPTY_PAYLOAD_ERROR)
    return data


def _url_input_refused_message() -> str:
    if OFFLINE_MODE:
        return _URL_OFFLINE_ERROR
    return _URL_INPUT_DISABLED_ERROR


def _split_url(url: str) -> Optional[SplitResult]:
    try:
        return urlsplit(url)
    except ValueError:
        return None


def _prepared_url(url: str) -> Optional[ParseResult]:
    try:
        if "\\" in urlparse(url).netloc:
            return None
        prepared_url = requests.Request(method="GET", url=url).prepare().url
        return urlparse(prepared_url)
    except (requests.exceptions.RequestException, ValueError):
        return None


def _invalid_url_message(url: str) -> str:
    if _prepared_url(url) is None:
        return _URL_INVALID_ERROR
    return _URL_NON_HTTPS_ERROR


def _forbidden_destination_message(url: str) -> str:
    parts = _prepared_url(url)
    if parts is None:
        return _URL_INVALID_ERROR
    host = (parts.hostname or "").lower()
    allowed = configuration.WHITELISTED_DESTINATIONS_FOR_URL_INPUT
    if allowed is not None and host not in allowed:
        return _URL_WHITELIST_ERROR
    blocked = configuration.BLACKLISTED_DESTINATIONS_FOR_URL_INPUT
    if blocked and host in blocked:
        return _URL_BLACKLIST_ERROR
    return _URL_DESTINATION_ERROR


def _url_fetch_error(response: Response) -> LegacyHTTPError:
    try:
        error_code = json.loads(response.body).get("error_code")
    except Exception:
        error_code = None
    failed_url = getattr(response, "failed_url", None)
    if failed_url is None and error_code in (
        "INVALID_URL",
        "URL_DESTINATION_FORBIDDEN",
    ):
        return image_load_error(_URL_FETCH_ERROR)
    if error_code == "URL_INPUT_DISABLED":
        return image_load_error(_url_input_refused_message())
    if error_code == "INVALID_URL":
        return image_load_error(_invalid_url_message(failed_url))
    if error_code == "URL_DESTINATION_FORBIDDEN":
        return image_load_error(_forbidden_destination_message(failed_url))
    if error_code in ("URL_FETCH_FAILED", "URL_FETCH_TIMEOUT"):
        return image_load_error(_URL_FETCH_ERROR)
    return _error_from_response(response)


def _error_from_response(response: Response) -> LegacyHTTPError:
    try:
        body = json.loads(response.body)
    except Exception:
        body = {}
    return LegacyHTTPError(
        response.status_code, body.get("description") or _IMAGE_ERROR
    )
