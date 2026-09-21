from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import orjson
from fastapi import Request, Response
from PIL import Image
from pydantic import BaseModel

from inference_server.auth import extract_bearer
from inference_server.configuration import (
    ALLOW_URL_INPUT,
    DEFAULT_API_KEY,
    OFFLINE_MODE,
)
from inference_server.framework.input_parsers.image_limits import too_many_images
from inference_server.framework.input_parsers.url_fetch import fetch_images_from_urls
from inference_server.legacy.errors import LegacyHTTPError

_NPY_MAGIC = b"\x93NUMPY"
_BASE64_DATA_TYPE_PATTERN = re.compile(rb"^data:image/[a-zA-Z]+;base64,")
_BASE64_ERROR = "Could not load valid image from base64 string."
_IMAGE_ERROR = "Could not load valid image from request."
_URL_REFUSED_ERROR = "Loading images from URLs is not allowed on this server."
_PICKLED_NUMPY_ERROR = (
    "Loading images serialized with pickle is not supported - send the image as "
    "base64 instead."
)


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


def image_dims(data: bytes) -> tuple[int, int]:
    if data[:6] == _NPY_MAGIC:
        buffer = io.BytesIO(data)
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(buffer)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(buffer)
        if len(shape) < 2:
            raise LegacyHTTPError(400, _IMAGE_ERROR)
        return int(shape[1]), int(shape[0])
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as error:
        raise LegacyHTTPError(400, _IMAGE_ERROR) from error
    return int(width), int(height)


def decode_inline_image(image: Any, *, ndarray_ok: bool) -> ImagePayload:
    image_type = _image_attribute(image, "type")
    value = _image_attribute(image, "value")
    if image_type == "url":
        raise ValueError("url")
    if image_type == "numpy":
        raise LegacyHTTPError(501, _PICKLED_NUMPY_ERROR)
    if image_type == "numpy_object":
        return _numpy_object_payload(value, ndarray_ok=ndarray_ok)
    if image_type == "base64":
        data = _decode_base64(value)
        width, height = image_dims(data)
        return ImagePayload(data, width, height)
    raise LegacyHTTPError(400, f"Invalid image type: {image_type}")


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
                raise LegacyHTTPError(400, _URL_REFUSED_ERROR)
            url_positions.append(position)
            urls.append(_image_attribute(image, "value"))
        else:
            payloads[position] = decode_inline_image(image, ndarray_ok=ndarray_ok)
    if urls:
        fetched, fetch_error = await fetch_images_from_urls(urls)
        if fetch_error is not None:
            raise _error_from_response(fetch_error)
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
        raise LegacyHTTPError(400, _IMAGE_ERROR)
    width, height = int(array.shape[1]), int(array.shape[0])
    if ndarray_ok:
        return ImagePayload(value, width, height)
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return ImagePayload(buffer.getvalue(), width, height)


def _decode_base64(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        raise LegacyHTTPError(400, _BASE64_ERROR)
    encoded = _BASE64_DATA_TYPE_PATTERN.sub(b"", encoded)
    try:
        data = base64.b64decode(encoded, validate=False)
    except Exception as error:
        raise LegacyHTTPError(400, _BASE64_ERROR) from error
    if not data:
        raise LegacyHTTPError(400, _BASE64_ERROR)
    return data


def _error_from_response(response: Response) -> LegacyHTTPError:
    try:
        body = json.loads(response.body)
    except Exception:
        body = {}
    return LegacyHTTPError(
        response.status_code, body.get("description") or _IMAGE_ERROR
    )
