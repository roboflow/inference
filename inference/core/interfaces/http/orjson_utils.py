import base64
from typing import Any, Dict, List, Optional, Union

import orjson
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from inference.core.entities.responses.inference import InferenceResponse
from inference.core.utils.image_utils import ImageType, encode_image_to_jpeg_bytes


class ORJSONResponseBytes(ORJSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(
            content,
            default=default,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )


JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]


def default(obj: Any) -> JSON:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    return obj


def orjson_response(
    response: Union[List[InferenceResponse], InferenceResponse, BaseModel]
) -> ORJSONResponseBytes:
    if isinstance(response, list):
        content = [r.dict(by_alias=True, exclude_none=True) for r in response]
    else:
        content = response.dict(by_alias=True, exclude_none=True)
    return ORJSONResponseBytes(content=content)


def serialise_workflow_result(
    result: Dict[str, Any],
    excluded_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if excluded_fields is None:
        excluded_fields = []
    excluded_fields = set(excluded_fields)
    serialised_result = {}
    for key, value in result.items():
        if key in excluded_fields:
            continue
        if contains_image(element=value):
            value = serialise_image(image=value)
        elif issubclass(type(value), dict):
            value = serialise_dict(elements=value)
        elif issubclass(type(value), list):
            value = serialise_list(elements=value)
        serialised_result[key] = value
    return serialised_result


def serialise_list(elements: List[Any]) -> List[Any]:
    result = []
    for element in elements:
        if contains_image(element=element):
            element = serialise_image(image=element)
        elif issubclass(type(element), dict):
            element = serialise_dict(elements=element)
        elif issubclass(type(element), list):
            element = serialise_list(elements=element)
        result.append(element)
    return result


def serialise_dict(elements: Dict[str, Any]) -> Dict[str, Any]:
    serialised_result = {}
    for key, value in elements.items():
        if contains_image(element=value):
            value = serialise_image(image=value)
        elif issubclass(type(value), dict):
            value = serialise_dict(elements=value)
        elif issubclass(type(value), list):
            value = serialise_list(elements=value)
        serialised_result[key] = value
    return serialised_result


def contains_image(element: Any) -> bool:
    return (
        issubclass(type(element), dict)
        and element.get("type") == ImageType.NUMPY_OBJECT.value
    )


def serialise_image(image: Dict[str, Any]) -> Dict[str, Any]:
    image["type"] = "base64"
    image["value"] = base64.b64encode(
        encode_image_to_jpeg_bytes(image["value"])
    ).decode("ascii")
    return image
