import base64
from typing import Any, Dict, List, Optional, Union

import orjson
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from inference.core.entities.responses.inference import InferenceResponse
from inference.core.utils.function import deprecated
from inference.core.utils.image_utils import ImageType
from inference.core.workflows.core_steps.common.serializers import (
    serialize_wildcard_kind,
)


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
    response: Union[List[InferenceResponse], InferenceResponse, BaseModel],
) -> ORJSONResponseBytes:
    if isinstance(response, list):
        content = [r.model_dump(by_alias=True, exclude_none=True) for r in response]
    else:
        content = response.model_dump(by_alias=True, exclude_none=True)
    return ORJSONResponseBytes(content=content)


@deprecated(
    reason="Function serialise_workflow_result(...) will be removed from `inference` end of Q1 2025. "
    "Workflows ecosystem shifted towards internal serialization - see Workflows docs: "
    "https://inference.roboflow.com/workflows/about/"
)
def serialise_workflow_result(
    result: List[Dict[str, Any]],
    excluded_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    return [
        serialise_single_workflow_result_element(
            result_element=result_element,
            excluded_fields=excluded_fields,
        )
        for result_element in result
    ]


@deprecated(
    reason="Function serialise_single_workflow_result_element(...) will be removed from `inference` end of Q1 2025. "
    "Workflows ecosystem shifted towards internal serialization - see Workflows docs: "
    "https://inference.roboflow.com/workflows/about/"
)
def serialise_single_workflow_result_element(
    result_element: Dict[str, Any],
    excluded_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if excluded_fields is None:
        excluded_fields = []
    excluded_fields = set(excluded_fields)
    serialised_result = {}
    for key, value in result_element.items():
        if key in excluded_fields:
            continue
        serialised_result[key] = serialize_wildcard_kind(value=value)
    return serialised_result


@deprecated(
    reason="Function contains_image(...) will be removed from `inference` end of Q3 2024"
)
def contains_image(element: Any) -> bool:
    return (
        isinstance(element, dict)
        and element.get("type") == ImageType.NUMPY_OBJECT.value
    )
