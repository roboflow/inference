import base64
from typing import Any, Dict, List, Optional, Union

import orjson
import supervision as sv
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from inference.core.entities.responses.inference import InferenceResponse
from inference.core.utils.function import deprecated
from inference.core.utils.image_utils import ImageType
from inference.core.workflows.core_steps.common.serializers import (
    serialise_image,
    serialise_sv_detections,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


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
        content = [r.model_dump(by_alias=True, exclude_none=True) for r in response]
    else:
        content = response.model_dump(by_alias=True, exclude_none=True)
    return ORJSONResponseBytes(content=content)


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
        if isinstance(value, WorkflowImageData):
            value = serialise_image(image=value)
        elif isinstance(value, dict):
            value = serialise_dict(elements=value)
        elif isinstance(value, list):
            value = serialise_list(elements=value)
        elif isinstance(value, sv.Detections):
            value = serialise_sv_detections(detections=value)
        serialised_result[key] = value
    return serialised_result


def serialise_list(elements: List[Any]) -> List[Any]:
    result = []
    for element in elements:
        if isinstance(element, WorkflowImageData):
            element = serialise_image(image=element)
        elif isinstance(element, dict):
            element = serialise_dict(elements=element)
        elif isinstance(element, list):
            element = serialise_list(elements=element)
        elif isinstance(element, sv.Detections):
            element = serialise_sv_detections(detections=element)
        result.append(element)
    return result


def serialise_dict(elements: Dict[str, Any]) -> Dict[str, Any]:
    serialised_result = {}
    for key, value in elements.items():
        if isinstance(value, WorkflowImageData):
            value = serialise_image(image=value)
        elif isinstance(value, dict):
            value = serialise_dict(elements=value)
        elif isinstance(value, list):
            value = serialise_list(elements=value)
        elif isinstance(value, sv.Detections):
            value = serialise_sv_detections(detections=value)
        serialised_result[key] = value
    return serialised_result


@deprecated(
    reason="Function contains_image(...) will be removed from `inference` end of Q3 2024"
)
def contains_image(element: Any) -> bool:
    return (
        isinstance(element, dict)
        and element.get("type") == ImageType.NUMPY_OBJECT.value
    )
