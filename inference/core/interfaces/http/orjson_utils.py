import base64
from typing import Any, Dict, List, Optional, Union

import orjson
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from inference.core.entities.responses.inference import InferenceResponse
from inference.core.utils.function import deprecated
from inference.core.utils.image_utils import ImageType


def _resolve_wildcard_serializer():
    """Pick the wildcard serialiser at call time (not import time).

    The tensor-native path yields torch (often CUDA) tensors that must be moved to
    CPU during serialisation before results cross the stream-manager process
    boundary via a multiprocessing queue: pickling a live CUDA tensor relies on
    CUDA IPC, which is unsupported on Jetson/Tegra and fails with "CUDA error:
    invalid argument". The tensor serialiser calls .detach().cpu(); the numpy one
    passes tensor-native objects through untouched.

    Resolution is deferred to call time because a load-time ``if FLAG: import`` in
    this module binds unreliably depending on whether ``inference.core.env`` was
    imported before this module; by the time a workflow result is serialised the
    flag is settled, and sys.modules caches the import after the first call.
    """
    from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION

    if ENABLE_TENSOR_DATA_REPRESENTATION:
        from inference.core.workflows.core_steps.common.serializers_tensor import (
            serialize_wildcard_kind,
        )
    else:
        from inference.core.workflows.core_steps.common.serializers import (
            serialize_wildcard_kind,
        )
    return serialize_wildcard_kind


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


def orjson_response_keeping_parent_id(
    response: Union[List[InferenceResponse], InferenceResponse, BaseModel],
) -> ORJSONResponseBytes:
    if isinstance(response, list):
        content = []
        for r in response:
            serialised = r.model_dump(by_alias=True, exclude_none=True)
            if "parent_id" not in serialised:
                serialised["parent_id"] = None
            content.append(serialised)
    else:
        content = response.model_dump(by_alias=True, exclude_none=True)
        if "parent_id" not in content:
            content["parent_id"] = None
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
    serialize_wildcard_kind = _resolve_wildcard_serializer()
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
