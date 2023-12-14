import base64
from typing import Any, List, Union

import orjson
from fastapi.responses import ORJSONResponse

from inference.core.entities.responses.inference import InferenceResponse


class ORJSONResponseBytes(ORJSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(
            content,
            default=default,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )


def default(obj):
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    return obj


def orjson_response(response: Union[List[InferenceResponse], InferenceResponse]):
    if isinstance(response, list):
        content = [r.dict(by_alias=True) for r in response]
    else:
        content = response.dict(by_alias=True)
    return ORJSONResponseBytes(content=content)
