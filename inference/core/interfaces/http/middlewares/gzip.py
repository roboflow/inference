import gzip
from typing import TypeVar, Union

from fastapi import Request, Response
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def gzip_response_if_requested(
    request: Request,
    response: T,
) -> Union[Response, T]:
    if "gzip" not in request.headers.get("Accept-Encoding", ""):
        return response
    response = Response(
        content=response.json(),
    )
    response.body = gzip.compress(response.body)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"] = str(len(response.body))
    return response
