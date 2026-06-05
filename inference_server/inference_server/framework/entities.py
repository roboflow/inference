from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from fastapi import Request, Response

from inference_server.proxies.base import ModelManagerProxy


@dataclass(slots=True)
class CommonRequestParams:
    model_id: str
    api_key: str
    action: str = "infer"
    response_style: str = "compact"
    model_package_id: Optional[str] = None
    instance: str = ""
    device: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class ServerHooks:
    request: Optional[Request] = None
    common: Optional[CommonRequestParams] = None


class InputParseError(Exception):
    def __init__(self, response: Response):
        self.response = response
        super().__init__(f"input parse error: status={response.status_code}")


@dataclass(frozen=True)
class ModelInterfaceDescription:
    task: str
    params: dict
    output_schema: dict


@dataclass(frozen=True)
class ModelHandlerDescription:
    input_parser: Callable[[Request, CommonRequestParams], Awaitable[dict]]
    handler: Callable[[str, dict, ModelManagerProxy, ServerHooks], Awaitable[Any]]
    output_serializer: Callable[[Any, CommonRequestParams], Response]
    interface_provider: Callable[[], ModelInterfaceDescription]
