"""Framework types — handler description, request params, hooks.

Hot-path types use `slots` (not frozen) to keep alloc cheap.
`ModelHandlerDescription` is frozen because it lives in the registry —
built once at import, zero per-request cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from fastapi import Request, Response

from inference_server.proxies.base import ModelManagerProxy


@dataclass(slots=True)
class CommonRequestParams:
    """Common query params shared by all model endpoints.

    Decoded once per request from query string + headers; passed through
    the dispatcher to the registry-keyed (model_type, task, op) lookup.
    """

    model_id: str
    api_key: str
    op: str = "infer"
    response_style: str = "compact"
    model_package_id: Optional[str] = None
    instance: str = ""
    device: str = ""
    task: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class ServerHooks:
    """Per-request side-channel for L1 handlers.

    Carries `Request` so handlers can pass it to
    `ModelManagerProxy.infer` for client-disconnect race detection.
    Additions (timing, tracing) live here without changing the handler
    signature.
    """

    request: Optional[Request] = None


@dataclass(frozen=True)
class ModelInterfaceDescription:
    """Per-(model_type, task, op) interface description.

    Reported by `interface_provider()` for `/v2/models/interface`. Static
    per registry entry — built at import time, never per-request.
    """

    task: str
    params: dict          # name → {"type", "required", "default", ...}
    output_schema: dict


@dataclass(frozen=True)
class ModelHandlerDescription:
    """One registry entry: input parser + handler + output serializer + interface.

    Keyed by `(model_type, task_type, op)` in `framework/registry.py`.
    Frozen because it never appears on the per-request path.
    """

    input_parser: Callable[[Request, CommonRequestParams], Awaitable[dict]]
    handler: Callable[
        [str, dict, ModelManagerProxy, ServerHooks], Awaitable[Any]
    ]
    output_serializer: Callable[[Any, CommonRequestParams], Response]
    interface_provider: Callable[[], ModelInterfaceDescription]
