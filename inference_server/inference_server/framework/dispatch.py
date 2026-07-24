from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import Request, Response
from starlette.requests import ClientDisconnect

from inference_models.errors import ModelInputError
from inference_server import configuration
from inference_server.auth import extract_bearer
from inference_server.errors import (
    PayloadTooLargeError,
    ServerBusyError,
    error_response,
)
from inference_server.framework.entities import (
    CommonRequestParams,
    InputParseError,
    ServerHooks,
)
from inference_server.framework.model_stat import stat_model_while_checking_auth
from inference_server.framework.registry import (
    DYNAMIC_MODELS_HANDLERS,
    has_handler_for_model_type,
    supported_actions_for,
)
from inference_server.proxies.base import ClientDisconnected, ModelManagerProxy

logger = logging.getLogger(__name__)


_RESERVED_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "model_id",
        "model_package_id",
        "action",
        "style",
        "response_style",
        "instance",
        "device",
        "image",
    }
)

_COERCIBLE_QUERY_TYPES: frozenset[str] = frozenset({"str", "int", "float", "bool"})


def _bearer_token(request: Request) -> str:
    return extract_bearer(request.headers.get("authorization", ""))


def decode_common_request_params(request: Request) -> CommonRequestParams:
    qp = request.query_params
    extra: dict[str, str] = {
        k: v for k, v in qp.items() if k not in _RESERVED_QUERY_KEYS
    }
    response_style = qp.get("response_style") or qp.get("style") or "compact"
    return CommonRequestParams(
        model_id=qp.get("model_id", ""),
        api_key=_bearer_token(request),
        action=qp.get("action") or "",
        response_style=response_style,
        model_package_id=qp.get("model_package_id") or None,
        instance=qp.get("instance", ""),
        device=qp.get("device", ""),
        extra=extra,
    )


def _apply_param_defaults(params_spec: dict, params: dict) -> None:
    """Fill params absent from the request with interface-declared defaults."""
    for name, spec in params_spec.items():
        if name not in params and "default" in spec:
            params[name] = spec["default"]


def _validate_action_params(params_spec: dict, params: dict) -> Response | None:
    for name, spec in params_spec.items():
        type_name = spec.get("type", "str")
        if type_name not in _COERCIBLE_QUERY_TYPES:
            continue
        if name not in params:
            if spec.get("required"):
                return error_response(
                    400,
                    "MISSING_PARAM",
                    f"required param missing: {name}",
                )
            continue
        value = params[name]
        if isinstance(value, str):
            try:
                params[name] = _coerce_param(value, type_name)
            except ValueError as exc:
                return error_response(
                    400,
                    "INVALID_PARAM",
                    f"param {name!r}: {exc}",
                )
    return None


def _coerce_param(raw: str, type_name: str) -> Any:
    if type_name == "str":
        return raw
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    if type_name == "bool":
        low = raw.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"expected bool, got {raw!r}")
    raise ValueError(f"unknown type {type_name!r}")


def _cap_request_body(request: Request, max_bytes: int) -> Request:
    inner = request.receive
    total = 0

    async def receive():
        nonlocal total
        message = await inner()
        if message["type"] == "http.request":
            total += len(message.get("body", b""))
            if total > max_bytes:
                raise PayloadTooLargeError(
                    f"request body exceeds {max_bytes} byte limit"
                )
        return message

    return Request(request.scope, receive)


async def handle_model_inference_request(
    request: Request, proxy: ModelManagerProxy
) -> Response | None:
    common = decode_common_request_params(request)
    if not common.model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")
    if common.response_style not in ("compact", "rich"):
        return error_response(
            400, "INVALID_PARAM", "response_style must be 'compact' or 'rich'"
        )

    max_body = configuration.MAX_BODY_BYTES
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_body:
        return error_response(
            413,
            "PAYLOAD_TOO_LARGE",
            f"request body exceeds {max_body} byte limit",
        )
    request = _cap_request_body(request, max_body)

    try:
        model_type, action_default = await stat_model_while_checking_auth(common)
    except PermissionError as exc:
        return error_response(401, "UNAUTHORIZED", str(exc) or "invalid api key")
    except LookupError as exc:
        return error_response(404, "MODEL_NOT_FOUND", str(exc) or "unknown model_id")
    except RuntimeError as exc:
        return error_response(
            503,
            "REGISTRY_UNAVAILABLE",
            str(exc) or "model registry unreachable",
            follow_up="retry shortly",
        )

    action = common.action or action_default
    common.action = action

    description = DYNAMIC_MODELS_HANDLERS.get((model_type, action))
    if description is None:
        if has_handler_for_model_type(model_type):
            return error_response(
                400,
                "INVALID_ACTION",
                f"action={action!r} not supported for model_type={model_type!r}",
                follow_up=f"supported actions: {supported_actions_for(model_type)}",
            )
        return None

    server_hooks = ServerHooks(request=request, common=common)

    try:
        input_data = await description.input_parser(request, common)
    except InputParseError as exc:
        return exc.response
    except PayloadTooLargeError as exc:
        return error_response(413, "PAYLOAD_TOO_LARGE", str(exc))
    except ClientDisconnect:
        logger.debug("[dispatch] client disconnected during body read")
        return Response(status_code=499)

    params_spec = description.interface_provider().params
    err = _validate_action_params(params_spec, input_data.setdefault("params", {}))
    if err is not None:
        return err
    _apply_param_defaults(params_spec, input_data["params"])

    status = await proxy.ensure_loaded(
        common.model_id, common.instance, common.api_key, common.device
    )
    if status[0] == "load_timeout":
        return error_response(
            503,
            "MODEL_LOADING",
            "model loading, try again shortly",
            follow_up="retry after Retry-After seconds",
            headers={"Retry-After": str(status[1])},
        )
    if status[0] == "error":
        return error_response(500, "LOAD_FAILED", "model load failed")

    try:
        prediction = await description.handler(action, input_data, proxy, server_hooks)
    except PayloadTooLargeError as exc:
        return error_response(413, "PAYLOAD_TOO_LARGE", str(exc))
    except ServerBusyError:
        return error_response(
            503,
            "SERVER_BUSY",
            "no capacity, try again",
            follow_up="retry in 1s",
            headers={"Retry-After": "1"},
        )
    except ValueError as exc:
        return error_response(400, "INVALID_PARAM", str(exc))
    except ModelInputError as exc:
        return error_response(400, "INVALID_PARAM", str(exc))
    except asyncio.TimeoutError:
        return error_response(504, "TIMEOUT", "inference timeout")
    except ClientDisconnected:
        return Response(status_code=499)
    except RuntimeError as exc:
        # Full error logged with a correlation id; clients get the id only —
        # backend error text can contain internal paths / stack fragments.
        ref = uuid.uuid4().hex[:8]
        logger.error("[dispatch] inference failed (ref %s): %s", ref, exc)
        return error_response(500, "INFERENCE_FAILED", f"inference failed (ref {ref})")

    return description.output_serializer(prediction, common)
