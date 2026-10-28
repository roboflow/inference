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
    ClientDisconnected,
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
from inference_server.middlewares.model_load import record_model_load
from inference_server.prometheus import measure_inference
from inference_server.routing import has_capability_marker, routing_key

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

_NON_MODEL_PARAMS: frozenset[str] = frozenset(
    {
        "self",
        "model",
        "model_id",
        "action",
        "raw_input",
        "validate",
        "serialize",
        "wire_marshalling",
        "timing",
        "api_key",
        "countinference",
        "service_secret",
        "stream_pipeline_context_id",
        "stream_pipeline_producer_id",
    }
)

_COERCIBLE_QUERY_TYPES: frozenset[str] = frozenset(
    {"str", "int", "float", "bool", "list", "list[str]", "confidence"}
)

_CONFIDENCE_MODES: tuple[str, ...] = ("best", "default")

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
}


def _bearer_token(request: Request) -> str:
    return extract_bearer(request.headers.get("authorization", ""))


def decode_common_request_params(request: Request) -> CommonRequestParams:
    qp = request.query_params
    extra: dict[str, Any] = {}
    for key in qp.keys():
        if key in _RESERVED_QUERY_KEYS:
            continue
        values = qp.getlist(key)
        extra[key] = values[0] if len(values) == 1 else values

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
        if name not in params:
            if spec.get("required") and type_name != "image":
                return error_response(
                    400,
                    "MISSING_PARAM",
                    f"required param missing: {name}",
                )
            continue
        if type_name not in _COERCIBLE_QUERY_TYPES:
            continue
        try:
            params[name] = _coerce_param(params[name], type_name)
        except ValueError as exc:
            logger.warning("Invalid param: %s", exc)
            expected = (
                "a number, 'best' or 'default'"
                if type_name == "confidence"
                else type_name
            )
            return error_response(
                400,
                "INVALID_PARAM",
                f"param {name!r} must be {expected}",
            )
    return None


def _coerce_param(value: Any, type_name: str) -> Any:
    if value is None:
        return value
    if type_name == "confidence":
        if isinstance(value, str) and value in _CONFIDENCE_MODES:
            return value
        type_name = "float"
    if type_name in ("list", "list[str]"):
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list):
            raise ValueError(f"expected {type_name}, got {type(value).__name__}")
        if type_name == "list[str]" and not all(isinstance(i, str) for i in items):
            raise ValueError("expected a list of strings")

        return items
    if isinstance(value, str):
        parsed = _parse_text_param(value, type_name)

        return parsed
    if not isinstance(value, _JSON_TYPES[type_name]) or (
        isinstance(value, bool) and type_name != "bool"
    ):
        raise ValueError(f"expected {type_name}, got {type(value).__name__}")
    if type_name == "float":
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError("value is out of float range") from exc

        return converted

    return value


def _parse_text_param(raw: str, type_name: str) -> Any:
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
    request: Request, proxy: Any
) -> Response | None:
    common = decode_common_request_params(request)
    if not common.model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")
    if common.response_style not in ("compact", "rich"):
        return error_response(
            400, "INVALID_PARAM", "response_style must be 'compact' or 'rich'"
        )
    if has_capability_marker(common.model_id):
        return error_response(404, "MODEL_NOT_FOUND", "unknown model_id")
    if has_capability_marker(common.instance):
        return error_response(400, "INVALID_PARAM", "param 'instance'")

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
        # The registry's wording can carry request details, so it is logged
        # rather than returned.
        logger.warning("Model access denied: %s", exc)
        return error_response(401, "UNAUTHORIZED", "invalid api key")
    except LookupError as exc:
        logger.warning("Model not found: %s", exc)
        return error_response(404, "MODEL_NOT_FOUND", "unknown model_id")
    except RuntimeError as exc:
        logger.warning("Runtime error when calling stat_model_while_checking_auth: %s", exc)
        return error_response(
            503,
            "REGISTRY_UNAVAILABLE",
            "model registry unreachable",
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
        logger.warning("Payload too large: %s", exc)
        return error_response(413, "PAYLOAD_TOO_LARGE", "payload too large")
    except ClientDisconnect:
        logger.debug("[dispatch] client disconnected during body read")
        return Response(status_code=499)

    params = input_data.setdefault("params", {})
    for name in _NON_MODEL_PARAMS:
        params.pop(name, None)
    params_spec = description.interface_provider().params
    err = _validate_action_params(params_spec, params)
    if err is not None:
        return err
    _apply_param_defaults(params_spec, input_data["params"])

    record_model_load(common.model_id, cold_start=False, load_time_s=0.0)
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
        with measure_inference(
            routing_key(common.model_id, common.instance),
            responses=max(len(input_data.get("images") or ()), 1),
        ):
            prediction = await description.handler(
                action, input_data, proxy, server_hooks
            )
    except PayloadTooLargeError as exc:
        logger.warning("Payload too large: %s", exc)
        return error_response(413, "PAYLOAD_TOO_LARGE", "payload too large")
    except ServerBusyError:
        return error_response(
            503,
            "SERVER_BUSY",
            "no capacity, try again",
            follow_up="retry in 1s",
            headers={"Retry-After": "1"},
        )
    except ValueError as exc:
        logger.warning("Invalid param: %s", exc)
        return error_response(400, "INVALID_PARAM", "invalid input")
    except ModelInputError as exc:
        logger.warning("Model input error: %s", exc)
        return error_response(400, "INVALID_PARAM", "invalid model input")
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
