"""L2 dispatcher — fail-fast pipeline for ``POST /v2/models/infer``.

Each step rejects with 400/401/404/501 before the next does expensive
work. Order matters: cheap query parse + auth + registry lookup +
param validation all run before body read, model load, slot alloc, or
inference itself.

Pipeline:

    1. ``decode_common_request_params``      — cheap query parse
    2. ``stat_model_while_checking_auth``    — 401/404, returns
                                               (model_type, action_default)
    3. resolve effective action              — client ``?action=``
                                               overrides API default
    4. registry lookup                       — 400 INVALID_ACTION or
                                               501 NOT_IMPLEMENTED
    5. ``_validate_action_params``           — 400 on required-missing /
                                               wrong-type (no body read yet)
    6. ``description.input_parser``          — body read; heavy
    7. ``proxy.ensure_loaded``               — model load; heavy
    8. ``description.handler`` → proxy.infer — inference; heavy
    9. ``description.output_serializer``     — typed → ``Response``
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import Request, Response
from starlette.requests import ClientDisconnect

from inference_server.errors import error_response
from inference_server.framework.entities import (
    CommonRequestParams,
    ServerHooks,
)
from inference_server.framework.registry import (
    DYNAMIC_MODELS_HANDLERS,
    has_handler_for_model_type,
    supported_actions_for,
)
from inference_server.proxies.base import (
    ClientDisconnected,
    ModelManagerProxy,
)

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


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


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


async def stat_model_while_checking_auth(
    common_params: CommonRequestParams,
) -> tuple[str, str]:
    """Resolve ``(model_type, action_default)`` for ``model_id`` while
    verifying ``api_key`` access.

    Phase B step 2 implements this with a TTL-LRU cache plus
    ``inference_models.weights_providers.roboflow.get_one_page_of_model_metadata``
    fallback. Until then every dispatch path that reaches here fails
    with 501.

    Raises:
        PermissionError — 401 (api key rejected).
        LookupError     — 404 (model id unknown / no access).
        RuntimeError    — 503 (upstream registry unreachable).
    """
    raise NotImplementedError("step 2: cache + Roboflow API fallback")


def _validate_action_params(
    params_spec: dict, query_extra: dict[str, str]
) -> Response | None:
    """Match ``query_extra`` against the handler's param spec.

    ``params_spec`` shape mirrors ``_TASK_CONFIGS[model_class][i][3]``
    in ``inference_model_manager.registry_defaults``: ``{name: {
    "type": "float"|"int"|"str"|"bool", "required": bool,
    "default": Any, ...}}``.

    Returns ``None`` on success, an error ``Response`` otherwise. No
    body read, no model load — runs before any heavy step so 400s
    stay cheap.
    """
    for name, spec in params_spec.items():
        if name not in query_extra:
            if spec.get("required"):
                return error_response(
                    400,
                    "MISSING_PARAM",
                    f"required query param missing: {name}",
                )
            continue
        raw = query_extra[name]
        type_name = spec.get("type", "str")
        try:
            _coerce_param(raw, type_name)
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


async def handle_model_inference_request(
    request: Request, proxy: ModelManagerProxy
) -> Response:
    common = decode_common_request_params(request)
    if not common.model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")
    if common.response_style not in ("compact", "rich"):
        return error_response(
            400, "INVALID_PARAM", "response_style must be 'compact' or 'rich'"
        )

    try:
        model_type, action_default = await stat_model_while_checking_auth(common)
    except PermissionError as exc:
        return error_response(401, "UNAUTHORIZED", str(exc) or "invalid api key")
    except LookupError as exc:
        return error_response(404, "MODEL_NOT_FOUND", str(exc) or "unknown model_id")
    except NotImplementedError as exc:
        return error_response(501, "NOT_IMPLEMENTED", str(exc))
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
        return error_response(
            501,
            "NOT_IMPLEMENTED",
            f"no handler registered for model_type={model_type!r}",
        )

    params_spec = description.interface_provider().params
    err = _validate_action_params(params_spec, common.extra)
    if err is not None:
        return err

    server_hooks = ServerHooks(request=request)

    try:
        input_data = await description.input_parser(request, common)
    except ClientDisconnect:
        logger.debug("[dispatch] client disconnected during body read")
        return Response(status_code=499)

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
        prediction = await description.handler(
            action, input_data, proxy, server_hooks
        )
    except ValueError as exc:
        return error_response(413, "PAYLOAD_TOO_LARGE", str(exc))
    except asyncio.TimeoutError:
        return error_response(504, "TIMEOUT", "inference timeout")
    except ClientDisconnected:
        return Response(status_code=499)
    except RuntimeError as exc:
        msg = str(exc) or "inference failed"
        low = msg.lower()
        if "no slots" in low or "alloc" in low:
            return error_response(
                503,
                "SERVER_BUSY",
                "no slots available, try again",
                follow_up="retry in 1s",
            )
        return error_response(500, "INFERENCE_FAILED", msg)

    return description.output_serializer(prediction, common)
