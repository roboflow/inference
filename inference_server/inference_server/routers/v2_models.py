"""v2 model endpoints — /v2/models/*"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request, Response

from inference_server.auth import extract_bearer
from inference_server.dependencies import get_model_manager
from inference_server.errors import error_response
from inference_server.framework.dispatch import handle_model_inference_request
from inference_server.framework.entities import CommonRequestParams
from inference_server.framework.model_stat import stat_model_while_checking_auth
from inference_server.framework.registry import (
    DYNAMIC_MODELS_HANDLERS,
    supported_actions_for,
)
from inference_server.proxies.base import ModelManagerProxy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/models")

_V2_TODO = Response(
    status_code=501,
    content=b'{"error_code":"NOT_IMPLEMENTED","description":"endpoint not yet implemented"}',
    media_type="application/json",
)


def _bearer_token(request: Request) -> str:
    return extract_bearer(request.headers.get("authorization", ""))


# ---------------------------------------------------------------------------
# POST /v2/models/infer
# ---------------------------------------------------------------------------


@router.post("/infer")
async def v2_infer(
    request: Request,
    api_key: str = Depends(_bearer_token),
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """v2 model inference — structured JSON response with typed predictions."""
    dispatched = await handle_model_inference_request(request, mm)
    if dispatched is not None:
        return dispatched
    model_type = request.query_params.get("model_id", "")
    return error_response(
        501,
        "NOT_IMPLEMENTED",
        f"no handler registered for resolved model_type of model_id={model_type!r}",
    )


# ---------------------------------------------------------------------------
# GET /v2/models/interface
# ---------------------------------------------------------------------------


@router.get("/interface")
async def v2_model_interface(
    request: Request,
    api_key: str = Depends(_bearer_token),
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Discover model interface — supported actions, params, output schemas.

    Resolution order:
      1. Ask the proxy for the loaded-model interface. Cheap, in-memory.
      2. If the model is not loaded, resolve its task_type via the
         Roboflow registry (TTL-cached) and serve the static handler
         interface for that task_type — no load required.
    """
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        info = await mm.interface(model_id)
        return Response(
            content=json.dumps(info).encode(),
            media_type="application/json",
        )
    except RuntimeError:
        pass
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    common = CommonRequestParams(model_id=model_id, api_key=api_key)
    registry_response = await _interface_from_registry(common)
    if registry_response is not None:
        return registry_response

    return error_response(
        404,
        "MODEL_NOT_LOADED",
        f"model {model_id!r} not loaded and no static interface registered",
        follow_up="load the model first via POST /v2/models/load",
    )


async def _interface_from_registry(
    common: CommonRequestParams,
) -> Response | None:
    try:
        model_type, _action_default = await stat_model_while_checking_auth(common)
    except PermissionError as exc:
        return error_response(401, "UNAUTHORIZED", str(exc) or "invalid api key")
    except (LookupError, RuntimeError):
        return None

    actions = supported_actions_for(model_type)
    if not actions:
        return None

    actions_payload: dict[str, dict] = {}
    for action in actions:
        desc = DYNAMIC_MODELS_HANDLERS.get((model_type, action))
        if desc is None:
            continue
        interface = desc.interface_provider()
        actions_payload[action] = {
            "task": interface.task,
            "params": interface.params,
            "output_schema": interface.output_schema,
        }

    body = {
        "model_id": common.model_id,
        "model_type": model_type,
        "actions": actions_payload,
    }
    return Response(
        content=json.dumps(body).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# GET /v2/models/compatibility
# ---------------------------------------------------------------------------


@router.get("/compatibility")
async def v2_model_compatibility() -> Response:
    """Discover models compatible with current server configuration.

    TODO: Query model registry for packages matching server runtime.
    """
    return _V2_TODO


# ---------------------------------------------------------------------------
# GET /v2/models (list)
# ---------------------------------------------------------------------------


@router.get("")
async def v2_list_models(
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """List currently loaded models with state, device, memory, queue depth."""
    try:
        stats = await mm.stats()
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    models = stats.get("mmp_models", {})
    return Response(
        content=json.dumps({"models": models}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# POST /v2/models/load
# ---------------------------------------------------------------------------


@router.post("/load")
async def v2_load_model(
    request: Request,
    api_key: str = Depends(_bearer_token),
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Load specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        result = await mm.load(model_id, api_key)
    except asyncio.TimeoutError:
        return error_response(504, "TIMEOUT", "load request timeout")
    except Exception:
        logger.exception("v2 load proxy error for '%s'", model_id)
        return error_response(503, "PROXY_ERROR", "model manager unreachable")

    if result[0] != "ok":
        return error_response(500, "LOAD_FAILED", "model load failed")
    return Response(
        content=json.dumps({"model_id": model_id, "status": "loaded"}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# POST /v2/models/unload
# ---------------------------------------------------------------------------


@router.post("/unload")
async def v2_unload_model(
    request: Request,
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Unload specified model."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        result = await mm.unload(model_id)
    except asyncio.TimeoutError:
        return error_response(504, "TIMEOUT", "unload request timeout")
    except Exception:
        logger.exception("v2 unload proxy error for '%s'", model_id)
        return error_response(503, "PROXY_ERROR", "model manager unreachable")

    if result[0] != "ok":
        return error_response(500, "UNLOAD_FAILED", "model unload failed")
    return Response(
        content=json.dumps({"model_id": model_id, "status": "unloaded"}).encode(),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# DELETE /v2/models (unload all)
# ---------------------------------------------------------------------------


@router.delete("")
async def v2_unload_all(
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Unload all models."""
    try:
        stats = await mm.stats()
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    models = stats.get("mmp_models", {})
    errors = []
    for model_id in list(models.keys()):
        try:
            result = await mm.unload(model_id)
            if result[0] != "ok":
                errors.append(model_id)
        except asyncio.TimeoutError:
            errors.append(model_id)

    if errors:
        return Response(
            content=json.dumps({"status": "partial", "failed": errors}).encode(),
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"status": "all_unloaded", "count": len(models)}).encode(),
        media_type="application/json",
    )
