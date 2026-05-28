"""v2 model endpoints — /v2/models/*"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request, Response

from inference_server.dependencies import get_model_manager
from inference_server.errors import error_response
from inference_server.framework.dispatch import handle_model_inference_request
from inference_server.proxies.base import ModelManagerProxy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/models")

_V2_TODO = Response(
    status_code=501,
    content=b'{"error_code":"NOT_IMPLEMENTED","description":"endpoint not yet implemented"}',
    media_type="application/json",
)


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else ""


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
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Discover model interface — supported tasks, params, response types."""
    model_id = request.query_params.get("model_id", "")
    if not model_id:
        return error_response(400, "MISSING_PARAM", "model_id query param required")

    try:
        info = await mm.interface(model_id)
    except RuntimeError as exc:
        return error_response(
            404,
            "MODEL_NOT_LOADED",
            str(exc),
            follow_up="load the model first via POST /v2/models/load",
        )
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    return Response(
        content=json.dumps(info).encode(),
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
