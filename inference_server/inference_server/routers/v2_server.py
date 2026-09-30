"""v2 server status endpoints — /v2/server/*"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from inference_server import configuration
from inference_server.dependencies import get_model_manager
from inference_server.errors import error_response

router = APIRouter(prefix="/v2/server")


@router.get("/health")
async def v2_health() -> Response:
    """Basic liveness check. No auth required."""
    return Response(content=b'{"status":"ok"}', media_type="application/json")


@router.get("/ready")
async def v2_ready(
    request: Request,
    mm: Any = Depends(get_model_manager),
) -> Response:
    """Readiness check — startup preload finished and model manager reachable."""
    try:
        stats = await mm.stats()
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    if not request.app.state.preload_finished:
        models = stats.get("models", {})
        startup_ids = (
            configuration.preload_model_ids() + configuration.pinned_model_ids()
        )
        pending = [
            mid
            for mid, _ in startup_ids
            if models.get(mid, {}).get("state") != "loaded"
        ]
        description = (
            f"model {pending[0]} not ready"
            if pending
            else "startup preload not finished"
        )
        return error_response(
            503,
            "MODEL_NOT_READY",
            description,
            follow_up="wait for model to finish loading",
        )

    return Response(content=b'{"ready":true}', media_type="application/json")


@router.get("/info")
async def v2_info(
    mm: Any = Depends(get_model_manager),
) -> Response:
    """Server information — version, loaded model count, capabilities."""
    try:
        stats = await mm.stats()
    except Exception:
        stats = {}

    models = stats.get("models", {})
    info = {
        "server": "inference-server",
        "models_loaded": len(models),
        "models": {
            mid: {"state": m.get("state"), "device": m.get("device")}
            for mid, m in models.items()
        },
    }
    return Response(content=json.dumps(info).encode(), media_type="application/json")


@router.get("/metrics")
async def v2_metrics(
    mm: Any = Depends(get_model_manager),
) -> Response:
    """JSON metrics from MMP stats snapshot.

    TODO: Phase 32f — add Prometheus text format option via Accept header.
    """
    try:
        stats = await mm.stats()
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    return Response(content=json.dumps(stats).encode(), media_type="application/json")
