"""v2 server status endpoints — /v2/server/*"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, Depends, Response

from inference_server import configuration
from inference_server.dependencies import get_model_manager
from inference_server.errors import error_response
from inference_server.proxies.base import ModelManagerProxy

router = APIRouter(prefix="/v2/server")


@router.get("/health")
async def v2_health() -> Response:
    """Basic liveness check. No auth required."""
    return Response(content=b'{"status":"ok"}', media_type="application/json")


@router.get("/ready")
async def v2_ready(
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Readiness check — all preloaded models loaded and healthy."""
    try:
        stats = await mm.stats()
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    preload_raw = os.environ.get(configuration.INFERENCE_PRELOAD_MODELS_ENV, "").strip()
    if preload_raw:
        preload_ids = {m.strip() for m in preload_raw.split(",") if m.strip()}
        models = stats.get("mmp_models", {})
        for mid in preload_ids:
            m = models.get(mid, {})
            if m.get("state") != "loaded":
                return error_response(
                    503,
                    "MODEL_NOT_READY",
                    f"model {mid} not ready",
                    follow_up="wait for model to finish loading",
                )

    return Response(content=b'{"ready":true}', media_type="application/json")


@router.get("/info")
async def v2_info(
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """Server information — version, loaded model count, capabilities."""
    try:
        stats = await mm.stats()
    except Exception:
        stats = {}

    models = stats.get("mmp_models", {})
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
    mm: ModelManagerProxy = Depends(get_model_manager),
) -> Response:
    """JSON metrics from MMP stats snapshot.

    TODO: Phase 32f — add Prometheus text format option via Accept header.
    """
    try:
        stats = await mm.stats()
    except Exception:
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    return Response(content=json.dumps(stats).encode(), media_type="application/json")
