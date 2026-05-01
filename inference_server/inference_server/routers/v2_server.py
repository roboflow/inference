"""v2 server status endpoints — /v2/server/*"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, Response

from inference_server import state
from inference_server.errors import error_response

router = APIRouter(prefix="/v2/server")


@router.get("/health")
async def v2_health() -> Response:
    """Basic liveness check. No auth required."""
    return Response(content=b'{"status":"ok"}', media_type="application/json")


@router.get("/ready")
async def v2_ready() -> Response:
    """Readiness check — all preloaded models loaded and healthy."""
    try:
        stats = await state.fetch_stats(timeout_s=3.0)
    except (asyncio.TimeoutError, Exception):
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    preload_raw = os.environ.get("INFERENCE_PRELOAD_MODELS", "").strip()
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
async def v2_info() -> Response:
    """Server information — version, loaded model count, capabilities."""
    try:
        stats = await state.fetch_stats(timeout_s=3.0)
    except (asyncio.TimeoutError, Exception):
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
async def v2_metrics() -> Response:
    """JSON metrics from MMP stats snapshot.

    TODO: Phase 32f — add Prometheus text format option via Accept header.
    """
    try:
        stats = await state.fetch_stats(timeout_s=3.0)
    except (asyncio.TimeoutError, Exception):
        return error_response(503, "STATS_UNAVAILABLE", "could not reach model manager")

    return Response(content=json.dumps(stats).encode(), media_type="application/json")
