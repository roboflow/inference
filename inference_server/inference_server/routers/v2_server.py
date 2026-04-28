"""v2 server status endpoints — /v2/server/*"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, Response

from inference_server import state

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
        return Response(
            status_code=503,
            content=b'{"ready":false,"reason":"stats_unavailable"}',
            media_type="application/json",
        )

    preload_raw = os.environ.get("INFERENCE_PRELOAD_MODELS", "").strip()
    if preload_raw:
        preload_ids = {m.strip() for m in preload_raw.split(",") if m.strip()}
        models = stats.get("models", {})
        for mid in preload_ids:
            m = models.get(mid, {})
            if m.get("state") != "loaded":
                return Response(
                    status_code=503,
                    content=json.dumps({"ready": False, "reason": f"model {mid} not ready", "state": m.get("state")}).encode(),
                    media_type="application/json",
                )

    return Response(content=b'{"ready":true}', media_type="application/json")


@router.get("/info")
async def v2_info() -> Response:
    """Server information — version, loaded model count, capabilities."""
    try:
        stats = await state.fetch_stats(timeout_s=3.0)
    except (asyncio.TimeoutError, Exception):
        stats = {}

    models = stats.get("models", {})
    info = {
        "server": "inference-server",
        "models_loaded": len(models),
        "models": {mid: {"state": m.get("state"), "device": m.get("device")} for mid, m in models.items()},
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
        return Response(status_code=503, content=b'{"error":"stats_unavailable"}', media_type="application/json")

    return Response(content=json.dumps(stats).encode(), media_type="application/json")
