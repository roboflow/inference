"""Staging-selectable live-video ingest for the video processor.

The production-shaped default remains the POC's measured low-latency PyAV
producer.  ``gstreamer_cuda`` is deliberately fail-loud: it constructs the
v1.4 CUDA/GStreamer producer directly instead of using the generic discovery
path, whose best-effort fallback to CPU decode would invalidate an A/B test.
"""

import math
import os
import re


PYAV_INGEST = "pyav"
GSTREAMER_CUDA_INGEST = "gstreamer_cuda"
SUPPORTED_INGEST_MODES = (PYAV_INGEST, GSTREAMER_CUDA_INGEST)
_SAFE_STAT_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def resolve_video_ingest_mode(value=None):
    mode = (value or os.getenv("PROCESSOR_VIDEO_INGEST_MODE", PYAV_INGEST)).strip()
    mode = mode.lower()
    if mode not in SUPPORTED_INGEST_MODES:
        raise ValueError(
            "PROCESSOR_VIDEO_INGEST_MODE must be one of "
            + ", ".join(SUPPORTED_INGEST_MODES)
        )
    if mode == GSTREAMER_CUDA_INGEST and not _env_flag(
        "ENABLE_TENSOR_DATA_REPRESENTATION"
    ):
        raise ValueError(
            "gstreamer_cuda ingest requires "
            "ENABLE_TENSOR_DATA_REPRESENTATION=true at process startup"
        )
    return mode


def process_runtime_identity(mode):
    """Return bounded, non-secret ingest configuration for job telemetry."""
    runtime = {
        "videoIngestMode": mode,
        "tensorRepresentationEnabled": _env_flag(
            "ENABLE_TENSOR_DATA_REPRESENTATION"
        ),
    }
    latency = os.getenv("ROBOFLOW_RTSP_LATENCY_MS")
    if latency:
        try:
            latency_ms = int(latency)
        except ValueError:
            latency_ms = None
        if latency_ms is not None and 0 <= latency_ms <= 60_000:
            runtime["rtspLatencyMs"] = latency_ms
    return runtime


def build_cuda_producer(video_reference, on_created=None):
    """Construct v1.4's tensor-native NVDEC producer without CPU fallback."""
    from inference.core.interfaces.camera.gstreamer_cuda_producer import (
        GstreamerCudaVideoFrameProducer,
    )

    producer = GstreamerCudaVideoFrameProducer(
        video_reference,
        output_tensor=True,
    )
    if on_created is not None:
        on_created(producer)
    return producer


def producer_runtime_identity(producer):
    """Expose only bounded numeric bridge counters and a safe class name."""
    if producer is None:
        return {}
    runtime = {"videoProducer": type(producer).__name__[:128]}
    try:
        raw_stats = producer.tensor_bridge_stats
    except Exception:
        return runtime
    if not isinstance(raw_stats, dict):
        return runtime
    stats = {}
    for key, value in list(raw_stats.items())[:32]:
        key = str(key)
        if not _SAFE_STAT_KEY.fullmatch(key) or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            stats[key] = value
    if stats:
        runtime["tensorBridge"] = stats
    return runtime
