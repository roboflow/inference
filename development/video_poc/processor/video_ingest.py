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


def configure_source_fps_limiter_default():
    """Prefer source-side sampling for hosted video jobs.

    Inference's legacy post-collection limiter uses wall-clock spacing between
    accepted frames. Bursty live producers can therefore lose throughput even
    when their average cadence is at or below ``maxFps``. The source-side path
    samples before buffering and avoids that cadence aliasing. ``setdefault``
    deliberately preserves an explicit deployment rollback to ``false``.

    This must run before importing ``inference.core.env``.
    """
    os.environ.setdefault(
        "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
        "true",
    )


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def resolve_video_ingest_mode(value=None, tensor_runtime_available=None):
    mode = (value or os.getenv("PROCESSOR_VIDEO_INGEST_MODE", PYAV_INGEST)).strip()
    mode = mode.lower()
    if mode not in SUPPORTED_INGEST_MODES:
        raise ValueError(
            "PROCESSOR_VIDEO_INGEST_MODE must be one of "
            + ", ".join(SUPPORTED_INGEST_MODES)
        )
    tensor_enabled = _env_flag("ENABLE_TENSOR_DATA_REPRESENTATION")
    if tensor_runtime_available is not None:
        tensor_enabled = tensor_enabled and bool(tensor_runtime_available)
    if mode == GSTREAMER_CUDA_INGEST and not tensor_enabled:
        raise ValueError(
            "gstreamer_cuda ingest requires "
            "ENABLE_TENSOR_DATA_REPRESENTATION=true at process startup"
        )
    return mode


def process_runtime_identity(mode, tensor_runtime_available=None):
    """Return bounded, non-secret ingest configuration for job telemetry."""
    tensor_enabled = _env_flag("ENABLE_TENSOR_DATA_REPRESENTATION")
    if tensor_runtime_available is not None:
        tensor_enabled = tensor_enabled and bool(tensor_runtime_available)
    runtime = {
        "videoIngestMode": mode,
        "tensorRepresentationEnabled": tensor_enabled,
        "sourceFpsLimiterAtProducer": _env_flag(
            "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING"
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
    appsink_sync = getattr(producer, "appsink_sync_enabled", None)
    if isinstance(appsink_sync, bool):
        runtime["gstreamerAppSinkSync"] = appsink_sync
    try:
        raw_stream = producer.source_stream_metadata
    except Exception:
        raw_stream = None
    if isinstance(raw_stream, dict):
        stream = {}
        for key in ("width", "height", "fps", "fpsNumerator", "fpsDenominator"):
            value = raw_stream.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                stream[key] = value
        codec = raw_stream.get("codec")
        if isinstance(codec, str) and _SAFE_STAT_KEY.fullmatch(codec):
            stream["codec"] = codec
        if stream:
            runtime["sourceStream"] = stream
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


def verify_cuda_frame(frame_image):
    """Fail an NVDEC experiment if a reconnect ever yields a host frame."""
    if not bool(getattr(frame_image, "is_cuda", False)):
        raise RuntimeError(
            "gstreamer_cuda ingest produced a non-CUDA frame; refusing CPU fallback"
        )
