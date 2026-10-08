"""Environment variables read by inference_server.

Single source of truth for env var names and defaults used across the
package. Mirrors the inference_models.configuration convention.

Values fetched at import-time are exposed as constants. Values that must
be re-read at runtime (e.g. inside uvicorn workers spawned after fork,
or because two call sites use different defaults) are exposed as
``*_ENV`` name constants plus ``*_DEFAULT`` defaults; the call site keeps
the ``os.environ.get`` so the read happens at the right moment.
"""

import importlib.metadata
import os
import uuid
import warnings
from typing import Optional

from inference_models.configuration import INFERENCE_HOME as _MODELS_INFERENCE_HOME
from inference_models.configuration import OFFLINE_MODE as _MODELS_OFFLINE_MODE
from inference_models.configuration import SECURE_GATEWAY as _MODELS_SECURE_GATEWAY
from inference_models.configuration import (
    VLLM_PROXY_ENABLED as _MODELS_VLLM_PROXY_ENABLED,
)
from inference_models.configuration import (
    VLLM_REQUEST_TIMEOUT_S as _MODELS_VLLM_REQUEST_TIMEOUT_S,
)
from inference_models.utils.environment import (
    get_boolean_from_env,
    get_float_from_env,
    get_integer_from_env,
    str2bool,
)


def _host_set(raw: Optional[str]) -> Optional[frozenset[str]]:
    if raw is None:
        return None
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def _optional_positive_integer_from_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _optional_integer_from_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return get_integer_from_env(name)


def _optional_float_from_env(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return get_float_from_env(name)


def _optional_boolean_from_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return get_boolean_from_env(name)


def _absolute_number_from_env(name: str, parser, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return abs(parser(raw))
    except ValueError:
        return default


def _telemetry_env_name(name: str) -> str:
    if name in os.environ:
        return name
    folded = name.lower()
    for candidate in sorted(os.environ):
        if candidate.lower() == folded:
            return candidate
    return name


# ── State timeouts (gateway.py) ───────────────────────────────────────────
LOAD_WAIT_S = get_float_from_env("INFERENCE_LOAD_WAIT_S", default=10.0)
# vLLM pools: must cover VLLM_REQUEST_TIMEOUT_S, or long generations 504 first.
INFER_TIMEOUT_S = get_float_from_env(
    "INFERENCE_INFER_TIMEOUT_S",
    default=(
        max(30.0, _MODELS_VLLM_REQUEST_TIMEOUT_S)
        if _MODELS_VLLM_PROXY_ENABLED
        else 30.0
    ),
)
# Hard ceiling on a single request body for the v2 dispatch path; enforced
# both from Content-Length and while streaming (chunked uploads have none).
# Also the aggregate budget for URL-sourced images, so URL inputs are bounded
# the same as body inputs.
MAX_BODY_BYTES = get_integer_from_env(
    "INFERENCE_MAX_BODY_BYTES", default=100 * 1024 * 1024
)
# Max number of images accepted in a single request, whatever the source
# (JSON base64 list, repeated multipart parts, ?image=<url> params). The byte
# budget alone does not bound the COUNT: a compact payload can carry a huge
# list and spawn one executor submission per entry.
MAX_IMAGES_PER_REQUEST = get_integer_from_env(
    "INFERENCE_MAX_IMAGES_PER_REQUEST", default=32
)
# Max images of one request processed concurrently (0 = unbounded).
MAX_CONCURRENT_IMAGES_PER_REQUEST = get_integer_from_env(
    "INFERENCE_MAX_CONCURRENT_IMAGES_PER_REQUEST", default=8
)

# ── URL image inputs (framework/input_parsers/url_fetch.py) ───────────────
# Destination guarding for ?image=<url>. Env names match the legacy inference
# package so one deployment configures both the same way.
# When set, only these hosts may be fetched — and they are trusted, so the
# non-global address check does not apply to them.
WHITELISTED_DESTINATIONS_FOR_URL_INPUT = _host_set(
    os.environ.get("WHITELISTED_DESTINATIONS_FOR_URL_INPUT")
)
# Always rejected, on top of everything else.
BLACKLISTED_DESTINATIONS_FOR_URL_INPUT = _host_set(
    os.environ.get("BLACKLISTED_DESTINATIONS_FOR_URL_INPUT")
)
# Loopback / RFC1918 / link-local (169.254.169.254) / reserved destinations are
# refused unless this is turned on. Off by default: a URL input reaching them is
# SSRF into the pod's own network.
ALLOW_URL_TO_NON_GLOBAL_ADDRESSES = get_boolean_from_env(
    "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", default=False
)
MAX_IMAGE_URL_REDIRECTS = get_integer_from_env("MAX_IMAGE_URL_REDIRECTS", default=3)
ALLOW_NON_HTTPS_URL_INPUT = get_boolean_from_env(
    "ALLOW_NON_HTTPS_URL_INPUT", default=False
)
ALLOW_URL_INPUT_WITHOUT_FQDN = get_boolean_from_env(
    "ALLOW_URL_INPUT_WITHOUT_FQDN", default=False
)
VALIDATE_IMAGE_URL_REDIRECTS = get_boolean_from_env(
    "VALIDATE_IMAGE_URL_REDIRECTS", default=False
)

# ── Workflows: Roboflow-platform blocks ───────────────────────────────────
# Reported as `device_id` by the model-monitoring block, as in `inference`.
DEVICE_ID = os.environ.get("DEVICE_ID")

# ── Auth (auth.py) ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.roboflow.com")
API_PROXY_BASE_URL = os.environ.get("API_PROXY_BASE_URL", API_BASE_URL)
AUTH_CACHE_TTL_S = get_integer_from_env("AUTH_CACHE_TTL_S", default=3600)
AUTH_CACHE_FAIL_TTL_S = get_integer_from_env("AUTH_CACHE_FAIL_TTL_S", default=60)
AUTH_CACHE_MAX_SIZE = get_integer_from_env("AUTH_CACHE_MAX_SIZE", default=10000)
# Model list/load/unload and server info/metrics validate a key but not its
# workspace — any customer key can drive them. Off unless the deployment
# trusts every key holder (single tenant).
ENABLE_CONTROL_PLANE_ROUTES = get_boolean_from_env(
    "ENABLE_CONTROL_PLANE_ROUTES", default=False
)
# API key used for INFERENCE_PRELOAD_MODELS startup loads (weight fetch).
PRELOAD_API_KEY = os.environ.get("PRELOAD_API_KEY") or os.environ.get(
    "ROBOFLOW_API_KEY", ""
)

# ── Prometheus (prometheus.py) ────────────────────────────────────────────
# GET /metrics in Prometheus text format, unauthenticated, like the legacy
# server (which served it regardless of this flag; its images set it True).
# Set to false to turn the route and the HTTP instrumentation off.
ENABLE_PROMETHEUS = get_boolean_from_env("ENABLE_PROMETHEUS", default=True)
METRICS_INCLUDE_SOURCE_LABELS = get_boolean_from_env(
    "METRICS_INCLUDE_SOURCE_LABELS", default=True
)

# ── Model-stat TTL-LRU cache (framework/model_stat.py) ────────────────────
MODEL_STAT_CACHE_SIZE = get_integer_from_env(
    "INFERENCE_MODEL_STAT_CACHE_SIZE", default=1024
)
MODEL_STAT_CACHE_TTL_S = get_float_from_env(
    "INFERENCE_MODEL_STAT_CACHE_TTL_S", default=300.0
)

# ── HTTP (app.py) ─────────────────────────────────────────────────────────
APP_PORT_DEFAULT = 9001
PORT_ENV = "PORT"
HOST = os.environ.get("HOST", "0.0.0.0")
NUM_WORKERS = get_integer_from_env("NUM_WORKERS", default=1)
CORRELATION_ID_HEADER = os.environ.get("CORRELATION_ID_HEADER", "X-Request-ID")
API_LOGGING_ENABLED = get_boolean_from_env("API_LOGGING_ENABLED", default=False)
EXECUTION_ID_HEADER = os.environ.get("EXECUTION_ID_HEADER", "execution_id")

# ── Logging (logging_config.py) ────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
STRUCTURED_API_LOGGING = get_boolean_from_env("STRUCTURED_API_LOGGING", default=False)
CORRELATION_ID_LOG_KEY = os.environ.get("CORRELATION_ID_LOG_KEY", "request_id")

# ── Landing page (app.py) ──────────────────────────────────────────────────
LANDING_DIR = os.environ.get(
    "LANDING_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "inference",
        "landing",
        "out",
    ),
)
ENABLE_DASHBOARD = get_boolean_from_env("ENABLE_DASHBOARD", default=False)

# ── Workflow Builder (app.py, builder/) ───────────────────────────────────
# NOTE: legacy resolves the builder origin from region + project; EU regions
# set BUILDER_ORIGIN explicitly.
ENABLE_BUILDER = get_boolean_from_env("ENABLE_BUILDER", default=False)
BUILDER_ORIGIN = os.environ.get(
    "BUILDER_ORIGIN",
    (
        "https://app.roboflow.one"
        if os.environ.get("PROJECT", "roboflow-platform") == "roboflow-staging"
        else "https://app.roboflow.com"
    ),
)

# ── App lifespan (app.py) ─────────────────────────────────────────────────
MULTIPART_SPOOL_MB = get_integer_from_env("INFERENCE_MULTIPART_SPOOL_MB", default=32)

# ── Preload / readiness (routers/v2_server) ────────────────────────────────
INFERENCE_PRELOAD_MODELS_ENV = "INFERENCE_PRELOAD_MODELS"
PINNED_MODELS_ENV = "PINNED_MODELS"
PRELOAD_HF_IDS_ENV = "PRELOAD_HF_IDS"


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _with_api_key(entry: str) -> tuple[str, str]:
    model_id, _, api_key = entry.partition(":")
    return model_id, api_key or PRELOAD_API_KEY


def preload_model_ids() -> list[tuple[str, str]]:
    """Return ``(model_id, api_key)`` pairs from ``INFERENCE_PRELOAD_MODELS``.

    An entry may carry its own key as ``model_id:api_key``; otherwise
    ``PRELOAD_API_KEY`` is used.

    Returns:
        Startup loads that are not pinned.
    """
    entries = [
        _with_api_key(entry) for entry in _env_list(INFERENCE_PRELOAD_MODELS_ENV)
    ]

    return entries


def pinned_model_ids() -> list[tuple[str, str]]:
    """Return ``(model_id, api_key)`` pairs from ``PINNED_MODELS``.

    An entry may carry its own key as ``model_id:api_key``; otherwise
    ``PRELOAD_API_KEY`` is used.

    Returns:
        Startup loads that are pinned against eviction.
    """
    entries = [_with_api_key(entry) for entry in _env_list(PINNED_MODELS_ENV)]

    return entries


def preload_hf_ids() -> list[str]:
    """Return the OWLv2 Hugging Face ids listed in ``PRELOAD_HF_IDS``.

    Returns:
        Backbone ids warmed at startup, never split on ``:``.
    """
    hf_ids = _env_list(PRELOAD_HF_IDS_ENV)

    return hf_ids


# ── Gateway resolution (gateway_resolver.resolve_gateway) ─────────────────
INFERENCE_GATEWAY_ENV = "INFERENCE_GATEWAY"
INFERENCE_GATEWAY_DEFAULT = "direct"

# ── API key fallback (server._preload_models) ─────────────────────────────
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"

# ── Legacy routes (legacy/, workflows/) ────────────────────────────────────
LEGACY_ROUTES_ENABLED = get_boolean_from_env("LEGACY_ROUTES_ENABLED", default=True)
ALLOW_API_KEY_FROM_HEADERS = get_boolean_from_env(
    "ALLOW_API_KEY_FROM_HEADERS", default=True
)
LEGACY_CATCH_ALL_ROUTE_ENABLED = get_boolean_from_env(
    "LEGACY_ROUTE_ENABLED", default=True
)
LEGACY_CONTROL_PLANE_ROUTES_ENABLED = get_boolean_from_env(
    "LEGACY_CONTROL_PLANE_ROUTES_ENABLED", default=True
)
DISABLE_WORKFLOW_ENDPOINTS = get_boolean_from_env(
    "DISABLE_WORKFLOW_ENDPOINTS", default=False
)
# Removes only the experimental `describe_workload` routes; every other Workflow
# route stays. `DISABLE_WORKFLOW_ENDPOINTS=True` still removes all of them.
DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS = get_boolean_from_env(
    "DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS", default=False
)
OFFLINE_MODE = get_boolean_from_env("OFFLINE_MODE", default=False)
LEGACY_OFFLINE_MODE = _MODELS_OFFLINE_MODE
DISABLE_INFERENCE_CACHE = get_boolean_from_env("DISABLE_INFERENCE_CACHE", default=False)
ACTIVE_LEARNING_ENABLED = get_boolean_from_env("ACTIVE_LEARNING_ENABLED", default=True)
if LEGACY_OFFLINE_MODE:
    ACTIVE_LEARNING_ENABLED = False
_ACTIVE_LEARNING_TAGS_RAW = os.environ.get("ACTIVE_LEARNING_TAGS")
ACTIVE_LEARNING_TAGS = (
    None if _ACTIVE_LEARNING_TAGS_RAW is None else _ACTIVE_LEARNING_TAGS_RAW.split(",")
)

# ── OpenTelemetry tracing and metrics (telemetry.py) ───────────────────────
OTEL_TRACING_ENABLED = get_boolean_from_env("OTEL_TRACING_ENABLED", default=False)
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "inference-server")
OTEL_EXPORTER_PROTOCOL = os.environ.get("OTEL_EXPORTER_PROTOCOL", "grpc")
OTEL_EXPORTER_ENDPOINT = os.environ.get("OTEL_EXPORTER_ENDPOINT", "localhost:4317")
OTEL_SAMPLING_RATE = get_float_from_env("OTEL_SAMPLING_RATE", default=1.0)
OTEL_TRACE_EXPORT_INTERVAL_MS = get_integer_from_env(
    "OTEL_TRACE_EXPORT_INTERVAL_MS", default=5000
)
OTEL_METRICS_ENABLED = get_boolean_from_env("OTEL_METRICS_ENABLED", default=True)
if OFFLINE_MODE:
    OTEL_TRACING_ENABLED = False
    OTEL_METRICS_ENABLED = False
OTEL_METRIC_EXPORTER_ENDPOINT = os.environ.get("OTEL_METRIC_EXPORTER_ENDPOINT", "")
OTEL_METRIC_EXPORT_INTERVAL_MS = get_integer_from_env(
    "OTEL_METRIC_EXPORT_INTERVAL_MS", default=10000
)

ALLOW_URL_INPUT = get_boolean_from_env("ALLOW_URL_INPUT", default=True)
ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM = get_boolean_from_env(
    "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", default=False
)
LEGACY_LOAD_TIMEOUT_S = get_float_from_env(
    "INFERENCE_LEGACY_LOAD_TIMEOUT_S", default=300.0
)
LEGACY_LOAD_POLL_INTERVAL_S = get_float_from_env(
    "INFERENCE_LEGACY_LOAD_POLL_INTERVAL_S", default=0.5
)
LEGACY_ROUTE_METADATA_TTL_S = get_float_from_env(
    "INFERENCE_LEGACY_ROUTE_METADATA_TTL_S", default=30.0
)
CONFIDENCE_LOWER_BOUND_OOM_PREVENTION = get_float_from_env(
    "CONFIDENCE_LOWER_BOUND_OOM_PREVENTION", default=0.01
)
CLIP_MAX_BATCH_SIZE = get_integer_from_env("CLIP_MAX_BATCH_SIZE", default=8)
CLIP_VERSION_ID = os.environ.get("CLIP_VERSION_ID", "ViT-B-16")
PERCEPTION_ENCODER_VERSION_ID = os.environ.get(
    "PERCEPTION_ENCODER_VERSION_ID", "PE-Core-L14-336"
)
SAM_VERSION_ID = os.environ.get("SAM_VERSION_ID", "vit_h")
SAM2_VERSION_ID = os.environ.get("SAM2_VERSION_ID", "hiera_large")
SAM3_MAX_PROMPT_BATCH_SIZE = get_integer_from_env(
    "SAM3_MAX_PROMPT_BATCH_SIZE", default=16
)
EASYOCR_VERSION_ID = os.environ.get("EASYOCR_VERSION_ID", "english_g2")
OWLV2_VERSION_ID = os.environ.get("OWLV2_VERSION_ID", "owlv2-large-patch14-ensemble")
CLASS_AGNOSTIC_NMS = get_boolean_from_env("CLASS_AGNOSTIC_NMS", default=False)
DEFAULT_CONFIDENCE = 0.4
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_CANDIDATES = 3000
DEFAULT_MAX_DETECTIONS = 300
ALLOW_ORIGINS = [o for o in os.environ.get("ALLOW_ORIGINS", "*").split(",") if o]
DEFAULT_API_KEY = (
    os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY") or None
)
INFERENCE_SERVER_ID = os.environ.get("INFERENCE_SERVER_ID") or None
GET_MODEL_REGISTRY_ENABLED = get_boolean_from_env(
    "GET_MODEL_REGISTRY_ENABLED", default=True
)
CORE_MODELS_ENABLED = get_boolean_from_env("CORE_MODELS_ENABLED", default=True)
CORE_MODEL_CLIP_ENABLED = get_boolean_from_env("CORE_MODEL_CLIP_ENABLED", default=True)
CORE_MODEL_PE_ENABLED = get_boolean_from_env("CORE_MODEL_PE_ENABLED", default=True)
CORE_MODEL_SAM_ENABLED = get_boolean_from_env("CORE_MODEL_SAM_ENABLED", default=True)
CORE_MODEL_SAM2_ENABLED = get_boolean_from_env("CORE_MODEL_SAM2_ENABLED", default=True)
CORE_MODEL_SAM3_ENABLED = get_boolean_from_env("CORE_MODEL_SAM3_ENABLED", default=True)
CORE_MODEL_OWLV2_ENABLED = get_boolean_from_env(
    "CORE_MODEL_OWLV2_ENABLED", default=False
)
CORE_MODEL_GAZE_ENABLED = get_boolean_from_env("CORE_MODEL_GAZE_ENABLED", default=True)
CORE_MODEL_DOCTR_ENABLED = get_boolean_from_env(
    "CORE_MODEL_DOCTR_ENABLED", default=True
)
CORE_MODEL_EASYOCR_ENABLED = get_boolean_from_env(
    "CORE_MODEL_EASYOCR_ENABLED", default=True
)
CORE_MODEL_TROCR_ENABLED = get_boolean_from_env(
    "CORE_MODEL_TROCR_ENABLED", default=True
)
CORE_MODEL_PPOCR_ENABLED = get_boolean_from_env(
    "CORE_MODEL_PPOCR_ENABLED", default=True
)
CORE_MODEL_GROUNDINGDINO_ENABLED = get_boolean_from_env(
    "CORE_MODEL_GROUNDINGDINO_ENABLED", default=True
)
CORE_MODEL_YOLO_WORLD_ENABLED = get_boolean_from_env(
    "CORE_MODEL_YOLO_WORLD_ENABLED", default=True
)
LMM_ENABLED = get_boolean_from_env("LMM_ENABLED", default=False)
MOONDREAM2_ENABLED = get_boolean_from_env("MOONDREAM2_ENABLED", default=True)
DEPTH_ESTIMATION_ENABLED = get_boolean_from_env(
    "DEPTH_ESTIMATION_ENABLED", default=True
)
SAM3_3D_OBJECTS_ENABLED = get_boolean_from_env("SAM3_3D_OBJECTS_ENABLED", default=False)
ACTION_RECOGNITION_ENABLED = get_boolean_from_env(
    "ACTION_RECOGNITION_ENABLED", default=True
)
MAX_VIDEO_DOWNLOAD_SIZE_MB = get_integer_from_env(
    "MAX_VIDEO_DOWNLOAD_SIZE_MB", default=512
)
VIDEO_DOWNLOAD_TIMEOUT_SECONDS = get_float_from_env(
    "VIDEO_DOWNLOAD_TIMEOUT_SECONDS", default=60.0
)
MAX_VIDEO_DURATION_SECONDS = get_float_from_env(
    "MAX_VIDEO_DURATION_SECONDS", default=600.0
)
SAM3_EXEC_MODE = os.environ.get("SAM3_EXEC_MODE", "local").lower()
SAM3_FINE_TUNED_MODELS_ENABLED = get_boolean_from_env(
    "SAM3_FINE_TUNED_MODELS_ENABLED", default=SAM3_EXEC_MODE != "remote"
)
if OFFLINE_MODE and SAM3_EXEC_MODE == "remote":
    warnings.warn(
        "SAM3_EXEC_MODE=remote is not available while OFFLINE_MODE is enabled. "
        "Forcing local SAM3 execution.",
        stacklevel=1,
    )
    SAM3_EXEC_MODE = "local"
    if os.environ.get("SAM3_FINE_TUNED_MODELS_ENABLED") is None:
        SAM3_FINE_TUNED_MODELS_ENABLED = True
DISABLE_SAM3_LOGITS_CACHE = get_boolean_from_env(
    "DISABLE_SAM3_LOGITS_CACHE", default=False
)
DISABLE_SAM2_LOGITS_CACHE = get_boolean_from_env(
    "DISABLE_SAM2_LOGITS_CACHE", default=False
)
SAM3_MAX_DETECTIONS = get_integer_from_env("SAM3_MAX_DETECTIONS", default=-1)
WORKFLOWS_MAX_CONCURRENT_STEPS = get_integer_from_env(
    "WORKFLOWS_MAX_CONCURRENT_STEPS", default=8
)
WORKFLOWS_THREAD_POOL_WORKERS = get_integer_from_env(
    "HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS", default=16
)
WORKFLOWS_THREAD_POOL_ENABLED = get_boolean_from_env(
    "HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED", default=True
)
HTTP_API_THREADPOOL_WORKERS = _optional_positive_integer_from_env(
    "HTTP_API_THREADPOOL_WORKERS"
)
ENABLE_WORKFLOWS_PROFILING = get_boolean_from_env(
    "ENABLE_WORKFLOWS_PROFILING", default=False
)
WORKFLOWS_PROFILER_BUFFER_SIZE = get_integer_from_env(
    "WORKFLOWS_PROFILER_BUFFER_SIZE", default=64
)
WORKFLOWS_DEFINITION_CACHE_TTL_S = get_integer_from_env(
    "WORKFLOWS_DEFINITION_CACHE_EXPIRY", default=15 * 60
)
USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS = get_boolean_from_env(
    "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS", default=True
)
SINGLE_TENANT_WORKFLOW_CACHE = get_boolean_from_env(
    "SINGLE_TENANT_WORKFLOW_CACHE", default=False
)
if LEGACY_OFFLINE_MODE:
    if not USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS:
        warnings.warn(
            "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS=False is not available "
            "while OFFLINE_MODE is enabled. Forcing the file cache on so "
            "pre-warmed Workflow definitions remain usable.",
            stacklevel=1,
        )
    USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS = True
    SINGLE_TENANT_WORKFLOW_CACHE = True
ALLOW_WORKFLOWS_FONTS_DOWNLOAD = get_boolean_from_env(
    "ALLOW_WORKFLOWS_FONTS_DOWNLOAD", default=True
)

# ── Stream API (streams/configuration.py, streams/host.py) ────────────────
ENABLE_STREAM_API = get_boolean_from_env("ENABLE_STREAM_API", default=False)
STREAM_API_PRELOADED_PROCESSES = get_integer_from_env(
    "STREAM_API_PRELOADED_PROCESSES", default=0
)
STREAM_MANAGER_HOST = os.environ.get("STREAM_MANAGER_HOST", "127.0.0.1")
STREAM_MANAGER_PORT = get_integer_from_env("STREAM_MANAGER_PORT", default=7070)
STREAM_MANAGER_SOCKET_TIMEOUT = get_float_from_env(
    "STREAM_MANAGER_SOCKET_TIMEOUT", default=5.0
)
STREAM_MANAGER_OPERATIONS_TIMEOUT = _optional_float_from_env(
    "STREAM_MANAGER_OPERATIONS_TIMEOUT"
)
STREAM_MANAGER_MAX_ACTIVE_PIPELINES = max(
    get_integer_from_env("STREAM_MANAGER_MAX_ACTIVE_PIPELINES", default=8),
    STREAM_API_PRELOADED_PROCESSES,
)
STREAM_MANAGER_MAX_RAM_MB = _absolute_number_from_env(
    "STREAM_MANAGER_MAX_RAM_MB", float, default=None
)
STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE = _absolute_number_from_env(
    "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE", int, default=10
)
VIDEO_SOURCE_BUFFER_SIZE = _optional_integer_from_env("VIDEO_SOURCE_BUFFER_SIZE")
VIDEO_SOURCE_BUFFER_SIZE_DEFAULT = 64
VIDEO_SOURCE_BUFFER_SIZE_TENSOR_DEFAULT = 8
VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE = _optional_boolean_from_env(
    "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE"
)
VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE = get_float_from_env(
    "VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE", default=5.0
)
VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE = get_float_from_env(
    "VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE", default=0.1
)
VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW = get_integer_from_env(
    "VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW", default=16
)
VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES = get_integer_from_env(
    "VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES", default=10
)
DISABLE_GSTREAMER_VIDEO_SOURCES = get_boolean_from_env(
    "DISABLE_GSTREAMER_VIDEO_SOURCES", default=False
)
DISABLE_NATIVE_STDERR_CAPTURE = get_boolean_from_env(
    "DISABLE_NATIVE_STDERR_CAPTURE", default=False
)
INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY = get_integer_from_env(
    "INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY", default=1
)
RUNS_ON_JETSON = str2bool(
    os.environ.get("RUNS_ON_JETSON", os.environ.get("RUNNING_ON_JETSON", "False")),
    variable_name="RUNS_ON_JETSON",
)
ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING = get_boolean_from_env(
    "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING", default=False
)
INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE = get_integer_from_env(
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE", default=512
)
INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE_EXPLICIT = (
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE" in os.environ
)
ALLOW_UNSAFE_GSTREAMER_PIPELINES = get_boolean_from_env(
    "ALLOW_UNSAFE_GSTREAMER_PIPELINES", default=False
)
DEBUG_AIORTC_QUEUES = get_boolean_from_env("DEBUG_AIORTC_QUEUES", default=False)
DEBUG_WEBRTC_PROCESSING_LATENCY = get_boolean_from_env(
    "DEBUG_WEBRTC_PROCESSING_LATENCY", default=False
)
WEBRTC_REALTIME_PROCESSING = get_boolean_from_env(
    "WEBRTC_REALTIME_PROCESSING", default=True
)

# ── Workflows: enterprise blocks and MQTT broker policy (workflows/host.py) ─
LOAD_ENTERPRISE_BLOCKS = get_boolean_from_env("LOAD_ENTERPRISE_BLOCKS", default=False)
MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST = get_boolean_from_env(
    "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", default=True
)
_MQTT_WHITELISTED_HOSTS_RAW = os.environ.get("MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS")
MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS = (
    None
    if _MQTT_WHITELISTED_HOSTS_RAW is None
    else tuple(
        entry.strip()
        for entry in _MQTT_WHITELISTED_HOSTS_RAW.split(",")
        if entry.strip()
    )
)

# ── Roboflow platform access (workflows/host.py) ──────────────────────────
SECURE_GATEWAY = _MODELS_SECURE_GATEWAY
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/cache")
INFERENCE_HOME = _MODELS_INFERENCE_HOME
ROBOFLOW_API_EXTRA_HEADERS = os.environ.get("ROBOFLOW_API_EXTRA_HEADERS")
ROBOFLOW_INTERNAL_SERVICE_NAME = os.environ.get("ROBOFLOW_INTERNAL_SERVICE_NAME")
ROBOFLOW_INTERNAL_SERVICE_SECRET = os.environ.get("ROBOFLOW_INTERNAL_SERVICE_SECRET")
# api_key -> workspace lookups made by the Workflows platform blocks. Same
# variables and defaults as the legacy server's `get_roboflow_workspace` cache.
WORKSPACE_CACHE_TTL_S = get_integer_from_env(
    "MODELS_CACHE_AUTH_CACHE_TTL", default=15 * 60
)
WORKSPACE_CACHE_MAX_SIZE = get_integer_from_env(
    "MODELS_CACHE_AUTH_CACHE_MAX_SIZE", default=100_000_000
)
try:
    SERVER_VERSION = importlib.metadata.version("inference-server")
except importlib.metadata.PackageNotFoundError:
    SERVER_VERSION = "0.0.0"
SERVER_ID = uuid.uuid4().hex

# ── Hosted deployments (hosted/, app.py, legacy/router.py) ────────────────
LAMBDA = get_boolean_from_env("LAMBDA", default=False)
GCP_SERVERLESS = get_boolean_from_env("GCP_SERVERLESS", default=False)
WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING = get_boolean_from_env(
    "WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING", default=True
)
ENFORCE_CREDITS_VERIFICATION = get_boolean_from_env(
    "ENFORCE_CREDITS_VERIFICATION", default=False
)
DEDICATED_DEPLOYMENT_ID = os.environ.get("DEDICATED_DEPLOYMENT_ID")
DEDICATED_DEPLOYMENT_WORKSPACE_ID = os.environ.get("DEDICATED_DEPLOYMENT_WORKSPACE_ID")
DEDICATED_DEPLOYMENT_WORKSPACE_URL = os.environ.get(
    "DEDICATED_DEPLOYMENT_WORKSPACE_URL"
)
_WORKSPACES_WHITELISTED_RAW = os.environ.get(
    "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT"
)
WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT = (
    None
    if _WORKSPACES_WHITELISTED_RAW is None
    else [entry.strip() for entry in _WORKSPACES_WHITELISTED_RAW.split(",")]
)
ROBOFLOW_SERVICE_SECRET = os.environ.get("ROBOFLOW_SERVICE_SECRET")
TRANSIENT_ROBOFLOW_API_ERRORS = {
    int(entry) for entry in _env_list("TRANSIENT_ROBOFLOW_API_ERRORS")
}
TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES = get_integer_from_env(
    "TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES", default=3
)
TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL = get_integer_from_env(
    "TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL", default=1
)
RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API = get_boolean_from_env(
    "RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", default=False
)
ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN = os.environ.get(
    "ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN"
) or os.environ.get("ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN")

# ── Operational routes (ops/) ─────────────────────────────────────────────
DOCKER_SOCKET_PATH = os.environ.get("DOCKER_SOCKET_PATH")
SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED = get_boolean_from_env(
    "SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED", default=False
)
SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT = get_float_from_env(
    "SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT", default=5.0
)
ROBOFLOW_API_VERIFY_SSL = get_boolean_from_env("ROBOFLOW_API_VERIFY_SSL", default=True)
NOTEBOOK_ENABLED = get_boolean_from_env("NOTEBOOK_ENABLED", default=False)
NOTEBOOK_PORT = get_integer_from_env("NOTEBOOK_PORT", default=9002)
NOTEBOOK_PASSWORD = os.environ.get("NOTEBOOK_PASSWORD") or None
ENABLE_IN_MEMORY_LOGS = get_boolean_from_env("ENABLE_IN_MEMORY_LOGS", default=False)

# ── Pingback (pingback.py) ────────────────────────────────────────────────
METRICS_ENABLED = get_boolean_from_env("METRICS_ENABLED", default=True)
if LAMBDA or GCP_SERVERLESS or LEGACY_OFFLINE_MODE:
    METRICS_ENABLED = False
METRICS_INTERVAL = get_integer_from_env("METRICS_INTERVAL", default=60)
METRICS_URL = os.environ.get("METRICS_URL", f"{API_BASE_URL}/inference-stats")
TINY_CACHE = get_boolean_from_env("TINY_CACHE", default=True)
TAGS = os.environ.get("TAGS", "").split(",")
METRICS_API_KEY = os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY")

# ── Usage reporting (usage/) ──────────────────────────────────────────────
METRICS_COLLECTOR_BASE_URL = os.environ.get("METRICS_COLLECTOR_BASE_URL", API_BASE_URL)
TELEMETRY_API_USAGE_ENDPOINT_URL = os.environ.get(
    _telemetry_env_name("TELEMETRY_API_USAGE_ENDPOINT_URL"),
    f"{METRICS_COLLECTOR_BASE_URL}/usage/inference",
)
TELEMETRY_FLUSH_INTERVAL = min(
    max(
        get_integer_from_env(
            _telemetry_env_name("TELEMETRY_FLUSH_INTERVAL"), default=10
        ),
        10,
    ),
    300,
)
TELEMETRY_QUEUE_SIZE = min(
    max(
        get_integer_from_env(_telemetry_env_name("TELEMETRY_QUEUE_SIZE"), default=10),
        10,
    ),
    10000,
)
TELEMETRY_USE_PERSISTENT_QUEUE = get_boolean_from_env(
    _telemetry_env_name("TELEMETRY_USE_PERSISTENT_QUEUE"), default=True
)
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = get_integer_from_env("REDIS_PORT", default=6379)
REDIS_SSL = get_boolean_from_env("REDIS_SSL", default=False)
REDIS_TIMEOUT = get_float_from_env("REDIS_TIMEOUT", default=2.0)
