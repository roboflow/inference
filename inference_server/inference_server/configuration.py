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
from typing import Optional

from inference_models.utils.environment import (
    get_boolean_from_env,
    get_float_from_env,
    get_integer_from_env,
)


def _host_set(raw: Optional[str]) -> Optional[frozenset[str]]:
    if raw is None:
        return None
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())

# ── State timeouts (gateway.py) ───────────────────────────────────────────
LOAD_WAIT_S = get_float_from_env("INFERENCE_LOAD_WAIT_S", default=10.0)
INFER_TIMEOUT_S = get_float_from_env("INFERENCE_INFER_TIMEOUT_S", default=30.0)
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

# ── Auth (auth.py) ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.roboflow.com")
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
PRELOAD_API_KEY = os.environ.get("PRELOAD_API_KEY", "")

# ── Model-stat TTL-LRU cache (framework/model_stat.py) ────────────────────
MODEL_STAT_CACHE_SIZE = get_integer_from_env(
    "INFERENCE_MODEL_STAT_CACHE_SIZE", default=1024
)
MODEL_STAT_CACHE_TTL_S = get_float_from_env(
    "INFERENCE_MODEL_STAT_CACHE_TTL_S", default=300.0
)

# ── HTTP (app.py) ─────────────────────────────────────────────────────────
APP_PORT_DEFAULT = 8000
PORT_ENV = "PORT"
NUM_WORKERS = get_integer_from_env("NUM_WORKERS", default=1)

# ── App lifespan (app.py) ─────────────────────────────────────────────────
MULTIPART_SPOOL_MB = get_integer_from_env("INFERENCE_MULTIPART_SPOOL_MB", default=32)

# ── Preload / readiness (routers/v2_server) ────────────────────────────────
INFERENCE_PRELOAD_MODELS_ENV = "INFERENCE_PRELOAD_MODELS"


def preload_model_ids() -> list[str]:
    raw = os.environ.get(INFERENCE_PRELOAD_MODELS_ENV, "")
    return [m.strip() for m in raw.split(",") if m.strip()]

# ── Gateway resolution (gateway_resolver.resolve_gateway) ─────────────────
INFERENCE_GATEWAY_ENV = "INFERENCE_GATEWAY"
INFERENCE_GATEWAY_DEFAULT = "direct"

# ── API key fallback (server._preload_models) ─────────────────────────────
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"

# ── Legacy routes (legacy/, workflows/) ────────────────────────────────────
LEGACY_ROUTES_ENABLED = get_boolean_from_env("LEGACY_ROUTES_ENABLED", default=True)
LEGACY_CATCH_ALL_ROUTE_ENABLED = get_boolean_from_env(
    "LEGACY_ROUTE_ENABLED", default=True
)
LEGACY_CONTROL_PLANE_ROUTES_ENABLED = get_boolean_from_env(
    "LEGACY_CONTROL_PLANE_ROUTES_ENABLED", default=True
)
DISABLE_WORKFLOW_ENDPOINTS = get_boolean_from_env(
    "DISABLE_WORKFLOW_ENDPOINTS", default=False
)
OFFLINE_MODE = get_boolean_from_env("OFFLINE_MODE", default=False)
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
_SAM3_EXEC_MODE = os.environ.get("SAM3_EXEC_MODE", "local").lower()
SAM3_FINE_TUNED_MODELS_ENABLED = get_boolean_from_env(
    "SAM3_FINE_TUNED_MODELS_ENABLED", default=_SAM3_EXEC_MODE != "remote"
)
WORKFLOWS_MAX_CONCURRENT_STEPS = get_integer_from_env(
    "WORKFLOWS_MAX_CONCURRENT_STEPS", default=8
)
WORKFLOWS_THREAD_POOL_WORKERS = get_integer_from_env(
    "HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS", default=16
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
try:
    SERVER_VERSION = importlib.metadata.version("inference-server")
except importlib.metadata.PackageNotFoundError:
    SERVER_VERSION = "0.0.0"
SERVER_ID = uuid.uuid4().hex
