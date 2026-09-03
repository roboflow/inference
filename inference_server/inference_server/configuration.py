"""Environment variables read by inference_server.

Single source of truth for env var names and defaults used across the
package. Mirrors the inference_models.configuration convention.

Values fetched at import-time are exposed as constants. Values that must
be re-read at runtime (e.g. inside uvicorn workers spawned after fork,
or because two call sites use different defaults) are exposed as
``*_ENV`` name constants plus ``*_DEFAULT`` defaults; the call site keeps
the ``os.environ.get`` so the read happens at the right moment.
"""

import os
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
