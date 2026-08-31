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

from inference_models.utils.environment import get_float_from_env, get_integer_from_env

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
# Max number of ?image=<url> params fetched per request.
MAX_IMAGE_URLS = get_integer_from_env("INFERENCE_MAX_IMAGE_URLS", default=32)

# ── Auth (auth.py) ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.roboflow.com")
AUTH_CACHE_TTL_S = get_integer_from_env("AUTH_CACHE_TTL_S", default=3600)
AUTH_CACHE_FAIL_TTL_S = get_integer_from_env("AUTH_CACHE_FAIL_TTL_S", default=60)
AUTH_CACHE_MAX_SIZE = get_integer_from_env("AUTH_CACHE_MAX_SIZE", default=10000)

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

# ── Gateway resolution (gateway_resolver.resolve_gateway) ─────────────────
INFERENCE_GATEWAY_ENV = "INFERENCE_GATEWAY"
INFERENCE_GATEWAY_DEFAULT = "direct"

# ── API key fallback (server._preload_models) ─────────────────────────────
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"
