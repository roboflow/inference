"""Environment variables read by inference_server.

Single source of truth for env var names and defaults used across the
package. Mirrors the inference_models.configuration convention.

Values fetched at import-time are exposed as constants. Values that must
be re-read at runtime (e.g. inside uvicorn workers spawned after fork,
or because two call sites use different defaults) are exposed as
``*_ENV`` name constants plus ``*_DEFAULT`` defaults; the call site keeps
the ``os.environ.get`` so the read happens at the right moment.
"""

import math
import os

from inference_models.utils.environment import (
    get_boolean_from_env,
    get_float_from_env,
    get_integer_from_env,
)

# ── State timeouts (state.py module-level) ────────────────────────────────
LOAD_WAIT_S = get_float_from_env("INFERENCE_LOAD_WAIT_S", default=10.0)
INFER_TIMEOUT_S = get_float_from_env("INFERENCE_INFER_TIMEOUT_S", default=30.0)
ALLOC_TIMEOUT_S = get_float_from_env("INFERENCE_ALLOC_TIMEOUT_S", default=2.0)
ENSURE_CACHE_TTL_S = get_float_from_env("INFERENCE_ENSURE_CACHE_TTL_S", default=5.0)
SHM_ADMISSION = get_boolean_from_env("INFERENCE_SHM_ADMISSION", default=True)
# Optional resolution reject gate: reject images whose header dims exceed this
# megapixel cap before alloc/SHM-write/decode. 0 = disabled (never reject).
MAX_DECODED_MEGAPIXELS = get_float_from_env(
    "INFERENCE_MAX_DECODED_MEGAPIXELS", default=0.0
)

# ── MMP connection (state.init_from_env, re-read per worker) ──────────────
INFERENCE_MMP_ADDR_ENV = "INFERENCE_MMP_ADDR"
INFERENCE_SHM_NAME_ENV = "INFERENCE_SHM_NAME"
INFERENCE_SHM_NAME_DEFAULT = "inference_pool"
INFERENCE_SHM_DATA_SIZE_ENV = "INFERENCE_SHM_DATA_SIZE"
INFERENCE_SHM_DATA_SIZE_DEFAULT = 25 * 1024 * 1024

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

# ── Server / MMP launch (server.main) ─────────────────────────────────────
NVIDIA_MPS = get_boolean_from_env("NVIDIA_MPS", default=False)
SERVER_N_SLOTS = get_integer_from_env("INFERENCE_N_SLOTS", default=32)
SERVER_INPUT_MB = get_float_from_env("INFERENCE_INPUT_MB", default=25.0)
SERVER_DECODER = os.environ.get("INFERENCE_DECODER", "imagecodecs")
SERVER_BATCH_MAX_SIZE = get_integer_from_env("INFERENCE_BATCH_MAX_SIZE", default=0)
SERVER_BATCH_MAX_WAIT_MS = get_float_from_env(
    "INFERENCE_BATCH_MAX_WAIT_MS", default=5.0
)
SERVER_MODEL_IDLE_TIMEOUT_S = get_float_from_env(
    "INFERENCE_MODEL_IDLE_TIMEOUT_S", default=300.0
)

# ── HTTP / TLS (server.main) ──────────────────────────────────────────────
# PORT defaults to 8000 for both ``server.main`` (orchestrated entry) and the
# ``__main__`` block in app.py (uvicorn dev runner). Keep both as constants.
SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
SERVER_PORT_DEFAULT = 8000
APP_PORT_DEFAULT = 8000
PORT_ENV = "PORT"
NUM_WORKERS = get_integer_from_env("NUM_WORKERS", default=4)
# Per-worker uvicorn in-flight cap. Bodies are buffered before SHM admission, so
# uncapped concurrency × payload size is what OOMs the server under load. Default
# spreads the slot pool across workers (+25% slack); requests past it get an
# immediate 503 before the body is read. Override with INFERENCE_LIMIT_CONCURRENCY.
LIMIT_CONCURRENCY_ENV = "INFERENCE_LIMIT_CONCURRENCY"


def limit_concurrency(n_slots: int, workers: int) -> int:
    override = os.environ.get(LIMIT_CONCURRENCY_ENV)
    if override:
        return int(override)
    # 125% of this worker's slot share (floored)
    return max(1, math.ceil(n_slots / max(1, workers)) * 5 // 4)


SSL_CERTFILE = os.environ.get("SSL_CERTFILE")
SSL_KEYFILE = os.environ.get("SSL_KEYFILE")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "warning").lower()

# ── App lifespan (app.py) ─────────────────────────────────────────────────
MULTIPART_SPOOL_MB = get_integer_from_env("INFERENCE_MULTIPART_SPOOL_MB", default=32)
DEBUG_PASSTHROUGH_MODEL = get_boolean_from_env("DEBUG_PASSTHROUGH_MODEL", default=False)

# ── Preload / readiness (server.main, routers/v2_server) ──────────────────
INFERENCE_PRELOAD_MODELS_ENV = "INFERENCE_PRELOAD_MODELS"

# ── Deployment mode (launcher.launch, app.py lifespan) ────────────────────
# Constants only — env is read at the entry boundary (app.py lifespan,
# server.main) so callable code stays testable without env manipulation.
INFERENCE_DEPLOYMENT_MODE_ENV = "INFERENCE_DEPLOYMENT_MODE"
MODE_BUNDLED = "bundled"  # MMWrapper over an in-process ModelManager
MODE_MMP = "mmp"  # MMPClient over ZMQ+SHM to ModelManagerProcess
INFERENCE_DEPLOYMENT_MODE_DEFAULT = MODE_BUNDLED

# ── API key fallback (server._preload_models) ─────────────────────────────
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"
