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

# ── State timeouts (state.py module-level) ────────────────────────────────
LOAD_WAIT_S = get_float_from_env("INFERENCE_LOAD_WAIT_S", default=10.0)
INFER_TIMEOUT_S = get_float_from_env("INFERENCE_INFER_TIMEOUT_S", default=30.0)
ALLOC_TIMEOUT_S = get_float_from_env("INFERENCE_ALLOC_TIMEOUT_S", default=2.0)

# ── MMP connection (state.init_from_env, re-read per worker) ──────────────
INFERENCE_MMP_ADDR_ENV = "INFERENCE_MMP_ADDR"
INFERENCE_SHM_NAME_ENV = "INFERENCE_SHM_NAME"
INFERENCE_SHM_NAME_DEFAULT = "inference_pool"
INFERENCE_SHM_DATA_SIZE_ENV = "INFERENCE_SHM_DATA_SIZE"
INFERENCE_SHM_DATA_SIZE_DEFAULT = 25 * 1024 * 1024

# ── Pipeline timing CSV (state.py) ────────────────────────────────────────
PIPELINE_CSV = os.environ.get("INFERENCE_PIPELINE_CSV", "")
PIPELINE_FLUSH_INTERVAL_S = get_float_from_env(
    "INFERENCE_PIPELINE_FLUSH_S", default=5.0
)

# ── Auth (auth.py) ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.roboflow.com")
AUTH_CACHE_TTL_S = get_integer_from_env("AUTH_CACHE_TTL_S", default=3600)
AUTH_CACHE_FAIL_TTL_S = get_integer_from_env("AUTH_CACHE_FAIL_TTL_S", default=60)
AUTH_CACHE_MAX_SIZE = get_integer_from_env("AUTH_CACHE_MAX_SIZE", default=10000)

# ── Server / MMP launch (server.main) ─────────────────────────────────────
NVIDIA_MPS_ENV = "NVIDIA_MPS"  # truthy check against literal "1"
SERVER_N_SLOTS = get_integer_from_env("INFERENCE_N_SLOTS", default=256)
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
# PORT has two distinct defaults: 8443 for ``server.main`` (TLS-capable
# orchestrated entry) and 8000 for the ``__main__`` block in app.py
# (uvicorn dev runner). Keep both defaults as constants.
SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
SERVER_PORT_DEFAULT = 8443
APP_PORT_DEFAULT = 8000
PORT_ENV = "PORT"
NUM_WORKERS = get_integer_from_env("NUM_WORKERS", default=4)
SSL_CERTFILE = os.environ.get("SSL_CERTFILE")
SSL_KEYFILE = os.environ.get("SSL_KEYFILE")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "warning").lower()

# ── App lifespan (app.py) ─────────────────────────────────────────────────
MULTIPART_SPOOL_MB = get_integer_from_env("INFERENCE_MULTIPART_SPOOL_MB", default=32)
DEBUG_BENCHMARK_MODE = os.environ.get("DEBUG_BENCHMARK_MODE", "").strip() == "1"

# ── Preload / readiness (server.main, routers/v2_server) ──────────────────
INFERENCE_PRELOAD_MODELS_ENV = "INFERENCE_PRELOAD_MODELS"

# ── Deployment mode (launcher.launch, app.py lifespan) ────────────────────
# Constants only — env is read at the entry boundary (app.py lifespan,
# server.main) so callable code stays testable without env manipulation.
INFERENCE_DEPLOYMENT_MODE_ENV = "INFERENCE_DEPLOYMENT_MODE"
MODE_BUNDLED = "bundled"      # MMWrapper over an in-process ModelManager
MODE_MMP = "mmp"              # MMPClient over ZMQ+SHM to ModelManagerProcess
INFERENCE_DEPLOYMENT_MODE_DEFAULT = MODE_BUNDLED

# ── API key fallback (server._preload_models) ─────────────────────────────
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"
