"""Environment variables read by inference_model_manager.

Single source of truth for env var names and defaults used across the
package. Mirrors the inference_models.configuration convention.

Values fetched at import-time are exposed as constants. Values that must
be re-read at runtime (e.g. inside forked workers, or per-call overrides)
are exposed as ``*_ENV`` name constants plus ``*_DEFAULT`` defaults; the
call site keeps the ``os.environ.get`` so the read happens at the right
moment.
"""

import os

from inference_models.utils.environment import (
    get_boolean_from_env,
    get_float_from_env,
    get_integer_from_env,
)

# ── Model Manager Process CLI defaults (model_manager_process.py) ──────────
MMP_N_SLOTS_DEFAULT = get_integer_from_env("MMP_N_SLOTS", default=32)
MMP_INPUT_MB_DEFAULT = get_float_from_env("MMP_INPUT_MB", default=20.0)

# ── VRAM-aware admission control (model_manager_process.py) ─────────────────
INFERENCE_VRAM_ADMISSION_CONTROL = get_boolean_from_env(
    "INFERENCE_VRAM_ADMISSION_CONTROL", default=False
)
INFERENCE_VRAM_WINDOW_SIZE = get_integer_from_env(
    "INFERENCE_VRAM_WINDOW_SIZE", default=60
)
# None → ModelManagerProcess reuses idle_timeout_s.
_vram_idle_cutoff = os.getenv("INFERENCE_VRAM_IDLE_CUTOFF_S")
INFERENCE_VRAM_IDLE_CUTOFF_S = (
    float(_vram_idle_cutoff) if _vram_idle_cutoff is not None else None
)
INFERENCE_VRAM_HEADROOM_MB = get_float_from_env(
    "INFERENCE_VRAM_HEADROOM_MB", default=512.0
)
INFERENCE_VRAM_RECENT_WINDOW_S = get_float_from_env(
    "INFERENCE_VRAM_RECENT_WINDOW_S", default=30.0
)

# ── Subprocess backend (backends/subproc.py) ───────────────────────────────
# Read inside the forked worker — names exposed here.
ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND_ENV = "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND"
ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND_DEFAULT = "False"
DEBUG_BENCHMARK_MODE_ENV = "DEBUG_BENCHMARK_MODE"

# ── ZMQ transport (backends/utils/transport.py) ────────────────────────────
# Re-read each call to allow per-call overrides.
INFERENCE_ZMQ_TRANSPORT_ENV = "INFERENCE_ZMQ_TRANSPORT"
INFERENCE_ZMQ_PORT_ENV_PREFIX = "INFERENCE_ZMQ_PORT_"
