"""Environment variables read by inference_model_manager.

Single source of truth for env var names and defaults used across the
package. Mirrors the inference_models.configuration convention.

Values fetched at import-time are exposed as constants. Values that must
be re-read at runtime (e.g. inside forked workers, or per-call overrides)
are exposed as ``*_ENV`` name constants plus ``*_DEFAULT`` defaults; the
call site keeps the ``os.environ.get`` so the read happens at the right
moment.
"""

from inference_models.utils.environment import (
    get_boolean_from_env,
    get_float_from_env,
    get_integer_from_env,
)

# ── ModelManager (model_manager.py) ─────────────────────────────────────────
INFERENCE_DIRECT_MAX_WORKERS = get_integer_from_env(
    "INFERENCE_DIRECT_MAX_WORKERS", default=8
)
INFERENCE_PROCESS_TIMEOUT_S = get_float_from_env(
    "INFERENCE_PROCESS_TIMEOUT_S", default=300.0
)
# Max concurrently loaded models; on load the least-recently-used unpinned
# model is drain-unloaded to make room. <=0 disables the cap.
INFERENCE_MAX_ACTIVE_MODELS = get_integer_from_env(
    "INFERENCE_MAX_ACTIVE_MODELS", default=8
)
# Fraction of free GPU memory below which a load first drain-unloads LRU
# models (up to 3 per pass, empty_cache re-check to avoid flapping).
# 0 disables the check.
INFERENCE_MEMORY_FREE_THRESHOLD = get_float_from_env(
    "INFERENCE_MEMORY_FREE_THRESHOLD", default=0.0
)

# ── Decode gate (backends/decode.py) ───────────────────────────────────────
# Decompression-bomb ceiling: encoded-byte limits say nothing about how much
# memory a decode allocates (a few hundred KB of PNG expands to gigabytes).
# Generous enough for real payloads — a 512MP image is ~1.5GB decoded — so it
# only ever catches deliberate bombs. Checked from the container header before
# decode where the format allows, and again on the decoded array. 0 disables.
INFERENCE_DECODE_MAX_MEGAPIXELS = get_float_from_env(
    "INFERENCE_DECODE_MAX_MEGAPIXELS", default=512.0
)

# ── In-memory model caches (backends/base.py attach_model_caches) ──────────
# Env names shared with the legacy inference package for deployment parity.
SAM_MAX_EMBEDDING_CACHE_SIZE = get_integer_from_env(
    "SAM_MAX_EMBEDDING_CACHE_SIZE", default=10
)
SAM2_MAX_EMBEDDING_CACHE_SIZE = get_integer_from_env(
    "SAM2_MAX_EMBEDDING_CACHE_SIZE", default=100
)
SAM2_MAX_LOGITS_CACHE_SIZE = get_integer_from_env(
    "SAM2_MAX_LOGITS_CACHE_SIZE", default=1000
)
SAM3_MAX_EMBEDDING_CACHE_SIZE = get_integer_from_env(
    "SAM3_MAX_EMBEDDING_CACHE_SIZE", default=100
)
SAM3_MAX_LOGITS_CACHE_SIZE = get_integer_from_env(
    "SAM3_MAX_LOGITS_CACHE_SIZE", default=1000
)
SAM3_INTERACTIVE_CACHE_SEND_TO_CPU = get_boolean_from_env(
    "SAM3_INTERACTIVE_CACHE_SEND_TO_CPU", default=True
)
OWLV2_MODEL_CACHE_SIZE = get_integer_from_env("OWLV2_MODEL_CACHE_SIZE", default=100)
OWLV2_IMAGE_CACHE_SIZE = get_integer_from_env("OWLV2_IMAGE_CACHE_SIZE", default=10000)
OWLV2_CACHE_SEND_TO_CPU = get_boolean_from_env("OWLV2_CACHE_SEND_TO_CPU", default=True)

# ── Watchdogs (watchdogs.py) ────────────────────────────────────────────────
# Legacy env names kept on purpose: one deployment can configure both the
# legacy inference stack and this one from the same environment.
MAX_INFERENCE_MODELS_CACHE_SIZE_MB = get_integer_from_env(
    "MAX_INFERENCE_MODELS_CACHE_SIZE_MB", default=-1
)
INFERENCE_MODELS_CACHE_WATCHDOG_INTERVAL_MINUTES = get_float_from_env(
    "INFERENCE_MODELS_CACHE_WATCHDOG_INTERVAL_MINUTES", default=60.0
)
ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG = get_boolean_from_env(
    "ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG", default=False
)
CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS = get_float_from_env(
    "CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS", default=300.0
)
