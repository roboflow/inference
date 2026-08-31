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
