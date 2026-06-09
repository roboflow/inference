"""Environment configuration for the vLLM proxy backend.

All env vars specific to this package are read here with `os.getenv` (the
package must not require changes to `inference/core/env.py`). Values that may
be changed between requests/tests are exposed as functions reading the
environment at call time; only the top-level enablement switch is resolved at
import time (it controls model-class registration which also happens at
import time).
"""

import os
from typing import Optional


def str_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes", "y", "on"}


# Top-level switch: when true, qwen3_5 model types are served by the vLLM
# proxy class instead of the in-process HF adapter.
VLLM_PROXY_ENABLED = str_to_bool(os.getenv("VLLM_PROXY_ENABLED"))

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_VLLM_REQUEST_TIMEOUT_S = 120
DEFAULT_VLLM_MAX_LORA_RANK = 64
DEFAULT_VLLM_MAX_REGISTERED_ADAPTERS = 64
DEFAULT_VLLM_VISION_LORA_NORM_THRESHOLD = 0.0
DEFAULT_VLLM_DORA_POLICY = "reject"
DEFAULT_VLLM_SERVED_BASE_VARIANT = "qwen3_5-0.8b"

# Template applied when remapping Roboflow adapter weight keys to the layout
# vLLM / PEFT expect for the qwen3_5 VL architecture. `{suffix}` is the part
# of the key following the `model.layers.` / `model.language_model.layers.`
# prefix. NOTE: the exact layout vLLM v0.22.1 accepts is empirically
# unconfirmed - this template is configurable so the staging lab can adjust
# it without a code change.
DEFAULT_VLLM_ADAPTER_KEY_TEMPLATE = (
    "base_model.model.model.language_model.layers.{suffix}"
)

# Default fallback for MODEL_CACHE_DIR mirrors inference/core/env.py.
DEFAULT_MODEL_CACHE_DIR = "/tmp/cache"


def get_vllm_base_url() -> str:
    return os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)


def get_vllm_request_timeout_s() -> float:
    return float(
        os.getenv("VLLM_REQUEST_TIMEOUT_S", str(DEFAULT_VLLM_REQUEST_TIMEOUT_S))
    )


def get_vllm_max_lora_rank() -> int:
    return int(os.getenv("VLLM_MAX_LORA_RANK", str(DEFAULT_VLLM_MAX_LORA_RANK)))


def get_vllm_max_registered_adapters() -> int:
    return int(
        os.getenv(
            "VLLM_MAX_REGISTERED_ADAPTERS",
            str(DEFAULT_VLLM_MAX_REGISTERED_ADAPTERS),
        )
    )


def get_vllm_vision_lora_norm_threshold() -> float:
    return float(
        os.getenv(
            "VLLM_VISION_LORA_NORM_THRESHOLD",
            str(DEFAULT_VLLM_VISION_LORA_NORM_THRESHOLD),
        )
    )


def get_vllm_dora_policy() -> str:
    return os.getenv("VLLM_DORA_POLICY", DEFAULT_VLLM_DORA_POLICY).strip().lower()


def get_vllm_served_base_variant() -> str:
    return os.getenv("VLLM_SERVED_BASE_VARIANT", DEFAULT_VLLM_SERVED_BASE_VARIANT)


def get_vllm_served_base_name() -> str:
    """Name under which vLLM serves the base model (`--served-model-name`).

    Defaults to the served base variant.
    """
    return os.getenv("VLLM_SERVED_BASE_NAME", get_vllm_served_base_variant())


def get_vllm_adapter_key_template() -> str:
    return os.getenv("VLLM_ADAPTER_KEY_TEMPLATE", DEFAULT_VLLM_ADAPTER_KEY_TEMPLATE)


def get_model_cache_dir() -> str:
    return os.getenv("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)
