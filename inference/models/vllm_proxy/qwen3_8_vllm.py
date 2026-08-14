"""Qwen3.8 VL model class proxying generation to a vLLM sidecar.

Qwen3.8 ships the qwen3_5 architecture, so preprocessing/postprocessing are
identical to `qwen3_5_vllm` (same `<system_prompt>` split, same pixel budget,
same think-tag parsing) and are reused from that module - mirroring how
`Qwen38HF` subclasses `Qwen35HF` on the in-process path.

Base-model serving only: Qwen3.8 has no fine-tuning support, so `"qwen3_8"`
is deliberately NOT added to `adapter_manager.SUPPORTED_MODEL_ARCHITECTURES`.
Requests for the served base variant short-circuit in
`AdapterManager.resolve_and_register` before that gate, while any qwen3_8
fine-tune adapter request is rejected pre-download with
`NotServableOnVLLMError`.
"""

from typing import Dict, Union

from inference.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    get_adapter_manager,
)
from inference.models.vllm_proxy.qwen3_5_vllm import (
    DEFAULT_SYSTEM_PROMPT,
    IMAGE_PATCH_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    post_process_generated_text,
)
from inference.models.vllm_proxy.qwen_vllm_base import QwenVLLMProxyBase
from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS,
)

__all__ = ["Qwen38VLLMProxy"]


class Qwen38VLLMProxy(QwenVLLMProxyBase):
    """Qwen3.8 VL served via a vLLM sidecar (base model only, no LoRA)."""

    image_patch_factor = IMAGE_PATCH_FACTOR
    min_pixels = MIN_PIXELS
    max_pixels = MAX_PIXELS
    default_system_prompt = DEFAULT_SYSTEM_PROMPT
    default_max_new_tokens = INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
    supports_thinking = True

    def _get_adapter_manager(self) -> AdapterManager:
        # Module-level lookup keeps `get_adapter_manager` patchable on this
        # module in tests.
        return get_adapter_manager()

    def post_process_text(self, text: str, **kwargs) -> Union[str, Dict[str, str]]:
        return post_process_generated_text(
            text=text,
            enable_thinking=kwargs.get("enable_thinking", False),
        )
