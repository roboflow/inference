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

from typing import Dict, List, Optional, Union

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.entities import ColorFormat
from inference_models.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    get_adapter_manager,
)
from inference_models.models.vllm_proxy.qwen3_5_vllm import (
    DEFAULT_SYSTEM_PROMPT,
    IMAGE_PATCH_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    post_process_generated_text,
)
from inference_models.models.vllm_proxy.qwen_vllm_base import (
    ImagesInput,
    QwenVLLMProxyBase,
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

    @classmethod
    def _get_adapter_manager(cls) -> AdapterManager:
        return get_adapter_manager()

    def prompt(
        self,
        images: ImagesInput,
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Runs one chat completion per image, mirroring `Qwen38HF.prompt`.

        Args:
            images: One image or a list of images.
            prompt: User prompt, optionally with a `<system_prompt>` marker.
            input_color_format: Accepted for parity with the HF class.
            max_new_tokens: Generation cap; the qwen3_8 default when None.
            do_sample: Ignored - the proxy always decodes greedily.
            skip_special_tokens: Ignored - special tokens are stripped.
            enable_thinking: Forwarded to the chat template; splits the
                output into thinking and answer.
            **kwargs: Ignored.

        Returns:
            One entry per image: a string, or a `{"thinking", "answer"}`
            dict when `enable_thinking` is set.
        """
        return super().prompt(
            images,
            prompt=prompt,
            input_color_format=input_color_format,
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
        )

    def post_process_text(
        self, text: str, enable_thinking: Optional[bool] = None
    ) -> Union[str, Dict[str, str]]:
        return post_process_generated_text(
            text=text, enable_thinking=bool(enable_thinking)
        )
