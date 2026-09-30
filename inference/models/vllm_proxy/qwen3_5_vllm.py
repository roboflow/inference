"""Qwen3.5 VL model class proxying generation to a vLLM sidecar.

Mirrors `inference/models/qwen3_5vl/qwen3_5vl_inference_models.py` (the
in-process HF adapter): the inference server keeps performing
auth/billing/model-resolution/image-preprocessing per request, while
generation runs in the vLLM container (continuous batching + dynamic LoRA).

Preprocessing mirrors `Qwen35HF.pre_process_generation`
(`inference_models/models/qwen3_5/qwen3_5_hf.py`): the same
`<system_prompt>` split semantics and the same pixel budget the HF processor
applies (min 16*32*32 / max 512*32*32 with patch factor 32). Postprocessing
replicates `Qwen35HF.post_process_generation` think-tag parsing so responses
are shape-identical to the HF path.

Shared proxy mechanics live in `qwen_vllm_base.QwenVLVLLMProxy`; this module
holds only the qwen3_5-specific bits.
"""

import re
from typing import Dict, Optional, Tuple, Union

from inference.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    get_adapter_manager,
)
from inference.models.vllm_proxy.qwen_vllm_base import (
    ALLOWED_RESIZE_MODES,
    DEFAULT_PROMPT,
    QwenVLLMProxyBase,
)
from inference.models.vllm_proxy.qwen_vllm_base import (
    smart_resize_dimensions as _smart_resize_dimensions,
)
from inference.models.vllm_proxy.qwen_vllm_base import (
    split_prompt_and_system_prompt as _split_prompt_and_system_prompt,
)
from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Mirrors the AutoProcessor configuration in Qwen35HF.from_pretrained.
IMAGE_PATCH_FACTOR = 32
MIN_PIXELS = 16 * 32 * 32
MAX_PIXELS = 512 * 32 * 32

__all__ = [
    "ALLOWED_RESIZE_MODES",
    "DEFAULT_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "IMAGE_PATCH_FACTOR",
    "MAX_PIXELS",
    "MIN_PIXELS",
    "Qwen35VLLMProxy",
    "post_process_generated_text",
    "smart_resize_dimensions",
    "split_prompt_and_system_prompt",
]


def split_prompt_and_system_prompt(prompt: Optional[str]) -> Tuple[str, str]:
    """Replicates the `<system_prompt>` split from Qwen35HF.pre_process_generation."""
    return _split_prompt_and_system_prompt(
        prompt=prompt,
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
        default_prompt=DEFAULT_PROMPT,
    )


def smart_resize_dimensions(
    height: int,
    width: int,
    factor: int = IMAGE_PATCH_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Computes the (height, width) the Qwen3.5 image processor would resize to."""
    return _smart_resize_dimensions(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def post_process_generated_text(
    text: str, enable_thinking: bool = False
) -> Union[str, Dict[str, str]]:
    """Replicates Qwen35HF.post_process_generation for a single decoded text.

    Cleans common artifacts and parses `<think>...</think>` blocks. When
    `enable_thinking` is set, the opening `<think>` tag is prepended when
    missing (the HF path's generation prompt ends with `<think>\\n`, so the
    tag is absent from generated tokens; vLLM applies the same chat template
    and is expected to behave identically - the guard keeps parsing correct
    either way).
    """
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("assistant\n", "")
    text = text.replace(" addCriterion\n", "")
    if enable_thinking:
        if not text.lstrip().startswith("<think>"):
            text = "<think>" + text
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        else:
            # Model hit max tokens before producing </think>.
            thinking = text.replace("<think>", "").strip()
            answer = ""
        return {"thinking": thinking, "answer": answer}
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


class Qwen35VLLMProxy(QwenVLLMProxyBase):
    """Qwen3.5 VL served via a vLLM sidecar (base model + dynamic LoRA)."""

    image_patch_factor = IMAGE_PATCH_FACTOR
    min_pixels = MIN_PIXELS
    max_pixels = MAX_PIXELS
    default_system_prompt = DEFAULT_SYSTEM_PROMPT
    default_max_new_tokens = INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS
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
