"""Qwen3-VL (instruct) model class proxying generation to a vLLM sidecar.

Mirrors `inference/models/qwen3vl/qwen3vl_inference_models.py` (the
in-process HF adapter): the inference server keeps performing
auth/billing/model-resolution/image-preprocessing per request, while
generation runs in the vLLM container (continuous batching + dynamic LoRA).

Preprocessing mirrors `Qwen3VLHF.pre_process_generation`
(`inference_models/models/qwen3vl/qwen3vl_hf.py`): the same
`<system_prompt>` split semantics and the same pixel budget the HF processor
is configured with (min 256*28*28 / max 1280*28*28; the patch factor is 32
from the Qwen3-VL checkpoint's preprocessor config: patch_size 16 *
merge_size 2). Postprocessing replicates `Qwen3VLHF.post_process_generation`:
plain artifact cleanup only - qwen3vl-instruct has NO thinking mode, so there
is no think-tag parsing and `<think>` is never prepended.

Shared proxy mechanics live in `qwen_vllm_base.QwenVLLMProxyBase`; this
module holds only the qwen3vl-specific bits.
"""

from typing import Optional, Tuple

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
    INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS,
)

# Mirrors Qwen3VLHF.__init__.
DEFAULT_SYSTEM_PROMPT = "You are a Qwen3-VL a helpful assistant for any visual task."

# Pixel budget mirrors the AutoProcessor configuration in
# Qwen3VLHF.from_pretrained (min_pixels=256*28*28, max_pixels=1280*28*28).
# The patch factor comes from the Qwen3-VL checkpoint's preprocessor config
# (patch_size=16, merge_size=2 -> 32), not from the 28-based pixel-budget
# constants the HF adapter inherited from the Qwen2.5-VL defaults.
IMAGE_PATCH_FACTOR = 32
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

__all__ = [
    "ALLOWED_RESIZE_MODES",
    "DEFAULT_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "IMAGE_PATCH_FACTOR",
    "MAX_PIXELS",
    "MIN_PIXELS",
    "Qwen3VLVLLMProxy",
    "post_process_generated_text",
    "smart_resize_dimensions",
    "split_prompt_and_system_prompt",
]


def split_prompt_and_system_prompt(prompt: Optional[str]) -> Tuple[str, str]:
    """Replicates the `<system_prompt>` split from Qwen3VLHF.pre_process_generation."""
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
    """Computes the (height, width) the Qwen3-VL image processor would resize to."""
    return _smart_resize_dimensions(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def post_process_generated_text(text: str) -> str:
    """Replicates Qwen3VLHF.post_process_generation for a single decoded text.

    The HF path decodes with `skip_special_tokens=True` and only cleans the
    `assistant\\n` / ` addCriterion\\n` artifacts; the special-token
    replacements below mirror that decode behaviour for the vLLM response
    (which normally carries no special tokens either). There is NO thinking
    mode for qwen3vl-instruct: `<think>` tags are neither prepended nor
    parsed - any such text is returned verbatim, exactly like the HF path.
    """
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("assistant\n", "")
    text = text.replace(" addCriterion\n", "")
    return text.strip()


class Qwen3VLVLLMProxy(QwenVLLMProxyBase):
    """Qwen3-VL served via a vLLM sidecar (base model + dynamic LoRA)."""

    image_patch_factor = IMAGE_PATCH_FACTOR
    min_pixels = MIN_PIXELS
    max_pixels = MAX_PIXELS
    default_system_prompt = DEFAULT_SYSTEM_PROMPT
    default_max_new_tokens = INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS
    # qwen3vl-instruct has no thinking mode - `enable_thinking` is never
    # forwarded to the chat template.
    supports_thinking = False

    def _get_adapter_manager(self) -> AdapterManager:
        # Module-level lookup keeps `get_adapter_manager` patchable on this
        # module in tests.
        return get_adapter_manager()

    def post_process_text(self, text: str, **kwargs) -> str:
        return post_process_generated_text(text=text)
