"""Qwen3-VL (instruct) model class proxying generation to a vLLM sidecar.

Mirrors `Qwen3VLHF` (`inference_models/models/qwen3vl/qwen3vl_hf.py`): the
same `<system_prompt>` split semantics and the same pixel budget the HF
processor is configured with (min 256*28*28 / max 1280*28*28; the patch
factor is 32 from the Qwen3-VL checkpoint's preprocessor config: patch_size
16 * merge_size 2). Postprocessing replicates
`Qwen3VLHF.post_process_generation`: plain artifact cleanup only -
qwen3vl-instruct has NO thinking mode, so there is no think-tag parsing and
`<think>` is never prepended.

Shared proxy mechanics live in `qwen_vllm_base.QwenVLLMProxyBase`; this
module holds only the qwen3vl-specific bits.
"""

from typing import List, Optional, Tuple

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.entities import ColorFormat
from inference_models.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    get_adapter_manager,
)
from inference_models.models.vllm_proxy.qwen_vllm_base import (
    ALLOWED_RESIZE_MODES,
    DEFAULT_PROMPT,
    ImagesInput,
    QwenVLLMProxyBase,
)
from inference_models.models.vllm_proxy.qwen_vllm_base import (
    smart_resize_dimensions as _smart_resize_dimensions,
)
from inference_models.models.vllm_proxy.qwen_vllm_base import (
    split_prompt_and_system_prompt as _split_prompt_and_system_prompt,
)

DEFAULT_SYSTEM_PROMPT = "You are a Qwen3-VL a helpful assistant for any visual task."

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
    """Replicates the `<system_prompt>` split from Qwen3VLHF.pre_process_generation.

    Args:
        prompt: Raw prompt, optionally carrying a `<system_prompt>` marker.

    Returns:
        `(user_prompt, system_prompt)`.
    """
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
    """Computes the (height, width) the Qwen3-VL image processor would resize to.

    Args:
        height: Source image height.
        width: Source image width.
        factor: Patch factor both dimensions must be divisible by.
        min_pixels: Lower pixel budget.
        max_pixels: Upper pixel budget.

    Returns:
        `(height, width)` after the smart resize.
    """
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
    replacements below mirror that decode behaviour for the vLLM response.
    There is NO thinking mode for qwen3vl-instruct: `<think>` tags are
    neither prepended nor parsed.

    Args:
        text: Raw assistant message content.

    Returns:
        The cleaned text.
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
    supports_thinking = False

    @classmethod
    def _get_adapter_manager(cls) -> AdapterManager:
        return get_adapter_manager()

    def prompt(
        self,
        images: ImagesInput,
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        """Runs one chat completion per image, mirroring `Qwen3VLHF.prompt`.

        Args:
            images: One image or a list of images.
            prompt: User prompt, optionally with a `<system_prompt>` marker.
            input_color_format: Accepted for parity with the HF class.
            max_new_tokens: Generation cap; the qwen3vl default when None.
            do_sample: Ignored - the proxy always decodes greedily.
            skip_special_tokens: Ignored - special tokens are stripped.
            **kwargs: Ignored (there is no thinking mode).

        Returns:
            One string per image.
        """
        return super().prompt(
            images,
            prompt=prompt,
            input_color_format=input_color_format,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens,
        )

    def post_process_text(
        self, text: str, enable_thinking: Optional[bool] = None
    ) -> str:
        return post_process_generated_text(text=text)
