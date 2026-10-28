"""Qwen3.5 VL model class proxying generation to a vLLM sidecar.

Mirrors `Qwen35HF` (`inference_models/models/qwen3_5/qwen3_5_hf.py`): the
same `<system_prompt>` split semantics and the same pixel budget the HF
processor applies (min 16*32*32 / max 512*32*32 with patch factor 32).
Postprocessing replicates `Qwen35HF.post_process_generation` think-tag
parsing so responses are shape-identical to the HF path.

Shared proxy mechanics live in `qwen_vllm_base.QwenVLLMProxyBase`; this
module holds only the qwen3_5-specific bits.
"""

import re
from typing import Dict, List, Optional, Tuple, Union

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS,
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

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

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
    """Replicates the `<system_prompt>` split from Qwen35HF.pre_process_generation.

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
    """Computes the (height, width) the Qwen3.5 image processor would resize to.

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

    Args:
        text: Raw assistant message content.
        enable_thinking: Whether to split thinking from the answer.

    Returns:
        The cleaned text, or a `{"thinking", "answer"}` dict.
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
        enable_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Runs one chat completion per image, mirroring `Qwen35HF.prompt`.

        Args:
            images: One image or a list of images.
            prompt: User prompt, optionally with a `<system_prompt>` marker.
            input_color_format: Accepted for parity with the HF class.
            max_new_tokens: Generation cap; the qwen3_5 default when None.
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
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
        )

    def post_process_text(
        self, text: str, enable_thinking: Optional[bool] = None
    ) -> Union[str, Dict[str, str]]:
        return post_process_generated_text(
            text=text, enable_thinking=bool(enable_thinking)
        )
