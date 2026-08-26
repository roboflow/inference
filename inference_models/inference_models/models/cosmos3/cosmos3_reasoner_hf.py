"""
This is inference-models wrapper for the reasoner tower of NVIDIA Cosmos 3 Edge,
originally published in https://huggingface.co/nvidia/Cosmos3-Edge
"""

import re
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
    RUNNING_ON_JETSON,
)
from inference_models.entities import ColorFormat

DEFAULT_PROMPT = "Describe what's in this image."
SYSTEM_PROMPT_SENTINEL = "<system_prompt>"
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
THINK_EXTRACT_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def _get_cosmos3_attn_implementation(device: torch.device) -> str:
    if (
        is_flash_attn_2_available()
        and device
        and "cuda" in str(device)
        and not RUNNING_ON_JETSON
    ):
        try:
            import flash_attn  # noqa: F401

            if _is_ampere_plus(device=device):
                return "flash_attention_2"
            return "eager"
        except ImportError:
            pass
    return "eager"


def _is_ampere_plus(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device=device)
    return major >= 8


def _resolve_default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class Cosmos3EdgeReasoner:
    """NVIDIA Cosmos 3 Edge reasoner tower (image/video + text -> text).

    Only the autoregressive reasoner is exposed here; the diffusion generator
    (image-to-video, dynamics, policy) has a separate implementation and
    registry entry.
    """

    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        quantization_config: Any = None,
        **kwargs,
    ) -> "Cosmos3EdgeReasoner":
        dtype = _resolve_default_dtype(device)
        attn_implementation = _get_cosmos3_attn_implementation(device)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return cls(model=model, processor=processor, device=device)

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._torch_dtype = next(model.parameters()).dtype
        self.default_system_prompt = (
            "You are Cosmos, a helpful assistant that understands physical scenes "
            "and answers questions about images and videos."
        )
        self._lock = Lock()

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        return_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        inputs = self.pre_process_generation(
            images=images, prompt=prompt, input_color_format=input_color_format
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
            return_thinking=return_thinking,
        )

    def prompt_video(
        self,
        frames: List[np.ndarray],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        return_thinking: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, str]]:
        inputs = self.pre_process_generation(
            images=frames,
            prompt=prompt,
            input_color_format=input_color_format,
            as_video=True,
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
            return_thinking=return_thinking,
        )[0]

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        as_video: bool = False,
        **kwargs,
    ) -> dict:
        if isinstance(images, np.ndarray):
            if input_color_format != "rgb":
                images = images[:, :, ::-1]
            images = images.copy()
        elif as_video and input_color_format != "rgb":
            images = [
                frame[:, :, ::-1].copy() if isinstance(frame, np.ndarray) else frame
                for frame in images
            ]
        prompt, system_prompt = self._parse_prompt(prompt=prompt)
        visual_content = (
            {"type": "video", "video": images}
            if as_video
            else {"type": "image", "image": images}
        )
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    visual_content,
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text_input = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs = {"videos": [images]} if as_video else {"images": images}
        model_inputs = self._processor(
            text=text_input,
            return_tensors="pt",
            padding=True,
            **processor_kwargs,
        )
        return {
            k: (
                v.to(self._device, dtype=self._torch_dtype)
                if v.is_floating_point()
                else v.to(self._device)
            )
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
        }

    def generate(
        self,
        inputs: dict,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS
        input_len = inputs["input_ids"].shape[-1]
        tokenizer = self._processor.tokenizer
        pad_token_id = (
            getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
        )
        with self._lock, torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        return_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )
        result = []
        for text in decoded:
            text = text.replace("assistant\n", "")
            # The chat template opens the reasoning block inside the prompt, so
            # decoded output carries a bare closing </think>. Restore the
            # opening tag so thinking and answer parse apart (qwen3_5 pattern).
            if "</think>" in text and "<think>" not in text:
                text = "<think>" + text
            if return_thinking:
                think_match = THINK_EXTRACT_PATTERN.search(text)
                if think_match:
                    thinking = think_match.group(1).strip()
                    answer = THINK_BLOCK_PATTERN.sub("", text).strip()
                else:
                    thinking = text.replace("<think>", "").strip()
                    answer = ""
                result.append({"thinking": thinking, "answer": answer})
            else:
                result.append(THINK_BLOCK_PATTERN.sub("", text).strip())
        return result

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        return self.prompt(images, **kwargs)

    def _parse_prompt(self, prompt: Optional[str]) -> Tuple[str, str]:
        if prompt is None:
            return DEFAULT_PROMPT, self.default_system_prompt
        split_prompt = prompt.split(SYSTEM_PROMPT_SENTINEL)
        parsed_prompt = split_prompt[0] or DEFAULT_PROMPT
        if len(split_prompt) == 1:
            return parsed_prompt, self.default_system_prompt
        return parsed_prompt, split_prompt[1] or self.default_system_prompt
