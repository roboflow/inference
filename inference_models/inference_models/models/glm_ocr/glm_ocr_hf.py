from threading import Lock
from typing import List, Union

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.entities import ColorFormat


def _get_glm_ocr_attn_implementation(device: torch.device) -> str:
    if is_flash_attn_2_available() and device and "cuda" in str(device):
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


class GlmOcrHF:
    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        quantization_config=None,
        disable_quantization: bool = False,
        **kwargs,
    ) -> "GlmOcrHF":
        dtype = cls.default_dtype
        attn_implementation = _get_glm_ocr_attn_implementation(device)

        model = (
            AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                device_map=device,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
            )
            .eval()
            .to(dtype)
        )

        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        return cls(
            model=model,
            processor=processor,
            device=device,
        )

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._lock = Lock()

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: int = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
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
        )

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        **kwargs,
    ) -> dict:
        prompt = prompt or "Text Recognition:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self._device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

        return inputs

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        input_len = inputs["input_ids"].shape[-1]

        with self._lock, torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )

        return [text.strip() for text in decoded]
