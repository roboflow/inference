"""
This is inference-models wrapper for the model originally published in https://github.com/zai-org/GLM-OCR
"""

from threading import Lock
from typing import Any, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
    RUNNING_ON_JETSON,
)
from inference_models.entities import ColorFormat

TEXT_RECOGNITION_PROMPT = "Text Recognition:"
FORMULA_RECOGNITION_PROMPT = "Formula Recognition:"
TABLE_RECOGNITION_PROMPT = "Table Recognition:"


def _get_glm_ocr_attn_implementation(device: torch.device) -> str:
    # TODO: look into jetson builds, as flash-attention compiled for Jetsons does not cooperate
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


class GlmOcrHF:
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

    def recognize_table(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        return self.prompt(
            images=images,
            prompt=FORMULA_RECOGNITION_PROMPT,
            input_color_format=input_color_format,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def recognize_formula(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        return self.prompt(
            images=images,
            prompt=FORMULA_RECOGNITION_PROMPT,
            input_color_format=input_color_format,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def recognize_text(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        return self.prompt(
            images=images,
            prompt=TEXT_RECOGNITION_PROMPT,
            input_color_format=input_color_format,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
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
        **kwargs,
    ) -> dict:
        prompt = prompt or TEXT_RECOGNITION_PROMPT
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
        max_new_tokens: Optional[int] = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS
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

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[str]:
        return self.prompt(images, **kwargs)
