from typing import List, Optional, Union

import numpy as np
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.models.common.roboflow.pre_processing import images_to_pillow
from transformers import AutoModelForImageTextToText, AutoProcessor


class SmolVLMHF:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "SmolVLMHF":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        return cls(
            model=model, processor=processor, device=device, torch_dtype=torch_dtype
        )

    def __init__(
        self,
        model: AutoModelForImageTextToText,
        processor: AutoProcessor,
        device: torch.device,
        torch_dtype: torch.dtype,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._torch_dtype = torch_dtype

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str,
        images_to_single_prompt: bool = True,
        input_color_format: Optional[ColorFormat] = None,
        max_new_tokens: int = 400,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        inputs = self.pre_process_generation(
            images=images,
            prompt=prompt,
            images_to_single_prompt=images_to_single_prompt,
            input_color_format=input_color_format,
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
        prompt: str,
        images_to_single_prompt: bool = True,
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> dict:
        messages = prepare_chat_messages(
            images=images,
            prompt=prompt,
            images_to_single_prompt=images_to_single_prompt,
            input_color_format=input_color_format,
        )
        return self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=len(messages) > 1,
        ).to(self._device, dtype=self._torch_dtype)

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 400,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        generation = self._model.generate(
            **inputs, do_sample=do_sample, max_new_tokens=max_new_tokens
        )
        input_len = inputs["input_ids"].shape[-1]
        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        decoded = self._processor.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens
        )
        return [result.strip() for result in decoded]


def prepare_chat_messages(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    prompt: str,
    images_to_single_prompt: bool,
    input_color_format: Optional[ColorFormat] = None,
) -> List[List[dict]]:
    pillow_images, _ = images_to_pillow(
        images=images, input_color_format=input_color_format, model_color_format="rgb"
    )
    if images_to_single_prompt:
        content = []
        for image in pillow_images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        return [
            [
                {
                    "role": "user",
                    "content": content,
                },
            ]
        ]
    result = []
    for image in pillow_images:
        result.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        )
    return result
