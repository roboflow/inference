from typing import List, Union

import numpy as np
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


class PaliGemmaHF:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "PaliGemmaHF":
        # TODO: Add int4/int8 inference
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        return cls(
            model=model, processor=processor, device=device, torch_dtype=torch_dtype
        )

    def __init__(
        self,
        model: PaliGemmaForConditionalGeneration,
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
        max_new_tokens: int = 400,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        inputs = self.pre_process_generation(images=images, prompt=prompt)
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
        **kwargs,
    ) -> dict:
        return self._processor(text=prompt, images=images, return_tensors="pt").to(
            self._device
        )

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 400,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        with torch.inference_mode():
            generation = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
            )
            input_len = inputs["input_ids"].shape[-1]
            return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        return self._processor.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens
        )
