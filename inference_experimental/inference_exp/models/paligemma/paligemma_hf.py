from typing import List, Union
import os

import numpy as np
import torch
from peft import PeftModel
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
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            base_model_path = os.path.join(model_name_or_path, "base")
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
            model = PeftModel.from_pretrained(model, model_name_or_path)
            model.merge_and_unload()
            model.to(device)

            processor = AutoProcessor.from_pretrained(
                base_model_path, trust_remote_code=True, local_files_only=True
            )
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
            processor = AutoProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True, local_files_only=True
            )
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
        num_images = 1
        if isinstance(images, list):
            num_images = len(images)
        elif hasattr(images, "shape") and len(images.shape) == 4:
            num_images = images.shape[0]

        if isinstance(prompt, str) and num_images > 1:
            prompt = [prompt] * num_images
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
