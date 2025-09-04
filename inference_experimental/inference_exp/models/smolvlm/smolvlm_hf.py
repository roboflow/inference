import os
from typing import List, Optional, Union

import numpy as np
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from peft import PeftModel
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

        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):

            base_model_path = os.path.join(model_name_or_path, "base")
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
            model = PeftModel.from_pretrained(model, model_name_or_path)
            model.merge_and_unload()
            model.to(device)

            processor = AutoProcessor.from_pretrained(
                base_model_path,
                padding_side="left",
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            print("smolvlm_hf.from_pretrained", "no adapter_config.json")
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                padding_side="left",
                trust_remote_code=True,
                local_files_only=True,
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
        def _to_tensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            is_numpy = isinstance(image, np.ndarray)
            if is_numpy:
                tensor_image = torch.from_numpy(image.copy()).permute(2, 0, 1)
            else:
                tensor_image = image
            if input_color_format == "bgr" or (is_numpy and input_color_format is None):
                tensor_image = tensor_image[[2, 1, 0], :, :]
            return tensor_image

        if isinstance(images, torch.Tensor) and images.ndim > 3:
            image_list = [_to_tensor(img) for img in images]
        elif not isinstance(images, list):
            image_list = [_to_tensor(images)]
        else:
            image_list = [_to_tensor(img) for img in images]

        if images_to_single_prompt:
            content = [{"type": "image"}] * len(image_list)
            content.append({"type": "text", "text": prompt})
            conversations = [[{"role": "user", "content": content}]]
        else:
            conversations = []
            for _ in image_list:
                conversations.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                )
        text_prompts = self._processor.apply_chat_template(
            conversations, add_generation_prompt=True
        )
        inputs = self._processor(
            text=text_prompts, images=image_list, return_tensors="pt", padding=True
        )
        return inputs.to(self._device, dtype=self._torch_dtype)

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
