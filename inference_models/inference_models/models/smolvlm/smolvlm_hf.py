import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)


class SmolVLMHF:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        disable_quantization: bool = False,
        **kwargs,
    ) -> "SmolVLMHF":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        inference_config_path = os.path.join(
            model_name_or_path, "inference_config.json"
        )
        inference_config = None
        if os.path.exists(inference_config_path):
            inference_config = parse_inference_config(
                config_path=inference_config_path,
                allowed_resize_modes={
                    ResizeMode.STRETCH_TO,
                    ResizeMode.LETTERBOX,
                    ResizeMode.CENTER_CROP,
                    ResizeMode.LETTERBOX_REFLECT_EDGES,
                },
            )
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if (
            quantization_config is None
            and device.type == "cuda"
            and not disable_quantization
        ):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        if os.path.exists(adapter_config_path):

            base_model_path = os.path.join(model_name_or_path, "base")
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
            )
            model = PeftModel.from_pretrained(model, model_name_or_path)
            if quantization_config is None:
                model.merge_and_unload()
            model.to(device)

            processor = AutoProcessor.from_pretrained(
                base_model_path,
                padding_side="left",
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                use_fast=True,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                dtype=torch_dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
            ).eval()
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                padding_side="left",
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                use_fast=True,
            )
        return cls(
            model=model,
            processor=processor,
            inference_config=inference_config,
            device=device,
            torch_dtype=torch_dtype,
        )

    def __init__(
        self,
        model: AutoModelForImageTextToText,
        processor: AutoProcessor,
        inference_config: Optional[InferenceConfig],
        device: torch.device,
        torch_dtype: torch.dtype,
    ):
        self._model = model
        self._processor = processor
        self._inference_config = inference_config
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
        image_size: Optional[Tuple[int, int]] = None,
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
            if image_size is not None:
                tensor_image = torch.nn.functional.interpolate(
                    image,
                    [image_size[1], image_size[0]],
                    mode="bilinear",
                )
            return tensor_image

        if self._inference_config is None:
            if isinstance(images, torch.Tensor) and images.ndim > 3:
                image_list = [_to_tensor(img) for img in images]
            elif not isinstance(images, list):
                image_list = [_to_tensor(images)]
            else:
                image_list = [_to_tensor(img) for img in images]
        else:
            images = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                image_size_wh=image_size,
            )[0]
            image_list = [e[0] for e in torch.split(images, 1, dim=0)]
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
        max_image_size = None
        if image_size:
            max_image_size = {"longest_edge": max(image_size[0], image_size[1])}

        inputs = self._processor(
            text=text_prompts,
            images=image_list,
            return_tensors="pt",
            padding=True,
            max_image_size=max_image_size,
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
