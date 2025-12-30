import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

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


class PaliGemmaHF:

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
    ) -> "PaliGemmaHF":
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
        if (
            quantization_config is None
            and device.type == "cuda"
            and not disable_quantization
        ):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            base_model_path = os.path.join(model_name_or_path, "base")
            model = PaliGemmaForConditionalGeneration.from_pretrained(
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
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                use_fast=True,
            )
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                dtype=torch_dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
            ).eval()
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
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
        model: PaliGemmaForConditionalGeneration,
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
        input_color_format: Optional[ColorFormat] = None,
        max_new_tokens: int = 400,
        do_sample: bool = False,
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
        prompt: str,
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
        num_images = len(image_list)

        if isinstance(prompt, str) and num_images > 1:
            prompt = [prompt] * num_images
        return self._processor(text=prompt, images=image_list, return_tensors="pt").to(
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
