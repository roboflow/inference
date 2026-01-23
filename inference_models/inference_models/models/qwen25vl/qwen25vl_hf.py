import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from transformers.utils import is_flash_attn_2_available

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS,
)
from inference_models.entities import ColorFormat
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)


class Qwen25VLHF:
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
    ) -> "Qwen25VLHF":
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
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
                    ResizeMode.FIT_LONGER_EDGE,
                },
            )
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

        attn_implementation = (
            "flash_attention_2"
            if (is_flash_attn_2_available() and device and "cuda" in str(device))
            else "eager"
        )

        if os.path.exists(adapter_config_path):
            base_model_path = os.path.join(model_name_or_path, "base")
            _patch_preprocessor_config(cache_dir=base_model_path)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path,
                dtype="auto",
                device_map=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
            )
            _patch_preprocessor_config(cache_dir=model_name_or_path)
            model = PeftModel.from_pretrained(model, model_name_or_path)
            if quantization_config is None:
                model.merge_and_unload()
                model.to(device)
            processor = Qwen2_5_VLProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                use_fast=True,
            )
        else:
            _patch_preprocessor_config(cache_dir=model_name_or_path)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                dtype="auto",
                device_map=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
            ).eval()
            Qwen2_5_VLProcessor.image_processor_class = "Qwen2VLImageProcessor"
            processor = Qwen2_5_VLProcessor.from_pretrained(
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
        )

    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: Qwen2_5_VLProcessor,
        inference_config: Optional[InferenceConfig],
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._inference_config = inference_config
        self._device = device
        self.default_system_prompt = (
            "You are a Qwen2.5-VL model that can answer questions about any image."
        )

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: int = INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS,
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
        # Handle prompt and system prompt parsing logic from original implementation
        if prompt is None:
            prompt = "Describe what's in this image."
            system_prompt = self.default_system_prompt
        else:
            split_prompt = prompt.split("<system_prompt>")
            if len(split_prompt) == 1:
                prompt = split_prompt[0] or "Describe what's in this image."
                system_prompt = self.default_system_prompt
            else:
                prompt = split_prompt[0] or "Describe what's in this image."
                system_prompt = split_prompt[1] or self.default_system_prompt

        # Construct conversation following original implementation structure
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Processor will handle the actual image
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template
        text_input = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # Process inputs - processor will handle tensor/array inputs directly
        model_inputs = self._processor(
            text=text_input,
            images=image_list,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to device
        model_inputs = {
            k: v.to(self._device)
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
        }

        return model_inputs

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                bos_token_id=self._processor.tokenizer.bos_token_id,
            )

        # Return only the newly generated tokens
        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        # Decode the generated tokens
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )

        # Apply the same post-processing as original implementation
        result = []
        for text in decoded:
            text = text.replace("assistant\n", "")
            text = text.replace(" addCriterion\n", "")
            result.append(text.strip())

        return result


def adjust_lora_model_state_dict(state_dict: dict) -> dict:
    return {
        refactor_adapter_weights_key(key=key): value
        for key, value in state_dict.items()
    }


def refactor_adapter_weights_key(key: str) -> str:
    if ".language_model." in key:
        return key
    return (
        key.replace("model.layers", "model.language_model.layers")
        .replace(".weight", ".default.weight")
        .replace(".lora_magnitude_vector", ".lora_magnitude_vector.default.weight")
    )


def _patch_preprocessor_config(cache_dir: str):
    """
    Checks and patches the preprocessor_config.json in the given cache directory
    to ensure the image_processor_type is recognized.
    """
    config_path = os.path.join(cache_dir, "preprocessor_config.json")
    target_key = "image_processor_type"
    correct_value = "Qwen2VLImageProcessor"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Preprocessor config not found at {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    if target_key in data and data[target_key] != correct_value:
        data[target_key] = correct_value
        with open(config_path, "w") as f:
            json.dump(data, f, indent=4)
    elif target_key in data:
        pass
    else:
        raise ValueError(f"'{target_key}' not found in {config_path}")
