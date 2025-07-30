from typing import List, Union
import os

import numpy as np
import torch
from peft import PeftModel
from inference_exp.configuration import DEFAULT_DEVICE
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoModelForCausalLM,
)

AutoModelForCausalLM.register(
    config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration
)


class Qwen25VLHF:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "Qwen25VLHF":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            base_model_path = os.path.join(model_name_or_path, "base")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
        device: torch.device,
        torch_dtype: torch.dtype,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._torch_dtype = torch_dtype
        self.default_system_prompt = (
            "You are a Qwen2.5-VL model that can answer questions about any image."
        )

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        skip_special_tokens: bool = False,
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
        prompt: str = None,
        **kwargs,
    ) -> dict:
        # Handle prompt and system prompt parsing logic from original implementation
        if prompt is None:
            prompt = ""
            system_prompt = self.default_system_prompt
        else:
            split_prompt = prompt.split("<system_prompt>")
            if len(split_prompt) == 1:
                prompt = split_prompt[0]
                system_prompt = self.default_system_prompt
            else:
                prompt = split_prompt[0]
                system_prompt = split_prompt[1]

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
            images=images,
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
        max_new_tokens: int = 512,
        do_sample: bool = False,
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
