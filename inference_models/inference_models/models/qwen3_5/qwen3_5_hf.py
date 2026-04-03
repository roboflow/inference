import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_QWEN3_5_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.entities import ColorFormat
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.qwen3vl.qwen3vl_hf import _get_qwen3vl_attn_implementation


class Qwen35HF:
    default_dtype = torch.bfloat16

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
    ) -> "Qwen35HF":
        load_weights = kwargs.pop("load_weights", True)
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

        attn_implementation = _get_qwen3vl_attn_implementation(device)

        if load_weights:
            if os.path.exists(adapter_config_path):
                # Has adapter - load base model then apply LoRA
                base_model_path = os.path.join(model_name_or_path, "base")
                base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    base_model_path,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    quantization_config=quantization_config,
                    attn_implementation=attn_implementation,
                )
                # Apply LoRA, eval, convert to dtype
                model = (
                    PeftModel.from_pretrained(base_model, model_name_or_path)
                    .eval()
                    .to(cls.default_dtype)
                )
                model = model.merge_and_unload()
            else:
                # No adapter - just load base model, eval, convert to dtype
                base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    quantization_config=quantization_config,
                    attn_implementation=attn_implementation,
                )
                model = base_model.eval().to(cls.default_dtype)
        else:
            model = None

        # Load processor with chat_template if available
        # Check both root and base/ directory for chat_template.jinja
        chat_template_path = os.path.join(model_name_or_path, "chat_template.jinja")
        if not os.path.exists(chat_template_path):
            chat_template_path = os.path.join(
                model_name_or_path, "base", "chat_template.jinja"
            )

        # Use base/ for processor when adapter exists - preprocessor configs differ
        if os.path.exists(adapter_config_path):
            processor_path = os.path.join(model_name_or_path, "base")
        else:
            processor_path = model_name_or_path

        if os.path.exists(chat_template_path):
            with open(chat_template_path, "r") as f:
                chat_template = f.read()
            processor = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                chat_template=chat_template,
                min_pixels=16 * 32 * 32,
                max_pixels=512 * 32 * 32,
            )
        else:
            processor = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                min_pixels=16 * 32 * 32,
                max_pixels=512 * 32 * 32,
            )

        return cls(
            model=model,
            processor=processor,
            inference_config=inference_config,
            device=device,
        )

    def __init__(
        self,
        model: Qwen3_5ForConditionalGeneration,
        processor: AutoProcessor,
        inference_config: Optional[InferenceConfig],
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._inference_config = inference_config
        self._device = device
        self.default_system_prompt = "You are a helpful assistant."
        self._lock = Lock()

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        inputs = self.pre_process_generation(
            images=images,
            prompt=prompt,
            input_color_format=input_color_format,
            enable_thinking=enable_thinking,
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
        )

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        image_size: Optional[Tuple[int, int]] = None,
        enable_thinking: bool = False,
        **kwargs,
    ) -> dict:
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

        # Construct conversation following qwen3vl inference pattern
        # Pass the actual image in the conversation for proper vision token handling
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # processor.__call__() doesn't forward enable_thinking to the Jinja
        # template, so we call apply_chat_template separately then tokenize.
        text = self._processor.apply_chat_template(
            [conversation],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if isinstance(text, list):
            text = text[0]

        model_inputs = self._processor(
            text=[text],
            images=[images],
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
        max_new_tokens: Optional[int] = None,
        do_sample: bool = INFERENCE_MODELS_QWEN3_5_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS
        input_len = inputs["input_ids"].shape[-1]

        with self._lock, torch.inference_mode():
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
        enable_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        # Decode the generated tokens
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )

        # Apply post-processing - clean up think tags and other artifacts
        result = []
        for text in decoded:
            # Clean common artifacts from all outputs
            text = text.replace("<|im_end|>", "")
            text = text.replace("<|endoftext|>", "")
            text = text.replace("assistant\n", "")
            text = text.replace(" addCriterion\n", "")

            if enable_thinking:
                # The generated output only contains NEW tokens (input is
                # trimmed in generate()). Since the input ends with "<think>\n",
                # the opening <think> tag is NOT in the decoded output.
                # Prepend it so the regex can parse thinking vs answer.
                text = "<think>" + text
                # Extract thinking and answer separately
                think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    answer = re.sub(
                        r"<think>.*?</think>\s*", "", text, flags=re.DOTALL
                    ).strip()
                else:
                    # Model hit max tokens before producing </think>.
                    # Everything after <think> is thinking, no answer yet.
                    thinking = text.replace("<think>", "").strip()
                    answer = ""
                result.append({"thinking": thinking, "answer": answer})
            else:
                # Remove <think>...</think> blocks (Qwen3.5 reasoning format)
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
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
