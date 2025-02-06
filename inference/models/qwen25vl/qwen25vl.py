import os
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from inference.models.florence2.utils import import_class_from_file

from inference.core.entities.responses.inference import LMMInferenceResponse
from inference.core.models.types import PreprocessReturnMetadata
from inference.models.transformers import TransformerModel, LoRATransformerModel
from inference.core.env import MODEL_CACHE_DIR, DEVICE
from peft import LoraConfig, PeftModel
import json
import transformers

class Qwen25VL(TransformerModel):
    generation_includes_input = True
    transformers_class = AutoModelForCausalLM
    default_dtype = torch.float32
    skip_special_tokens = False

    system_message = "You are a Qwen2.5-VL model that can answer questions about any image."

    def initialize_model(self):
        self.transformers_class = import_class_from_file(
            os.path.join(self.cache_dir, "modeling_qwen2_5_vl.py"),
            "Qwen2_5_VLForConditionalGeneration",
        )
        self.processor_class = import_class_from_file(
            os.path.join(self.cache_dir, "processing_qwen2_5_vl.py"),
            "Qwen2_5_VLProcessor",
        )
        self.image_processor_class = import_class_from_file(
            os.path.join(self.cache_dir, "image_processing_qwen2_5_vl.py"),
            "Qwen2_5_VLImageProcessor",
            "transformers.Qwen2_5_VLImageProcessor"
        )
        transformers.Qwen2_5_VLImageProcessor = self.image_processor_class
        config_file = os.path.join(self.cache_dir, "adapter_config.json")

        with open(config_file, 'r') as file:
            config = json.load(file)

        keys_to_remove = ['eva_config', 'lora_bias', 'exclude_modules']

        for key in keys_to_remove:
            config.pop(key, None)

        with open(config_file, 'w') as file:
            json.dump(config, file, indent=2)

        lora_config = LoraConfig(**config)
        model_id = lora_config.base_model_name_or_path
        revision = lora_config.revision
        if revision is not None:
            try:
                self.dtype = getattr(torch, revision)
            except AttributeError:
                pass
        if not self.load_base_from_roboflow:
            model_load_id = model_id
            cache_dir = os.path.join(MODEL_CACHE_DIR, "huggingface")
            revision = revision
            token = self.huggingface_token
        else:
            model_load_id = self.get_lora_base_from_roboflow(model_id, revision)
            cache_dir = model_load_id
            revision = None
            token = None
        self.base_model = self.transformers_class.from_pretrained(
            model_load_id,
            revision=revision,
            device_map=DEVICE,
            cache_dir=cache_dir,
            token=token,
        ).to(self.dtype)
        self.model = (
            PeftModel.from_pretrained(self.base_model, self.cache_dir)
            .eval()
            .to(self.dtype)
        )

        self.model.merge_and_unload()
        preprocessor_config_path = os.path.join(self.cache_dir, "chat_template.json")
        with open(preprocessor_config_path, "r") as f:
            chat_template = json.load(f)["chat_template"]

        self.processor = self.processor_class.from_pretrained(
            model_load_id, revision=revision, cache_dir=cache_dir, token=token, chat_template=chat_template
        )

    def predict(self, image_in: Image.Image, prompt="", **kwargs):
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_in},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text_input = self.processor.apply_chat_template(conversation, tokenize=False)

        model_inputs = self.processor(
            text=text_input,
            images=image_in,
            return_tensors="pt",
            padding=True,
        )

        model_inputs = {
            k: v.to(self.model.device)
            for k, v in model_inputs.items() if isinstance(v, torch.Tensor)
        }

        input_len = model_inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
            )

        if not self.generation_includes_input:
            generation = generation[:, input_len:]
        
        decoded = self.processor.decode(
            generation[0],
            skip_special_tokens=self.skip_special_tokens,
        )

        decoded = decoded.replace("assistant\n", "")
        
        return (decoded,)


class LoRAQwen25VL(LoRATransformerModel):
    load_base_from_roboflow = True
    generation_includes_input = True
    skip_special_tokens = True
    transformers_class = AutoModelForCausalLM
    default_dtype = torch.float32

    system_message = "You are a Qwen2.5-VL model that can answer questions about any image."

    def get_lora_base_from_roboflow(self, model_id, revision):
        cache_dir = super().get_lora_base_from_roboflow(model_id, revision)
        return cache_dir
    
    def initialize_model(self):
        self.transformers_class = import_class_from_file(
            os.path.join(self.cache_dir, "modeling_qwen2_5_vl.py"),
            "Qwen2_5_VLForConditionalGeneration",
        )
        print("ABOUT TO IMPORT PROCESSOR")
        self.processor_class = import_class_from_file(
            os.path.join(self.cache_dir, "processing_qwen2_5_vl.py"),
            "Qwen2_5_VLProcessor",
        )
        self.image_processor_class = import_class_from_file(
            os.path.join(self.cache_dir, "image_processing_qwen2_5_vl.py"),
            "Qwen2_5_VLImageProcessor",
            "transformers.Qwen2_5_VLImageProcessor"
        )
        
        transformers.Qwen2_5_VLImageProcessor = self.image_processor_class

        config_file = os.path.join(self.cache_dir, "adapter_config.json")

        with open(config_file, 'r') as file:
            config = json.load(file)

        keys_to_remove = ['eva_config', 'lora_bias', 'exclude_modules']

        for key in keys_to_remove:
            config.pop(key, None)

        with open(config_file, 'w') as file:
            json.dump(config, file, indent=2)
        
        lora_config = LoraConfig(**config)
        model_id = lora_config.base_model_name_or_path
        revision = lora_config.revision
        if revision is not None:
            try:
                self.dtype = getattr(torch, revision)
            except AttributeError:
                pass
        if not self.load_base_from_roboflow:
            model_load_id = model_id
            cache_dir = os.path.join(MODEL_CACHE_DIR, "huggingface")
            revision = revision
            token = self.huggingface_token
        else:
            model_load_id = self.get_lora_base_from_roboflow(model_id, revision)
            cache_dir = model_load_id
            revision = None
            token = None
        self.base_model = self.transformers_class.from_pretrained(
            model_load_id,
            revision=revision,
            device_map=DEVICE,
            cache_dir=cache_dir,
            token=token,
        ).to(self.dtype)

        self.model = (
            PeftModel.from_pretrained(self.base_model, self.cache_dir)
            .eval()
            .to(self.dtype)
        )

        self.model.merge_and_unload()
        preprocessor_config_path = os.path.join(self.cache_dir, "chat_template.json")
        with open(preprocessor_config_path, "r") as f:
            chat_template = json.load(f)["chat_template"]

        self.processor = self.processor_class.from_pretrained(
            model_load_id, revision=revision, cache_dir=cache_dir, token=token, chat_template=chat_template
        )

    def predict(self, image_in: Image.Image, prompt=""):
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_in},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text_input = self.processor.apply_chat_template(conversation, tokenize=False)

        model_inputs = self.processor(
            text=text_input,
            images=image_in,
            return_tensors="pt",
            padding=True,
        )

        model_inputs = {
            k: v.to(self.model.device)
            for k, v in model_inputs.items() if isinstance(v, torch.Tensor)
        }

        input_len = model_inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
            )

        if self.generation_includes_input:
            generation = generation[:, input_len:]
        
        decoded = self.processor.decode(
            generation[0],
            skip_special_tokens=self.skip_special_tokens,
        )

        decoded = decoded.replace("assistant\n", "")
        
        return (decoded,)
