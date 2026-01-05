import json
import os

import torch
from peft import LoraConfig, PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers.utils import is_flash_attn_2_available

from inference.core.env import DEVICE, MODEL_CACHE_DIR

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
from inference.models.transformers import LoRATransformerModel, TransformerModel


class SmolVLM(TransformerModel):
    generation_includes_input = True
    transformers_class = AutoModelForImageTextToText
    load_base_from_roboflow = True
    version_id = None
    default_dtype = torch.bfloat16
    load_weights_as_transformers = True
    endpoint = "smolvlm2/smolvlm-2.2b-instruct"

    def __init__(self, *args, **kwargs):
        if not "model_id" in kwargs:
            kwargs["model_id"] = self.endpoint
        super().__init__(*args, **kwargs)

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        prompt = prompt or "Describe what's in this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_in},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        generated_ids = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=64
        )
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts


class LoRASmolVLM(LoRATransformerModel):
    load_base_from_roboflow = True
    generation_includes_input = True
    skip_special_tokens = True
    transformers_class = AutoModelForImageTextToText
    default_dtype = torch.bfloat16
    use_quantization = True

    def get_lora_base_from_roboflow(self, model_id, revision):
        cache_dir = super().get_lora_base_from_roboflow(model_id, revision)
        return cache_dir

    def initialize_model(self, **kwargs):
        config_file = os.path.join(self.cache_dir, "adapter_config.json")

        with open(config_file, "r") as file:
            config = json.load(file)

        keys_to_remove = [
            "eva_config",
            "corda_config",
            "lora_bias",
            "exclude_modules",
            "trainable_token_indices",
        ]

        for key in keys_to_remove:
            config.pop(key, None)

        with open(config_file, "w") as file:
            json.dump(config, file, indent=2)

        lora_config = LoraConfig(**config)
        model_id = lora_config.base_model_name_or_path
        revision = lora_config.revision
        self.dtype = torch.bfloat16
        model_load_id = self.get_lora_base_from_roboflow(model_id, revision)
        cache_dir = model_load_id
        revision = None
        token = None

        is_smolvlm_256m = "smolvlm-256m" in model_id

        if is_smolvlm_256m:
            rm_weights = os.path.join(
                MODEL_CACHE_DIR, "lora-bases/smolvlm2/smolvlm-256m/main/weights.tar.gz"
            )
        else:
            rm_weights = os.path.join(
                MODEL_CACHE_DIR, "lora-bases/smolvlm2/main/weights.tar.gz"
            )

        if os.path.exists(rm_weights):
            os.remove(rm_weights)

        attn_implementation = (
            "flash_attention_2"
            if (is_flash_attn_2_available() and "cuda" in DEVICE)
            else "eager"
        )

        self.base_model = self.transformers_class.from_pretrained(
            model_load_id,
            revision=revision,
            device_map=DEVICE,
            cache_dir=cache_dir,
            token=token,
            attn_implementation=attn_implementation,
        )

        self.model = (
            PeftModel.from_pretrained(self.base_model, self.cache_dir)
            .eval()
            .to(self.dtype)
        )

        self.model.merge_and_unload()

        if is_smolvlm_256m:
            self.processor = self.processor_class.from_pretrained(
                os.path.join(MODEL_CACHE_DIR, "lora-bases/smolvlm2/smolvlm-256m/main")
            )
        else:
            self.processor = self.processor_class.from_pretrained(
                os.path.join(MODEL_CACHE_DIR, "lora-bases/smolvlm2/main")
            )

    def predict(self, image_in: Image.Image, prompt="", **kwargs):
        prompt = prompt or "Describe what's in this image."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        model_inputs = self.processor(
            text=text_prompt, images=image_in, return_tensors="pt", padding=True
        )

        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(self.model.device)
                if v.dtype != torch.int64 and v.dtype != torch.int32:
                    model_inputs[k] = v.to(self.model.device, dtype=self.dtype)

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        if self.generation_includes_input:
            generation = generation[:, input_len:]

        decoded = self.processor.decode(
            generation[0],
            skip_special_tokens=self.skip_special_tokens,
        )

        parts = decoded.split("Assistant: ")
        if len(parts) > 1:
            decoded = parts[-1].strip()

        return (decoded,)
