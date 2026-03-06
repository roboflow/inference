import json
import os
import tarfile

import torch
from peft import LoraConfig, PeftModel
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from transformers.utils import is_flash_attn_2_available

from inference.core.cache.model_artifacts import get_cache_dir, get_cache_file_path
from inference.core.env import DEVICE, HUGGINGFACE_TOKEN, MODEL_CACHE_DIR
from inference.core.roboflow_api import get_roboflow_base_lora, stream_url_to_cache
from inference.models.transformers import LoRATransformerModel, TransformerModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.bfloat16,
)


class Qwen3VL(TransformerModel):
    generation_includes_input = True
    transformers_class = Qwen3VLForConditionalGeneration
    processor_class = AutoProcessor
    default_dtype = torch.bfloat16
    skip_special_tokens = True

    default_system_prompt = (
        "You are a Qwen3-VL model that can answer questions about any image."
    )

    def __init__(
        self,
        model_id,
        *args,
        dtype=None,
        huggingface_token=HUGGINGFACE_TOKEN,
        use_quantization=False,
        **kwargs,
    ):
        self.use_quantization = use_quantization
        super().__init__(model_id, *args, **kwargs)

    def initialize_model(self, **kwargs):
        config_file = os.path.join(self.cache_dir, "adapter_config.json")

        with open(config_file, "r") as file:
            config = json.load(file)

        keys_to_remove = ["eva_config", "lora_bias", "exclude_modules"]

        for key in keys_to_remove:
            config.pop(key, None)

        with open(config_file, "w") as file:
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

        attn_implementation = (
            "flash_attention_2"
            if (is_flash_attn_2_available() and DEVICE and "cuda" in DEVICE)
            else "eager"
        )

        if self.use_quantization:
            self.base_model = self.transformers_class.from_pretrained(
                model_load_id,
                revision=revision,
                device_map=DEVICE,
                cache_dir=cache_dir,
                token=token,
                quantization_config=bnb_config,
                attn_implementation=attn_implementation,
            )
        else:
            self.base_model = self.transformers_class.from_pretrained(
                model_load_id,
                revision=revision,
                device_map=DEVICE,
                cache_dir=cache_dir,
                token=token,
                attn_implementation=attn_implementation,
            )
        self.model = self.base_model.eval().to(self.dtype)

        self.processor = self.processor_class.from_pretrained(
            model_load_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

    def get_lora_base_from_roboflow(self, repo, revision) -> str:
        base_dir = os.path.join("lora-bases", repo, revision)
        cache_dir = get_cache_dir(base_dir)
        if os.path.exists(cache_dir):
            return cache_dir
        api_data = get_roboflow_base_lora(self.api_key, repo, revision, self.device_id)
        if "weights" not in api_data:
            raise RuntimeError(
                "`weights` key not available in Roboflow API response while downloading model weights."
            )

        weights_url = api_data["weights"]["model"]
        filename = weights_url.split("?")[0].split("/")[-1]
        assert filename.endswith("tar.gz")
        stream_url_to_cache(
            url=weights_url,
            filename=filename,
            model_id=base_dir,
        )
        tar_file_path = get_cache_file_path(filename, base_dir)
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)

        return cache_dir

    def predict(self, image_in: Image.Image, prompt=None, **kwargs):
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

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
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
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
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
        decoded = decoded.replace(" addCriterion\n", "")
        return (decoded,)


class LoRAQwen3VL(LoRATransformerModel):
    load_base_from_roboflow = True
    generation_includes_input = True
    skip_special_tokens = True
    transformers_class = Qwen3VLForConditionalGeneration
    processor_class = AutoProcessor
    default_dtype = torch.bfloat16
    use_quantization = False

    default_system_prompt = (
        "You are a Qwen3-VL a helpful assistant for any visual task."
    )

    def get_lora_base_from_roboflow(self, model_id, revision):
        cache_dir = super().get_lora_base_from_roboflow(model_id, revision)
        return cache_dir

    def initialize_model(self, **kwargs):
        config_file = os.path.join(self.cache_dir, "adapter_config.json")

        with open(config_file, "r") as file:
            config = json.load(file)

        keys_to_remove = ["eva_config", "lora_bias", "exclude_modules"]

        for key in keys_to_remove:
            config.pop(key, None)

        with open(config_file, "w") as file:
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

        rm_weights = os.path.join(
            MODEL_CACHE_DIR, "lora-bases/qwen/qwen3vl-2b-instruct/main/weights.tar.gz"
        )
        if os.path.exists(rm_weights):
            os.remove(rm_weights)

        attn_implementation = (
            "flash_attention_2"
            if (is_flash_attn_2_available() and DEVICE and "cuda" in DEVICE)
            else "eager"
        )

        if self.use_quantization:
            self.base_model = self.transformers_class.from_pretrained(
                model_load_id,
                revision=revision,
                device_map=DEVICE,
                cache_dir=cache_dir,
                token=token,
                quantization_config=bnb_config,
                attn_implementation=attn_implementation,
            )
        else:
            self.base_model = self.transformers_class.from_pretrained(
                model_load_id,
                revision=revision,
                device_map=DEVICE,
                cache_dir=cache_dir,
                token=token,
                attn_implementation=attn_implementation,
            )

        if model_load_id != "qwen-pretrains/2":
            self.model = (
                PeftModel.from_pretrained(self.base_model, self.cache_dir)
                .eval()
                .to(self.dtype)
            )
        else:
            self.model = self.base_model.eval().to(self.dtype)

        self.model.merge_and_unload()

        preprocessor_config_path = os.path.join(self.cache_dir, "chat_template.json")
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, "r") as f:
                chat_template = json.load(f)["chat_template"]
            self.processor = self.processor_class.from_pretrained(
                model_load_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                chat_template=chat_template,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
        else:
            self.processor = self.processor_class.from_pretrained(
                model_load_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )

    def predict(self, image_in: Image.Image, prompt=None, **kwargs):
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

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
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
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
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
        decoded = decoded.replace(" addCriterion\n", "")

        return (decoded,)
