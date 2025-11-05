import json
import os
from typing import Any, Dict

import torch
from peft import get_peft_model, load_peft_weights, set_peft_model_state_dict
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from inference.core.env import DEVICE, MODEL_CACHE_DIR
from inference.models.florence2.utils import import_class_from_file
from inference.models.transformers import LoRATransformerModel, TransformerModel


class Florence2Processing:
    def predict(self, image_in: Image, prompt="", history=None, **kwargs):
        (decoded,) = super().predict(image_in, prompt, history, **kwargs)
        parsed_answer = self.processor.post_process_generation(
            decoded, task=prompt.split(">")[0] + ">", image_size=image_in.size
        )

        return (parsed_answer,)


class Florence2(Florence2Processing, TransformerModel):
    transformers_class = AutoModelForCausalLM
    default_dtype = torch.float32
    skip_special_tokens = False

    def initialize_model(self, **kwargs):
        try:
            from transformers import (
                AutoConfig,
                Florence2Config,
                Florence2ForConditionalGeneration,
                Florence2Processor,
            )

            self.transformers_class = Florence2ForConditionalGeneration
            self.processor_class = Florence2Processor
        except ImportError:
            self.transformers_class = import_class_from_file(
                os.path.join(self.cache_dir, "modeling_florence2.py"),
                "Florence2ForConditionalGeneration",
            )
            self.processor_class = import_class_from_file(
                os.path.join(self.cache_dir, "processing_florence2.py"),
                "Florence2Processor",
            )
        super().initialize_model(**kwargs)

    def prepare_generation_params(
        self, preprocessed_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "input_ids": preprocessed_inputs["input_ids"],
            "pixel_values": preprocessed_inputs["pixel_values"],
        }


class LoRAFlorence2(Florence2Processing, LoRATransformerModel):
    load_base_from_roboflow = True
    transformers_class = AutoModelForCausalLM
    default_dtype = torch.float32

    def initialize_model(self, **kwargs):
        import torch
        from peft import LoraConfig
        from peft.peft_model import PeftModel

        lora_config = LoraConfig.from_pretrained(self.cache_dir, device_map=DEVICE)
        model_id = lora_config.base_model_name_or_path
        revision = lora_config.revision
        original_revision_pre_mapping = revision
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
        try:
            from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
            from transformers import (
                AutoConfig,
                Florence2Config,
                Florence2ForConditionalGeneration,
                Florence2Processor,
            )

            adapter_cfg_path = os.path.join(self.cache_dir, "adapter_config.json")
            with open(adapter_cfg_path, "r") as f:
                adapter_cfg = json.load(f)

            requested_target_modules = adapter_cfg.get("target_modules") or []
            adapter_task_type = adapter_cfg.get("task_type") or "SEQ_2_SEQ_LM"
            lora_config = LoraConfig(
                r=adapter_cfg.get("r", 8),
                lora_alpha=adapter_cfg.get("lora_alpha", 8),
                lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
                bias="none",
                target_modules=sorted(requested_target_modules),
                use_dora=bool(adapter_cfg.get("use_dora", False)),
                use_rslora=bool(adapter_cfg.get("use_rslora", False)),
                task_type=adapter_task_type,
            )

            model = get_peft_model(self.base_model, lora_config)
            # Load adapter weights
            adapter_state = load_peft_weights(self.cache_dir, device=DEVICE)
            adapter_state = normalize_adapter_state_dict(adapter_state)
            load_result = set_peft_model_state_dict(
                model, adapter_state, adapter_name="default"
            )
            tuner = lora_config.peft_type
            tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(tuner, "")
            adapter_missing_keys = []
            # Filter missing keys specific to the current adapter and tuner prefix.
            for key in load_result.missing_keys:
                if tuner_prefix in key and "default" in key:
                    adapter_missing_keys.append(key)
            load_result.missing_keys.clear()
            load_result.missing_keys.extend(adapter_missing_keys)
            if original_revision_pre_mapping == "refs/pr/6":
                if len(load_result.missing_keys) > 2:
                    raise RuntimeError(
                        "Could not load LoRA weights for the model - found missing checkpoint keys "
                        f"({len(load_result.missing_keys)}): {load_result.missing_keys}",
                    )

            else:
                if len(load_result.missing_keys) > 0:
                    raise RuntimeError(
                        "Could not load LoRA weights for the model - found missing checkpoint keys "
                        f"({len(load_result.missing_keys)}): {load_result.missing_keys}",
                    )

            self.model = model
        except ImportError:
            self.model = (
                PeftModel.from_pretrained(self.base_model, self.cache_dir)
                .eval()
                .to(self.dtype)
            )

        self.model.merge_and_unload()

        self.processor = self.processor_class.from_pretrained(
            model_load_id, revision=revision, cache_dir=cache_dir, token=token
        )

    def get_lora_base_from_roboflow(self, model_id, revision):

        try:
            from transformers import (
                AutoConfig,
                Florence2Config,
                Florence2ForConditionalGeneration,
                Florence2Processor,
            )

            revision_mapping = {
                ("microsoft/Florence-2-base-ft", "refs/pr/6"): "refs/pr/29-converted",
                ("microsoft/Florence-2-base-ft", "refs/pr/22"): "refs/pr/29-converted",
                ("microsoft/Florence-2-large-ft", "refs/pr/20"): "refs/pr/38-converted",
            }
            revision = revision_mapping.get((model_id, revision), revision)
            cache_dir = super().get_lora_base_from_roboflow(model_id, revision)

            self.transformers_class = Florence2ForConditionalGeneration
            self.processor_class = Florence2Processor
        except ImportError:
            cache_dir = super().get_lora_base_from_roboflow(model_id, revision)
            self.transformers_class = import_class_from_file(
                os.path.join(cache_dir, "modeling_florence2.py"),
                "Florence2ForConditionalGeneration",
            )
            self.processor_class = import_class_from_file(
                os.path.join(cache_dir, "processing_florence2.py"),
                "Florence2Processor",
            )

        return cache_dir

    def prepare_generation_params(
        self, preprocessed_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "input_ids": preprocessed_inputs["input_ids"],
            "pixel_values": preprocessed_inputs["pixel_values"],
        }


def normalize_adapter_state_dict(adapter_state: dict) -> dict:
    normalized = {}
    for key, value in adapter_state.items():
        new_key = key
        # Ensure Florence-2 PEFT prefix matches injected structure
        if (
            "base_model.model.vision_tower." in new_key
            and "base_model.model.model.vision_tower." not in new_key
        ):
            new_key = new_key.replace(
                "base_model.model.vision_tower.",
                "base_model.model.model.vision_tower.",
            )
        # Normalize original repo FFN path to HF-native
        if ".ffn.fn.net.fc1" in new_key:
            new_key = new_key.replace(".ffn.fn.net.fc1", ".ffn.fc1")
        if ".ffn.fn.net.fc2" in new_key:
            new_key = new_key.replace(".ffn.fn.net.fc2", ".ffn.fc2")
        # Normalize language path if needed
        if ".language_model.model." in new_key:
            new_key = new_key.replace(
                ".language_model.model.", ".model.language_model."
            )
        normalized[new_key] = value
    return normalized
