import os
from typing import Any, Dict

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from inference.core.entities.responses.inference import LMMInferenceResponse
from inference.core.models.types import PreprocessReturnMetadata
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

    def initialize_model(self):
        self.transformers_class = import_class_from_file(
            os.path.join(self.cache_dir, "modeling_florence2.py"),
            "Florence2ForConditionalGeneration",
        )

        self.processor_class = import_class_from_file(
            os.path.join(self.cache_dir, "processing_florence2.py"),
            "Florence2Processor",
        )
        super().initialize_model()

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

    def get_lora_base_from_roboflow(self, model_id, revision):
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
