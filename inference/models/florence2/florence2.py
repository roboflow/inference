import json
import os
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from inference.models.florence2.utils import import_class_from_file
from inference.models.transformers import LoRATransformerModel, TransformerModel

class Florence2Processing:
    def prepare_generation_params(
        self, preprocessed_inputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any]]:
        return ({
            "input_ids": preprocessed_inputs["input_ids"],
            "pixel_values": preprocessed_inputs["pixel_values"],
            "max_new_tokens": 1024,
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 3,
        }, {"skip_special_tokens": False})

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        decoded, = super().predict(image_in, prompt, history, **kwargs)
        parsed_answer = self.processor.post_process_generation(
            decoded, task=prompt.split(">")[0] + ">", image_size=image_in.size
        )

        return (
            decoded,
            parsed_answer,
        )



class Florence2(Florence2Processing, TransformerModel):
    transformers_class = AutoModelForCausalLM
    default_dtype = torch.float32

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
