import json
import os
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from inference.models.florence2.utils import import_class_from_file
from inference.models.transformers import LoRATransformerModel, TransformerModel


class Florence2(TransformerModel):
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

    def prepare_generation_params(
        self, preprocessed_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "input_ids": preprocessed_inputs["input_ids"],
            "pixel_values": preprocessed_inputs["pixel_values"],
        }

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        model_inputs = self.processor(
            text=prompt, images=image_in, return_tensors="pt"
        ).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            prepared_inputs = self.prepare_generation_params(
                preprocessed_inputs=model_inputs
            )
            generation = self.model.generate(
                **prepared_inputs,
                max_new_tokens=1024,
                do_sample=False,
                early_stopping=False,
                num_beams=3
            )
            generation = generation[0]
            if self.generation_includes_input:
                generation = generation[input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=False)
            parsed_answer = self.processor.post_process_generation(
                decoded, task=prompt.split(">")[0] + ">", image_size=image_in.size
            )
        return (
            decoded,
            parsed_answer,
        )


class LoRAFlorence2(LoRATransformerModel):
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

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        model_inputs = self.processor(
            text=prompt, images=image_in, return_tensors="pt"
        ).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            prepared_inputs = self.prepare_generation_params(
                preprocessed_inputs=model_inputs
            )
            generation = self.model.generate(
                **prepared_inputs,
                max_new_tokens=1024,
                do_sample=False,
                early_stopping=False,
                num_beams=3
            )
            generation = generation[0]
            if self.generation_includes_input:
                generation = generation[input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=False)
            parsed_answer = self.processor.post_process_generation(
                decoded, task=prompt.split(">")[0] + ">", image_size=image_in.size
            )

        return (json.dumps(parsed_answer),)
