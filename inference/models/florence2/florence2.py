import os

import torch
from transformers import AutoModelForCausalLM

from inference.models.florence2.utils import import_class_from_file
from inference.models.transformers import LoRATransformerModel, TransformerModel
from inference.core.env import DEVICE

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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

        self.model = (
            self.transformers_class.from_pretrained(
                self.cache_dir,
                device_map=DEVICE,
                token=self.huggingface_token,
            )
            .eval()
            .to(self.dtype)
        )

        self.processor = AutoProcessor.from_pretrained(
            self.cache_dir, token=self.huggingface_token
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
