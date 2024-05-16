import os

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from inference.core.env import MODEL_CACHE_DIR

cache_dir = os.path.join(MODEL_CACHE_DIR)
import time

import os
from time import perf_counter
from typing import Any, List, Tuple, Union

import torch
from PIL import Image

from inference.core.entities.requests.paligemma import PaliGemmaInferenceRequest
from inference.core.entities.responses.paligemma import PaliGemmaInferenceResponse
from inference.core.env import API_KEY, MODEL_CACHE_DIR, PALIGEMMA_VERSION_ID
from inference.core.models.base import PreprocessReturnMetadata
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PaliGemma(RoboflowCoreModel):
    def __init__(self, *args, model_id=f"paligemma/{PALIGEMMA_VERSION_ID}", **kwargs):
        super().__init__(*args, model_id=model_id, **kwargs)
        self.model_id = model_id
        self.endpoint = model_id
        self.api_key = API_KEY
        self.dataset_id, self.version_id = model_id.split("/")
        self.cache_dir = os.path.join(MODEL_CACHE_DIR, self.endpoint + "/")
        dtype = torch.bfloat16
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.cache_dir,
            torch_dtype=dtype,
            device_map=DEVICE,
            revision="bfloat16",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.cache_dir,
        )
        self.task_type = "lmm"

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[Image.Image, PreprocessReturnMetadata]:
        pil_image = Image.fromarray(load_image_rgb(image))

        return pil_image, PreprocessReturnMetadata({})

    def postprocess(
        self,
        predictions: Tuple[str],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions[0]

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        model_inputs = self.processor(
            text=prompt, images=image_in, return_tensors="pt"
        ).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        return (decoded,)

    def infer_from_request(
        self, request: PaliGemmaInferenceRequest
    ) -> PaliGemmaInferenceResponse:
        t1 = perf_counter()
        text = self.infer(**request.dict())
        response = PaliGemmaInferenceResponse(response=text)
        response.time = perf_counter() - t1
        return response

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return [
            "config.json",
            "model-00002-of-00002.safetensors",
            "special_tokens_map.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "model-00001-of-00002.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ]


if __name__ == "__main__":
    m = PaliGemma()
    print(m.infer())
