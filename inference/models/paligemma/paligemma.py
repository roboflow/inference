import os
import re

import numpy as np
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from inference.core.env import HUGGINGFACE_TOKEN, MODEL_CACHE_DIR

cache_dir = os.path.join(MODEL_CACHE_DIR)
import os
import time
from time import perf_counter
from typing import Any, List, Tuple, Union

import torch
from PIL import Image

from inference.core.cache.model_artifacts import save_bytes_in_cache
from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import API_KEY, MODEL_CACHE_DIR, PALIGEMMA_VERSION_ID
from inference.core.exceptions import ModelArtefactError
from inference.core.logger import logger
from inference.core.models.base import PreprocessReturnMetadata
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_from_url,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import load_image_rgb

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PaliGemma(RoboflowInferenceModel):
    """By using you agree to the terms listed at https://ai.google.dev/gemma/terms"""

    task_type = "lmm"

    def __init__(self, model_id, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)
        self.cache_model_artefacts()

        self.api_key = API_KEY
        self.cache_dir = os.path.join(MODEL_CACHE_DIR, self.endpoint + "/")
        self.initialize_model()

    def initialize_model(self):
        self.dtype = torch.float16
        self.model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                self.cache_dir,
                device_map=DEVICE,
            )
            .eval()
            .to(self.dtype)
        )

        self.processor = AutoProcessor.from_pretrained(
            self.cache_dir,
        )

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[Image.Image, PreprocessReturnMetadata]:
        pil_image = Image.fromarray(load_image_rgb(image))
        image_dims = pil_image.size

        return pil_image, PreprocessReturnMetadata({"image_dims": image_dims})

    def postprocess(
        self,
        predictions: Tuple[str],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> LMMInferenceResponse:
        text = predictions[0]
        image_dims = preprocess_return_metadata["image_dims"]
        response = LMMInferenceResponse(
            response=text,
            image=InferenceResponseImage(width=image_dims[0], height=image_dims[1]),
        )
        return [response]

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

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return [
            "config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            re.compile(r"model-\d{5}-of-\d{5}\.safetensors"),
            "preprocessor_config.json",
            "tokenizer_config.json",
        ]

    def download_model_artifacts_from_roboflow_api(self) -> None:
        api_data = get_roboflow_model_data(
            api_key=self.api_key,
            model_id=self.endpoint,
            endpoint_type=ModelEndpointType.ORT,
            device_id=self.device_id,
        )
        if "weights" not in api_data["ort"]:
            raise ModelArtefactError(
                f"`weights` key not available in Roboflow API response while downloading model weights."
            )
        for weights_url in api_data["ort"]["weights"].values():
            t1 = perf_counter()
            filename = weights_url.split("?")[0].split("/")[-1]
            if filename.endswith(".npz"):
                continue
            model_weights_response = get_from_url(weights_url, json_response=False)
            save_bytes_in_cache(
                content=model_weights_response.content,
                file=filename,
                model_id=self.endpoint,
            )
            if perf_counter() - t1 > 120:
                logger.debug(
                    "Weights download took longer than 120 seconds, refreshing API request"
                )
                api_data = get_roboflow_model_data(
                    api_key=self.api_key,
                    model_id=self.endpoint,
                    endpoint_type=ModelEndpointType.ORT,
                    device_id=self.device_id,
                )

    @property
    def weights_file(self) -> None:
        return None

    def download_model_artefacts_from_s3(self) -> None:
        raise NotImplementedError()


class LoRAPaliGemma(PaliGemma):
    def __init__(self, model_id, *args, huggingface_token=HUGGINGFACE_TOKEN, **kwargs):
        self.huggingface_token = huggingface_token
        super().__init__(model_id, *args, **kwargs)

    def initialize_model(self):
        lora_config = LoraConfig.from_pretrained(self.cache_dir, device_map=DEVICE)
        model_id = lora_config.base_model_name_or_path
        revision = lora_config.revision
        self.dtype = torch.float16
        if revision is not None:
            self.dtype = getattr(torch, revision)
        base_cache_dir = os.path.join(MODEL_CACHE_DIR, "huggingface")
        if self.huggingface_token is None:
            raise RuntimeError(
                "Must set environment variable HUGGINGFACE_TOKEN to load LoRA "
                "(or pass huggingface_token to this __init__)"
            )
        self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            device_map=DEVICE,
            cache_dir=base_cache_dir,
            token=self.huggingface_token,
        ).to(self.dtype)
        self.model = get_peft_model(self.base_model, lora_config).eval().to(self.dtype)

        self.processor = AutoProcessor.from_pretrained(self.cache_dir)
