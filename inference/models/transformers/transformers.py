import os
import re
import tarfile

import numpy as np
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from PIL import Image
from transformers import AutoModel, AutoProcessor, PaliGemmaForConditionalGeneration

from inference.core.env import HUGGINGFACE_TOKEN, MODEL_CACHE_DIR

cache_dir = os.path.join(MODEL_CACHE_DIR)
import os
import time
from time import perf_counter
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image

from inference.core.cache.model_artifacts import (
    get_cache_dir,
    get_cache_file_path,
    save_bytes_in_cache,
)
from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import API_KEY, DEVICE, MODEL_CACHE_DIR
from inference.core.exceptions import ModelArtefactError
from inference.core.logger import logger
from inference.core.models.base import PreprocessReturnMetadata
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_from_url,
    get_roboflow_base_lora,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import load_image_rgb

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TransformerModel(RoboflowInferenceModel):
    task_type = "lmm"
    transformers_class = AutoModel
    processor_class = AutoProcessor
    default_dtype = torch.float16
    generation_includes_input = False
    needs_hf_token = False
    skip_special_tokens = True

    def __init__(
        self, model_id, *args, dtype=None, huggingface_token=HUGGINGFACE_TOKEN, **kwargs
    ):
        super().__init__(model_id, *args, **kwargs)
        self.huggingface_token = huggingface_token
        if self.needs_hf_token and self.huggingface_token is None:
            raise RuntimeError(
                "Must set environment variable HUGGINGFACE_TOKEN to load LoRA "
                "(or pass huggingface_token to this __init__)"
            )
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = self.default_dtype
        self.cache_model_artefacts()

        self.cache_dir = os.path.join(MODEL_CACHE_DIR, self.endpoint + "/")
        self.initialize_model()

    def initialize_model(self):
        self.model = (
            self.transformers_class.from_pretrained(
                self.cache_dir,
                device_map=DEVICE,
                token=self.huggingface_token,
            )
            .eval()
            .to(self.dtype)
        )

        self.processor = self.processor_class.from_pretrained(
            self.cache_dir, token=self.huggingface_token
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
            prepared_inputs = self.prepare_generation_params(
                preprocessed_inputs=model_inputs
            )
            generation = self.model.generate(
                **prepared_inputs,
                max_new_tokens=1000,
                do_sample=False,
                early_stopping=False,
            )
            generation = generation[0]
            if self.generation_includes_input:
                generation = generation[input_len:]
            decoded = self.processor.decode(
                generation, skip_special_tokens=self.skip_special_tokens
            )

        return (decoded,)

    def prepare_generation_params(
        self, preprocessed_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return preprocessed_inputs

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


class LoRATransformerModel(TransformerModel):
    load_base_from_roboflow = False

    def initialize_model(self):
        lora_config = LoraConfig.from_pretrained(self.cache_dir, device_map=DEVICE)
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

        self.processor = self.processor_class.from_pretrained(
            self.cache_dir, revision=revision
        )

    def get_lora_base_from_roboflow(self, repo, revision) -> str:
        base_dir = os.path.join("lora-bases", repo, revision)
        cache_dir = get_cache_dir(base_dir)
        if os.path.exists(cache_dir):
            return cache_dir
        api_data = get_roboflow_base_lora(self.api_key, repo, revision, self.device_id)
        if "weights" not in api_data:
            raise ModelArtefactError(
                f"`weights` key not available in Roboflow API response while downloading model weights."
            )

        weights_url = api_data["weights"]["model"]
        model_weights_response = get_from_url(weights_url, json_response=False)
        filename = weights_url.split("?")[0].split("/")[-1]
        assert filename.endswith("tar.gz")
        save_bytes_in_cache(
            content=model_weights_response.content,
            file=filename,
            model_id=base_dir,
        )
        tar_file_path = get_cache_file_path(filename, base_dir)
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)

        return cache_dir

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return [
            "adapter_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "adapter_model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ]
