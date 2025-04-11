import os
import re
import subprocess
import tarfile

from peft import LoraConfig
from peft.peft_model import PeftModel
from PIL import Image
from transformers import AutoModel, AutoProcessor

from inference.core.env import HUGGINGFACE_TOKEN, MODEL_CACHE_DIR

cache_dir = os.path.join(MODEL_CACHE_DIR)
import os
from time import perf_counter
from typing import Any, Dict, Tuple

import torch
from PIL import Image

from inference.core.cache.model_artifacts import (
    get_cache_dir,
    get_cache_file_path,
    save_bytes_in_cache,
)
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
    get_roboflow_instant_model_data,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import load_image_rgb

# Update device selection logic
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

class TransformerModel(RoboflowInferenceModel):
    task_type = "lmm"
    transformers_class = AutoModel
    processor_class = AutoProcessor
    default_dtype = torch.float16
    generation_includes_input = False
    needs_hf_token = False
    skip_special_tokens = True
    load_weights_as_transformers = False
    load_base_from_roboflow = True

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

    def initialize_model(self) -> None:
        """Initialize the model and processor."""
        if self.load_base_from_roboflow:
            model = self.transformers_class.from_pretrained(
                self.dataset_id,
                revision=self.version_id,
                token=self.api_key,
                torch_dtype=self.default_dtype,
            )
        else:
            try:
                model = self.transformers_class.from_pretrained(
                    self.model_id,
                    token=self.huggingface_token,
                    torch_dtype=self.default_dtype,
                )
            except Exception as e:
                raise
        
        try:
            processor = self.processor_class.from_pretrained(
                self.model_id,
                token=self.huggingface_token,
            )
        except Exception as e:
            raise
        
        self.model = model
        self.processor = processor
        
        self.model = self.model.to(DEVICE)

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
        if self.task_type == "llm":
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
                    no_repeat_ngram_size=0,
                )
                generation = generation[0]
                if self.generation_includes_input:
                    generation = generation[input_len:]

                decoded = self.processor.decode(
                    generation, skip_special_tokens=self.skip_special_tokens
                )

            return (decoded,)
        elif self.task_type == "depth-estimation":
            inputs = self.processor(images=image_in, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            post_processed_outputs = self.processor.post_process_depth_estimation(outputs, target_sizes=[(image_in.height, image_in.width)])
            return (post_processed_outputs,)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
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
            "tokenizer.json",
            re.compile(r"model.*\.safetensors"),
            "preprocessor_config.json",
            "tokenizer_config.json",
        ]

    def download_model_artifacts_from_roboflow_api(self) -> None:
        if self.load_weights_as_transformers:
            api_data = get_roboflow_model_data(
                api_key=self.api_key,
                model_id=self.endpoint,
                endpoint_type=ModelEndpointType.CORE_MODEL,
                device_id=self.device_id,
            )
            if "weights" not in api_data:
                raise ModelArtefactError(
                    f"`weights` key not available in Roboflow API response while downloading model weights."
                )
            weights = api_data["weights"]
        elif self.version_id is not None:
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
            weights = api_data["ort"]["weights"]
        else:
            api_data = get_roboflow_instant_model_data(
                api_key=self.api_key,
                model_id=self.endpoint,
            )
            if "modelFiles" not in api_data:
                raise ModelArtefactError(
                    f"`modelFiles` key not available in Roboflow API response while downloading model weights."
                )
            if "transformers" not in api_data["modelFiles"]:
                raise ModelArtefactError(
                    f"`transformers` key not available in Roboflow API response while downloading model weights."
                )
            weights = api_data["modelFiles"]["transformers"]
        files_to_download = list(weights.keys())
        for file_name in files_to_download:
            weights_url = weights[file_name]
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
            if filename.endswith("tar.gz"):
                try:
                    subprocess.run(
                        [
                            "tar",
                            "-xzf",
                            os.path.join(self.cache_dir, filename),
                            "-C",
                            self.cache_dir,
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    raise ModelArtefactError(
                        f"Failed to extract model archive {filename}. Error: {str(e)}"
                    ) from e

            if perf_counter() - t1 > 120:
                logger.debug(
                    "Weights download took longer than 120 seconds, refreshing API request"
                )
                if self.version_id is not None:
                    api_data = get_roboflow_model_data(
                        api_key=self.api_key,
                        model_id=self.endpoint,
                        endpoint_type=ModelEndpointType.ORT,
                        device_id=self.device_id,
                    )
                    weights = api_data["ort"]["weights"]
                elif self.load_weights_as_transformers:
                    api_data = get_roboflow_model_data(
                        api_key=self.api_key,
                        model_id=self.endpoint,
                        endpoint_type=ModelEndpointType.CORE_MODEL,
                        device_id=self.device_id,
                    )
                    weights = api_data["weights"]
                else:
                    api_data = get_roboflow_instant_model_data(
                        api_key=self.api_key,
                        model_id=self.endpoint,
                    )
                    weights = api_data["modelFiles"]["transformers"]

    @property
    def weights_file(self) -> None:
        return None

    def download_model_artefacts_from_s3(self) -> None:
        raise NotImplementedError()

    def cache_model_artefacts(self):
        """Cache model artifacts from either S3 or Roboflow API."""
        if self.load_weights_as_transformers and not self.load_base_from_roboflow:
            # Skip downloading if loading directly from transformers
            return None
        if self.load_weights_as_transformers:
            self.download_model_artefacts_from_s3()
            return None
        self.download_model_artifacts_from_roboflow_api()


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

        self.model.merge_and_unload()

        self.processor = self.processor_class.from_pretrained(
            model_load_id, revision=revision, cache_dir=cache_dir, token=token
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
            "adapter_model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ]
