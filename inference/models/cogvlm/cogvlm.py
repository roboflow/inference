import os
from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.entities.responses.cogvlm import CogVLMResponse
from inference.core.env import (
    API_KEY,
    COGVLM_LOAD_4BIT,
    COGVLM_LOAD_8BIT,
    COGVLM_VERSION_ID,
    MODEL_CACHE_DIR,
)
from inference.core.models.base import Model, PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image_rgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CogVLM(Model):
    def __init__(self, model_id=f"cogvlm/{COGVLM_VERSION_ID}", **kwargs):
        self.model_id = model_id
        self.endpoint = model_id
        self.api_key = API_KEY
        self.dataset_id, self.version_id = model_id.split("/")
        if COGVLM_LOAD_4BIT and COGVLM_LOAD_8BIT:
            raise ValueError(
                "Only one of environment variable `COGVLM_LOAD_4BIT` or `COGVLM_LOAD_8BIT` can be true"
            )
        self.cache_dir = os.path.join(MODEL_CACHE_DIR, self.endpoint)
        with torch.inference_mode():
            self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
            self.model = AutoModelForCausalLM.from_pretrained(
                f"THUDM/{self.version_id}",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                load_in_4bit=COGVLM_LOAD_4BIT,
                load_in_8bit=COGVLM_LOAD_8BIT,
                cache_dir=self.cache_dir,
            ).eval()
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
        images = [image_in]
        if history is None:
            history = []
        built_inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, history=history, images=images
        )  # chat mode
        inputs = {
            "input_ids": built_inputs["input_ids"].unsqueeze(0).to(DEVICE),
            "token_type_ids": built_inputs["token_type_ids"].unsqueeze(0).to(DEVICE),
            "attention_mask": built_inputs["attention_mask"].unsqueeze(0).to(DEVICE),
            "images": [[built_inputs["images"][0].to(DEVICE).to(torch.float16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(outputs[0])
            if text.endswith("</s>"):
                text = text[:-4]
            return (text,)

    def infer_from_request(self, request: CogVLMInferenceRequest) -> CogVLMResponse:
        t1 = perf_counter()
        text = self.infer(**request.dict())
        response = CogVLMResponse(response=text)
        response.time = perf_counter() - t1
        return response


if __name__ == "__main__":
    m = CogVLM()
    m.infer()
