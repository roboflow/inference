"""Qwen3.5 VL model class proxying generation to a vLLM sidecar.

Mirrors `inference/models/qwen3_5vl/qwen3_5vl_inference_models.py` (the
in-process HF adapter): the inference server keeps performing
auth/billing/model-resolution/image-preprocessing per request, while
generation runs in the vLLM container (continuous batching + dynamic LoRA).

Preprocessing mirrors `Qwen35HF.pre_process_generation`
(`inference_models/models/qwen3_5/qwen3_5_hf.py`): the same
`<system_prompt>` split semantics and the same pixel budget the HF processor
applies (min 16*32*32 / max 512*32*32 with patch factor 32). Postprocessing
replicates `Qwen35HF.post_process_generation` think-tag parsing so responses
are shape-identical to the HF path.
"""

import base64
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from inference.core.entities.responses import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import API_KEY
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference.models.vllm_proxy.adapter_manager import get_adapter_manager
from inference.models.vllm_proxy.vllm_client import build_image_content_part
from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPT = "Describe what's in this image."

# Mirrors the AutoProcessor configuration in Qwen35HF.from_pretrained.
IMAGE_PATCH_FACTOR = 32
MIN_PIXELS = 16 * 32 * 32
MAX_PIXELS = 512 * 32 * 32

# Same allowed resize modes as Qwen35HF.from_pretrained.
ALLOWED_RESIZE_MODES = {
    ResizeMode.STRETCH_TO,
    ResizeMode.LETTERBOX,
    ResizeMode.CENTER_CROP,
    ResizeMode.LETTERBOX_REFLECT_EDGES,
    ResizeMode.FIT_LONGER_EDGE,
}


def split_prompt_and_system_prompt(prompt: Optional[str]) -> Tuple[str, str]:
    """Replicates the `<system_prompt>` split from Qwen35HF.pre_process_generation."""
    if prompt is None:
        return DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT
    split_prompt = prompt.split("<system_prompt>")
    if len(split_prompt) == 1:
        return split_prompt[0] or DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT
    return (
        split_prompt[0] or DEFAULT_PROMPT,
        split_prompt[1] or DEFAULT_SYSTEM_PROMPT,
    )


def smart_resize_dimensions(
    height: int,
    width: int,
    factor: int = IMAGE_PATCH_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Computes the (height, width) the Qwen image processor would resize to.

    Mirrors the `smart_resize` math of the HF Qwen VL image processor so the
    image sent to vLLM carries the same pixel budget the in-process HF path
    used (min/max pixels, dimensions divisible by the patch factor).
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "Absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def post_process_generated_text(
    text: str, enable_thinking: bool = False
) -> Union[str, Dict[str, str]]:
    """Replicates Qwen35HF.post_process_generation for a single decoded text.

    Cleans common artifacts and parses `<think>...</think>` blocks. When
    `enable_thinking` is set, the opening `<think>` tag is prepended when
    missing (the HF path's generation prompt ends with `<think>\\n`, so the
    tag is absent from generated tokens; vLLM applies the same chat template
    and is expected to behave identically - the guard keeps parsing correct
    either way).
    """
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("assistant\n", "")
    text = text.replace(" addCriterion\n", "")
    if enable_thinking:
        if not text.lstrip().startswith("<think>"):
            text = "<think>" + text
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        else:
            # Model hit max tokens before producing </think>.
            thinking = text.replace("<think>", "").strip()
            answer = ""
        return {"thinking": thinking, "answer": answer}
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


class Qwen35VLLMProxy(Model):
    """Qwen3.5 VL served via a vLLM sidecar (base model + dynamic LoRA)."""

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        self.task_type = "lmm"
        self.model_id = model_id
        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        self._adapter_manager = get_adapter_manager()
        # Cheap "load": resolves metadata, downloads/patches/registers the
        # adapter when needed - no model weights are loaded in this process.
        self._served_name = self._adapter_manager.resolve_and_register(
            model_id=model_id,
            api_key=self.api_key,
            weights_provider_extra_headers=extra_weights_provider_headers,
        )
        self._client = self._adapter_manager.client
        self._inference_config = self._load_inference_config()

    def _load_inference_config(self) -> Optional[InferenceConfig]:
        """Parses inference_config.json from the adapter package if present.

        Mirrors Qwen35HF.from_pretrained, which parses the config with the
        same allowed resize modes (the HF path holds the config without
        applying it during generation pre-processing - the processor's smart
        resize governs sizing; the same applies here).
        """
        registration = self._adapter_manager.get_registration(self._served_name)
        if registration is None:
            return None
        inference_config_path = os.path.join(
            registration.source_dir, "inference_config.json"
        )
        if not os.path.exists(inference_config_path):
            return None
        return parse_inference_config(
            config_path=inference_config_path,
            allowed_resize_modes=ALLOWED_RESIZE_MODES,
        )

    def preprocess(self, image: Any, prompt: str = "", **kwargs):
        is_batch = isinstance(image, list)
        if is_batch:
            raise ValueError("This model does not support batched-inference.")
        np_image = load_image_bgr(
            image,
            disable_preproc_auto_orient=kwargs.get(
                "disable_preproc_auto_orient", False
            ),
        )
        input_shape = PreprocessReturnMetadata({"image_dims": np_image.shape[:2][::-1]})
        user_prompt, system_prompt = split_prompt_and_system_prompt(prompt=prompt)
        image_base64 = self._encode_image_to_png_base64(np_image=np_image)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    build_image_content_part(image_base64=image_base64),
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return messages, input_shape

    @staticmethod
    def _encode_image_to_png_base64(np_image: np.ndarray) -> str:
        height, width = np_image.shape[:2]
        target_height, target_width = smart_resize_dimensions(
            height=height, width=width
        )
        if (target_height, target_width) != (height, width):
            np_image = cv2.resize(
                np_image,
                (target_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )
        success, encoded_image = cv2.imencode(".png", np_image)
        if not success:
            raise ValueError("Could not encode input image to PNG.")
        return base64.b64encode(encoded_image.tobytes()).decode("ascii")

    def predict(self, inputs: List[dict], **kwargs) -> str:
        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS
        chat_template_kwargs = None
        if "enable_thinking" in kwargs and kwargs["enable_thinking"] is not None:
            chat_template_kwargs = {"enable_thinking": bool(kwargs["enable_thinking"])}
        response = self._client.chat_completion(
            model=self._served_name,
            messages=inputs,
            temperature=0,
            max_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )
        return response["choices"][0]["message"]["content"] or ""

    def postprocess(
        self,
        predictions: str,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        result = post_process_generated_text(
            text=predictions,
            enable_thinking=kwargs.get("enable_thinking", False),
        )
        return [
            LMMInferenceResponse(
                response=result,
                image=InferenceResponseImage(
                    width=preprocess_return_metadata["image_dims"][0],
                    height=preprocess_return_metadata["image_dims"][1],
                ),
            )
        ]

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        pass
