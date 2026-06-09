"""Shared logic for Qwen VL family models proxied to a vLLM sidecar.

The proxy classes mirror the in-process HF adapters (see
`inference/models/qwen3_5vl/` and `inference/models/qwen3vl/`): the inference
server keeps performing auth/billing/model-resolution/image-preprocessing per
request, while generation runs in the vLLM container (continuous batching +
dynamic LoRA).

`QwenVLLMProxyBase` implements everything that is identical across families:
message construction (the `<system_prompt>` split semantics are shared by all
Qwen HF adapters), smart-resize to the HF processor's pixel budget, the
chat-completion call and the response shape. Family-specific bits are class
attributes / hooks on the subclass:

- `image_patch_factor` / `min_pixels` / `max_pixels` - the pixel budget the
  family's HF AutoProcessor is configured with.
- `default_system_prompt` - differs between families.
- `default_max_new_tokens` - each family reads its own env-configured default.
- `supports_thinking` - whether `enable_thinking` is forwarded to the chat
  template (qwen3_5 only; qwen3vl-instruct has no thinking mode).
- `post_process_text` - family-specific decoded-text cleanup (think-tag
  parsing for qwen3_5, plain artifact cleanup for qwen3vl).
- `_get_adapter_manager` - defined in the family module so its module-level
  `get_adapter_manager` symbol stays patchable in tests.
"""

import base64
import math
import os
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
from inference.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    get_adapter_manager,
)
from inference.models.vllm_proxy.vllm_client import build_image_content_part
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)

DEFAULT_PROMPT = "Describe what's in this image."

# Same allowed resize modes as the Qwen HF adapters' from_pretrained.
ALLOWED_RESIZE_MODES = {
    ResizeMode.STRETCH_TO,
    ResizeMode.LETTERBOX,
    ResizeMode.CENTER_CROP,
    ResizeMode.LETTERBOX_REFLECT_EDGES,
    ResizeMode.FIT_LONGER_EDGE,
}


def split_prompt_and_system_prompt(
    prompt: Optional[str],
    default_system_prompt: str,
    default_prompt: str = DEFAULT_PROMPT,
) -> Tuple[str, str]:
    """Replicates the `<system_prompt>` split shared by the Qwen HF adapters."""
    if prompt is None:
        return default_prompt, default_system_prompt
    split_prompt = prompt.split("<system_prompt>")
    if len(split_prompt) == 1:
        return split_prompt[0] or default_prompt, default_system_prompt
    return (
        split_prompt[0] or default_prompt,
        split_prompt[1] or default_system_prompt,
    )


def smart_resize_dimensions(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """Computes the (height, width) the Qwen image processor would resize to.

    Mirrors the `smart_resize` math of the HF Qwen VL image processors so the
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


class QwenVLLMProxyBase(Model):
    """Base class for Qwen VL models served via a vLLM sidecar."""

    # Family-specific knobs - subclasses must define these.
    image_patch_factor: int
    min_pixels: int
    max_pixels: int
    default_system_prompt: str
    default_max_new_tokens: int
    default_prompt: str = DEFAULT_PROMPT
    supports_thinking: bool = False

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
        self._adapter_manager = self._get_adapter_manager()
        # Cheap "load": resolves metadata, downloads/patches/registers the
        # adapter when needed - no model weights are loaded in this process.
        self._served_name = self._adapter_manager.resolve_and_register(
            model_id=model_id,
            api_key=self.api_key,
            weights_provider_extra_headers=extra_weights_provider_headers,
        )
        self._client = self._adapter_manager.client
        self._inference_config = self._load_inference_config()

    def _get_adapter_manager(self) -> AdapterManager:
        """Family modules override this so their module-level
        `get_adapter_manager` symbol stays patchable in tests."""
        return get_adapter_manager()

    def _load_inference_config(self) -> Optional[InferenceConfig]:
        """Parses inference_config.json from the adapter package if present.

        Mirrors the HF adapters' from_pretrained, which parses the config
        with the same allowed resize modes (the HF path holds the config
        without applying it during generation pre-processing - the
        processor's smart resize governs sizing; the same applies here).
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
        user_prompt, system_prompt = split_prompt_and_system_prompt(
            prompt=prompt,
            default_system_prompt=self.default_system_prompt,
            default_prompt=self.default_prompt,
        )
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

    def _encode_image_to_png_base64(self, np_image: np.ndarray) -> str:
        height, width = np_image.shape[:2]
        target_height, target_width = smart_resize_dimensions(
            height=height,
            width=width,
            factor=self.image_patch_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
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
            max_new_tokens = self.default_max_new_tokens
        response = self._client.chat_completion(
            model=self._served_name,
            messages=inputs,
            temperature=0,
            max_tokens=max_new_tokens,
            chat_template_kwargs=self._build_chat_template_kwargs(kwargs=kwargs),
        )
        return response["choices"][0]["message"]["content"] or ""

    def _build_chat_template_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.supports_thinking:
            # The family's chat template has no thinking switch - forwarding
            # `enable_thinking` would be a template error / silent no-op.
            return None
        if "enable_thinking" in kwargs and kwargs["enable_thinking"] is not None:
            return {"enable_thinking": bool(kwargs["enable_thinking"])}
        return None

    def post_process_text(self, text: str, **kwargs) -> Union[str, Dict[str, str]]:
        """Family-specific cleanup of the decoded generation."""
        raise NotImplementedError

    def postprocess(
        self,
        predictions: str,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        result = self.post_process_text(text=predictions, **kwargs)
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
