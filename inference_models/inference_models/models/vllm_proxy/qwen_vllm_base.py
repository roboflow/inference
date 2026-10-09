"""Shared logic for Qwen VL family models proxied to a vLLM sidecar.

The proxy classes mirror the in-process HF implementations
(`inference_models/models/qwen3_5/`, `qwen3vl/`, `qwen3_8/`): image
preprocessing happens in this process, while generation runs in the vLLM
container (continuous batching + dynamic LoRA).

`QwenVLLMProxyBase` implements everything that is identical across families:
message construction (the `<system_prompt>` split semantics are shared by all
Qwen HF implementations), smart-resize to the HF processor's pixel budget,
the chat-completion call and the `prompt()` contract. Family-specific bits
are class attributes / hooks on the subclass:

- `image_patch_factor` / `min_pixels` / `max_pixels` - the pixel budget the
  family's HF AutoProcessor is configured with.
- `default_system_prompt` - differs between families.
- `default_max_new_tokens` - each family reads its own env-configured default.
- `supports_thinking` - whether `enable_thinking` is forwarded to the chat
  template (qwen3_5 / qwen3_8 only; qwen3vl-instruct has no thinking mode).
- `post_process_text` - family-specific decoded-text cleanup (think-tag
  parsing for qwen3_5, plain artifact cleanup for qwen3vl).
- `_get_adapter_manager` - defined in the family module so its module-level
  `get_adapter_manager` symbol stays patchable in tests.
"""

import base64
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from inference_models import configuration
from inference_models.entities import ColorFormat
from inference_models.logger import LOGGER
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    get_adapter_manager,
)
from inference_models.models.vllm_proxy.errors import VLLMHTTPError
from inference_models.models.vllm_proxy.vllm_client import build_image_content_part
from inference_models.weights_providers.entities import ModelMetadata

DEFAULT_PROMPT = "Describe what's in this image."

ALLOWED_RESIZE_MODES = {
    ResizeMode.STRETCH_TO,
    ResizeMode.LETTERBOX,
    ResizeMode.CENTER_CROP,
    ResizeMode.LETTERBOX_REFLECT_EDGES,
    ResizeMode.FIT_LONGER_EDGE,
}

ImageInput = Union[torch.Tensor, np.ndarray]
ImagesInput = Union[ImageInput, List[ImageInput]]


def split_prompt_and_system_prompt(
    prompt: Optional[str],
    default_system_prompt: str,
    default_prompt: str = DEFAULT_PROMPT,
) -> Tuple[str, str]:
    """Replicates the `<system_prompt>` split shared by the Qwen HF implementations.

    Args:
        prompt: Raw prompt, optionally carrying a `<system_prompt>` marker.
        default_system_prompt: System prompt used when none is given.
        default_prompt: User prompt used when none is given.

    Returns:
        `(user_prompt, system_prompt)`.
    """
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

    Args:
        height: Source image height.
        width: Source image width.
        factor: Patch factor both dimensions must be divisible by.
        min_pixels: Lower pixel budget.
        max_pixels: Upper pixel budget.

    Returns:
        `(height, width)` after the smart resize.

    Raises:
        ValueError: If the aspect ratio exceeds 200.
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


class QwenVLLMProxyBase:
    """Base class for Qwen VL models served via a vLLM sidecar."""

    image_patch_factor: int
    min_pixels: int
    max_pixels: int
    default_system_prompt: str
    default_max_new_tokens: int
    default_prompt: str = DEFAULT_PROMPT
    supports_thinking: bool = False

    @classmethod
    def from_model_metadata(
        cls,
        model_id: str,
        metadata: ModelMetadata,
        api_key: Optional[str] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "QwenVLLMProxyBase":
        """Resolves and registers `model_id` with the vLLM sidecar.

        This is the cheap "load": the adapter (if any) is downloaded, patched
        and registered with vLLM - no model weights are loaded in this
        process.

        Args:
            model_id: Roboflow model id as requested by the caller.
            metadata: Model metadata already fetched by the loader.
            api_key: Roboflow API key, kept for the request-path self-heal.
            weights_provider_extra_headers: Extra weights-provider headers,
                kept for the request-path self-heal.
            **kwargs: Loader init kwargs irrelevant to the proxy (ignored).

        Returns:
            The proxy instance.
        """
        api_key = api_key or configuration.ROBOFLOW_API_KEY
        adapter_manager = cls._get_adapter_manager()
        served_name = adapter_manager.resolve_and_register(
            model_id,
            metadata=metadata,
            api_key=api_key,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )
        return cls(
            model_id=model_id,
            served_name=served_name,
            adapter_manager=adapter_manager,
            api_key=api_key,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )

    def __init__(
        self,
        model_id: str,
        served_name: str,
        adapter_manager: AdapterManager,
        api_key: Optional[str] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self._weights_provider_extra_headers = weights_provider_extra_headers
        self._adapter_manager = adapter_manager
        self._served_name = served_name
        self._client = adapter_manager.client
        self._inference_config = self._load_inference_config()

    @classmethod
    def _get_adapter_manager(cls) -> AdapterManager:
        return get_adapter_manager()

    def _load_inference_config(self) -> Optional[InferenceConfig]:
        registration = self._adapter_manager.get_registration(self._served_name)
        if registration is None:
            return None
        inference_config_path = Path(registration.source_dir) / "inference_config.json"
        if not inference_config_path.exists():
            return None
        return parse_inference_config(
            config_path=str(inference_config_path),
            allowed_resize_modes=ALLOWED_RESIZE_MODES,
        )

    def prompt(
        self,
        images: ImagesInput,
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        """Runs one chat completion per image and returns the cleaned texts.

        Mirrors the HF classes' `prompt()`: images are BGR arrays (or RGB
        CHW tensors), a list is served as one completion per image in order,
        decoding is greedy (`do_sample` and `skip_special_tokens` are
        accepted for signature parity and ignored).

        Args:
            images: One image or a list of images.
            prompt: User prompt, optionally with a `<system_prompt>` marker.
            input_color_format: Accepted for parity with the HF classes;
                arrays are assumed BGR like there.
            max_new_tokens: Generation cap; the family default when None.
            do_sample: Ignored - the proxy always decodes greedily.
            skip_special_tokens: Ignored - special tokens are stripped.
            **kwargs: `enable_thinking` for families supporting it.

        Returns:
            One entry per image: a string, or a `{"thinking", "answer"}`
            dict when thinking is enabled on a family that supports it.
        """
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        enable_thinking = kwargs.get("enable_thinking")
        chat_template_kwargs = self._build_chat_template_kwargs(
            enable_thinking=enable_thinking
        )
        image_list = images if isinstance(images, list) else [images]

        results = []
        for image in image_list:
            messages = self._build_messages(image=image, prompt=prompt)
            text = self._generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )
            results.append(
                self.post_process_text(text=text, enable_thinking=enable_thinking)
            )
        return results

    def _build_messages(
        self, image: ImageInput, prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        np_image = self._to_bgr_array(image=image)
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
        return messages

    @staticmethod
    def _to_bgr_array(image: ImageInput) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        np_image = image.detach().cpu().numpy()
        if np_image.ndim == 3 and np_image.shape[0] in (1, 3):
            np_image = np.transpose(np_image, (1, 2, 0))
        if np_image.dtype != np.uint8:
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np_image[:, :, ::-1])

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

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        chat_template_kwargs: Optional[Dict[str, Any]],
    ) -> str:
        try:
            response = self._chat_completion(
                messages=messages,
                max_new_tokens=max_new_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )
        except VLLMHTTPError as error:
            if not self._is_unknown_served_model_error(error=error):
                raise
            LOGGER.warning(
                "vLLM does not know served model %s (model_id=%s) despite "
                "local registration - re-registering and retrying once. "
                "vLLM said: %r",
                self._served_name,
                self.model_id,
                (error.response_body or "")[:200],
            )
            self._adapter_manager.invalidate(served_name=self._served_name)
            self._served_name = self._adapter_manager.resolve_and_register(
                self.model_id,
                api_key=self.api_key,
                weights_provider_extra_headers=self._weights_provider_extra_headers,
            )
            response = self._chat_completion(
                messages=messages,
                max_new_tokens=max_new_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )
        return response["choices"][0]["message"]["content"] or ""

    def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        chat_template_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self._client.chat_completion(
            model=self._served_name,
            messages=messages,
            temperature=0,
            max_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )

    def _is_unknown_served_model_error(self, error: VLLMHTTPError) -> bool:
        """True iff vLLM rejected the request because OUR served model is unknown.

        vLLM's OpenAI server answers 404 with `The model `<name>` does not
        exist.` for unknown served models. Matched defensively: HTTP 404 +
        the served name in the body + a not-found phrasing. Anything else
        (other adapters, genuine 4xx/5xx) must NOT trigger the self-heal
        retry.
        """
        if error.status_code != 404:
            return False
        body = (error.response_body or "").lower()
        if self._served_name.lower() not in body:
            return False
        return "does not exist" in body or "not found" in body

    def _build_chat_template_kwargs(
        self, enable_thinking: Optional[bool]
    ) -> Optional[Dict[str, Any]]:
        if not self.supports_thinking:
            return None
        if enable_thinking is None:
            return None
        return {"enable_thinking": bool(enable_thinking)}

    def post_process_text(
        self, text: str, enable_thinking: Optional[bool] = None
    ) -> Union[str, Dict[str, str]]:
        """Family-specific cleanup of the decoded generation.

        Args:
            text: Raw assistant message content.
            enable_thinking: Whether think-tag parsing applies (families
                with thinking support only).

        Returns:
            The cleaned text, or a `{"thinking", "answer"}` dict.
        """
        raise NotImplementedError
