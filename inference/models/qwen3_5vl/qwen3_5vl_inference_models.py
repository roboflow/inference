import base64
import binascii
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.responses import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    VLLM_LMM_API_KEY,
    VLLM_LMM_BASE_URL,
    VLLM_LMM_ENABLED,
    VLLM_LMM_MODEL_NAME,
    VLLM_LMM_TEMPERATURE,
    VLLM_LMM_TIMEOUT_SECONDS,
)
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image_bgr
from inference_models import AutoModel
from inference_models.models.qwen3_5.qwen3_5_hf import Qwen35HF

NATIVE_SYSTEM_PROMPT_SEPARATOR = "<system_prompt>"


def _extract_base64_image_payload(image: Any) -> Optional[str]:
    image_type = None
    image_value = None
    if isinstance(image, InferenceRequestImage):
        image_type = image.type
        image_value = image.value
    elif isinstance(image, dict):
        image_type = image.get("type")
        image_value = image.get("value")

    if image_type is None or image_type.lower() != "base64":
        return None
    if isinstance(image_value, bytes):
        image_value = image_value.decode("ascii")
    if not isinstance(image_value, str):
        return None
    return image_value


def _base64_payload_to_data_url(base64_payload: str) -> str:
    if base64_payload.startswith("data:image/"):
        return base64_payload
    media_type = _guess_base64_image_media_type(base64_payload)
    return f"data:{media_type};base64,{base64_payload}"


def _guess_base64_image_media_type(base64_payload: str) -> str:
    sample = base64_payload[:64]
    try:
        header = base64.b64decode(sample, validate=False)
    except (binascii.Error, ValueError):
        return "image/jpeg"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"BM"):
        return "image/bmp"
    return "image/jpeg"


def _normalize_vllm_base_url(base_url: str) -> str:
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        return normalized_base_url
    return f"{normalized_base_url}/v1"


def _split_native_prompt(prompt: Optional[str]) -> Tuple[str, Optional[str]]:
    if not prompt:
        return "", None
    user_prompt, separator, system_prompt = prompt.partition(
        NATIVE_SYSTEM_PROMPT_SEPARATOR
    )
    if not separator:
        return prompt, None
    return user_prompt, system_prompt or None


def _encode_image_data_url(image: Any) -> str:
    jpeg_bytes = encode_image_to_jpeg_bytes(image)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _image_data_url_from_input(image: Any, loaded_image: Any) -> str:
    base64_payload = _extract_base64_image_payload(image)
    if base64_payload is not None:
        return _base64_payload_to_data_url(base64_payload)
    return _encode_image_data_url(loaded_image)


def _build_vllm_messages(prompt: Optional[str], image_data_url: str) -> List[Dict]:
    user_prompt, system_prompt = _split_native_prompt(prompt)
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    messages.append({"role": "user", "content": user_content})
    return messages


class InferenceModelsQwen35VLAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "lmm"
        self._model: Optional[Qwen35HF] = None
        self._vllm_client: Optional[OpenAI] = None
        self._vllm_model_name = VLLM_LMM_MODEL_NAME
        self._vllm_temperature = VLLM_LMM_TEMPERATURE

        if VLLM_LMM_ENABLED:
            if not VLLM_LMM_BASE_URL:
                raise ValueError(
                    "VLLM_LMM_BASE_URL must be set when VLLM_LMM_ENABLED=True"
                )
            self._vllm_client = OpenAI(
                base_url=_normalize_vllm_base_url(VLLM_LMM_BASE_URL),
                api_key=VLLM_LMM_API_KEY,
                timeout=VLLM_LMM_TIMEOUT_SECONDS,
            )
            return

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )

        self._model = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

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
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if self._vllm_client is not None:
            return (
                {
                    "image_data_url": _image_data_url_from_input(image, np_image),
                    "prompt": prompt,
                },
                input_shape,
            )
        return (
            self._model.pre_process_generation(np_image, prompt, **mapped_kwargs),
            input_shape,
        )

    def predict(self, inputs, **kwargs) -> Any:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if self._vllm_client is not None:
            return self._generate_with_vllm(inputs, **mapped_kwargs)
        return self._model.generate(inputs, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Any,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if self._vllm_client is not None:
            result = predictions
        else:
            result = self._model.post_process_generation(predictions, **mapped_kwargs)[
                0
            ]
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

    def _generate_with_vllm(self, inputs: Dict[str, Any], **kwargs) -> str:
        messages = _build_vllm_messages(
            prompt=inputs["prompt"],
            image_data_url=inputs["image_data_url"],
        )
        request_kwargs: Dict[str, Any] = {
            "model": self._vllm_model_name,
            "messages": messages,
            "temperature": self._vllm_temperature,
        }
        max_tokens = kwargs.get("max_new_tokens") or kwargs.get("max_tokens")
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        extra_body = _build_vllm_extra_body(kwargs)
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = self._vllm_client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if content is None:
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
            raise RuntimeError(
                f"vLLM returned empty message content, finish_reason={finish_reason}"
            )
        return content


def _build_vllm_extra_body(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if "enable_thinking" not in kwargs:
        return {}
    return {
        "chat_template_kwargs": {
            "enable_thinking": bool(kwargs["enable_thinking"]),
        }
    }
