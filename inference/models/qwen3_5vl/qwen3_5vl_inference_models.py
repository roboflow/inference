from time import perf_counter
from typing import TYPE_CHECKING, Any, List

import torch

from inference.core import logger
from inference.core.entities.responses import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    LMM_QWEN_BACKEND,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODE,
    VLLM_NATIVE_FALLBACK_ENABLED,
    VLLM_REQUEST_TIMEOUT,
)
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference_models import AutoModel

from inference.models.qwen3_5vl.vllm import QwenVLLMClient

if TYPE_CHECKING:
    from inference_models.models.qwen3_5.qwen3_5_hf import Qwen35HF

QWEN_BACKEND_NATIVE = "native"
QWEN_BACKEND_VLLM = "vllm"
QWEN_BACKEND_AUTO = "auto"
VLLM_ENABLED_MODES = {"sidecar", "remote"}


class InferenceModelsQwen35VLAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        self.model_id = model_id
        self._native_init_kwargs = dict(kwargs)
        self._model = None
        self._backend = resolve_qwen_backend(
            qwen_backend=LMM_QWEN_BACKEND,
            vllm_mode=VLLM_MODE,
        )

        self.task_type = "lmm"

        if self._backend == QWEN_BACKEND_VLLM:
            self._vllm_client = QwenVLLMClient(
                base_url=VLLM_BASE_URL,
                api_key=VLLM_API_KEY,
                request_timeout=VLLM_REQUEST_TIMEOUT,
            )
            return

        self._vllm_client = None
        self._ensure_native_model()

    def _ensure_native_model(self) -> None:
        if self._model is not None:
            return
        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=self._native_init_kwargs.get("countinference"),
            service_secret=self._native_init_kwargs.get("service_secret"),
        )

        self._model: "Qwen35HF" = AutoModel.from_pretrained(
            model_id_or_path=self.model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **self._native_init_kwargs,
        )

    def infer_from_request(
        self,
        request,
    ):
        if self._backend != QWEN_BACKEND_VLLM:
            return super().infer_from_request(request=request)
        try:
            return self._infer_from_request_vllm(request=request)
        except Exception as error:
            if not VLLM_NATIVE_FALLBACK_ENABLED:
                raise
            logger.warning(
                "Qwen vLLM inference failed for model %s; falling back to native backend: %s",
                self.model_id,
                error,
            )
            self._ensure_native_model()
            return super().infer_from_request(request=request)

    def _infer_from_request_vllm(self, request) -> LMMInferenceResponse:
        if isinstance(request.image, list):
            raise ValueError("This model does not support batched-inference.")
        if self._vllm_client is None:
            raise RuntimeError("vLLM client is not configured for Qwen backend.")
        t1 = perf_counter()
        np_image = load_image_bgr(
            request.image,
            disable_preproc_auto_orient=request.disable_preproc_auto_orient,
        )
        raw_response = self._vllm_client.generate(
            model_id=self.model_id,
            image_bgr=np_image,
            prompt=request.prompt,
            enable_thinking=request.enable_thinking,
            max_tokens=request.max_new_tokens,
        )
        response = LMMInferenceResponse(
            response=raw_response,
            image=InferenceResponseImage(
                width=np_image.shape[1],
                height=np_image.shape[0],
            ),
        )
        response.time = perf_counter() - t1
        if request.id:
            response.inference_id = request.id
        return response

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
        return (
            self._model.pre_process_generation(np_image, prompt, **mapped_kwargs),
            input_shape,
        )

    def predict(self, inputs, **kwargs) -> torch.Tensor:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.generate(inputs, **mapped_kwargs)

    def postprocess(
        self,
        predictions: torch.Tensor,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        result = self._model.post_process_generation(predictions, **mapped_kwargs)[0]
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


def resolve_qwen_backend(qwen_backend: str, vllm_mode: str) -> str:
    if qwen_backend == QWEN_BACKEND_NATIVE:
        return QWEN_BACKEND_NATIVE
    if qwen_backend == QWEN_BACKEND_AUTO:
        if vllm_mode in VLLM_ENABLED_MODES:
            return QWEN_BACKEND_VLLM
        return QWEN_BACKEND_NATIVE
    if qwen_backend == QWEN_BACKEND_VLLM:
        if vllm_mode not in VLLM_ENABLED_MODES:
            raise ValueError(
                "LMM_QWEN_BACKEND=vllm requires VLLM_MODE to be one of: "
                f"{sorted(VLLM_ENABLED_MODES)}"
            )
        return QWEN_BACKEND_VLLM
    raise ValueError(
        f"Unknown LMM_QWEN_BACKEND={qwen_backend!r}. "
        f"Expected one of: {QWEN_BACKEND_NATIVE}, {QWEN_BACKEND_VLLM}, {QWEN_BACKEND_AUTO}."
    )
