from typing import Any, List

import torch

from inference.core.entities.responses import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
)
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference_models import AutoModel
from inference_models.entities import ImageDimensions
from inference_models.models.florence2.florence2_hf import Florence2HF


class InferenceModelsFlorence2Adapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        self.task_type = "lmm"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: Florence2HF = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def preprocess(self, image: Any, prompt: str, **kwargs):
        is_batch = isinstance(image, list)
        if is_batch:
            raise ValueError("This model does not support batched-inference.")
        np_image = load_image_bgr(
            image,
            disable_preproc_auto_orient=kwargs.get(
                "disable_preproc_auto_orient", False
            ),
        )
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process_generation(np_image, prompt, **mapped_kwargs)[:2]

    def predict(self, inputs, **kwargs) -> torch.Tensor:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.generate(inputs, **mapped_kwargs)

    def postprocess(
        self,
        predictions: torch.Tensor,
        preprocess_return_metadata: List[ImageDimensions],
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        task = kwargs.get("prompt", "").split(">")[0] + ">"
        if not task:
            task = None
        result = self._model.post_process_generation(
            predictions,
            task=task,
            image_dimensions=preprocess_return_metadata,
            **mapped_kwargs,
        )[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [
            LMMInferenceResponse(
                response=result,
                image=InferenceResponseImage(
                    width=preprocess_return_metadata[0].width,
                    height=preprocess_return_metadata[0].height,
                ),
            )
        ]

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
