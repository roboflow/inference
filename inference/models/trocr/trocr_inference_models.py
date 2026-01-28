from time import perf_counter
from typing import Any, Tuple

import numpy as np

from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
)
from inference.core.models.base import Model, PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image_rgb
from inference_models import AutoModel


class InferenceModelsTrOCRAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "ocr"

        self._model: TROcrHF = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            **kwargs,
        )

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        return load_image_rgb(image), PreprocessReturnMetadata({})

    def postprocess(
        self,
        predictions: Tuple[str],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions[0]

    def predict(self, image_in: np.ndarray, **kwargs):
        results = self._model.infer(images=image_in, **kwargs)[0]
        return (results,)

    def infer_from_request(
        self, request: TrOCRInferenceRequest
    ) -> OCRInferenceResponse:
        t1 = perf_counter()
        text = self.infer(**request.model_dump())
        t2 = perf_counter()
        response = OCRInferenceResponse(result=text[0], time=t2 - t1)
        return response
