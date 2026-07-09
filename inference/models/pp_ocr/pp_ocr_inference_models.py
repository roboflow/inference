import copy
import uuid
from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core.entities.requests.pp_ocr import PPOCRInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionPrediction,
)
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.models.base import Model, PreprocessReturnMetadata
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference_models.models.auto_loaders.core import AutoModel
from inference_models.models.pp_ocrv6.pp_ocrv6_pipeline import PPOCRv6Pipeline


def _parse_det_rec(model_id: str) -> Tuple[str, str]:
    parts = model_id.split("/")
    token = parts[1] if len(parts) > 1 and parts[1] else "small"
    if "-" in token:
        det, rec = token.split("-", 1)
        return det or "none", rec or "none"
    return token, token


class InferenceModelsPPOCRAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "ocr"

        det, rec = _parse_det_rec(model_id)

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        det_model = (
            AutoModel.from_pretrained(
                f"pp-ocrv6-det/{det}",
                api_key=self.api_key,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                weights_provider_extra_headers=extra_weights_provider_headers,
                backend=backend,
                **kwargs,
            )
            if det != "none"
            else None
        )
        rec_model = (
            AutoModel.from_pretrained(
                f"pp-ocrv6-rec/{rec}",
                api_key=self.api_key,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                weights_provider_extra_headers=extra_weights_provider_headers,
                backend=backend,
                **kwargs,
            )
            if rec != "none"
            else None
        )
        self._pipeline = PPOCRv6Pipeline(det_model=det_model, rec_model=rec_model)

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        image = load_image_bgr(image)
        return image, PreprocessReturnMetadata(
            {
                "image_metadata": InferenceResponseImage(
                    width=image.shape[1], height=image.shape[0]
                )
            }
        )

    def postprocess(
        self,
        predictions: Tuple[Any, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions[0], preprocess_return_metadata["image_metadata"]

    def predict(self, image_in: np.ndarray, **kwargs):
        result = self._pipeline.infer(images=image_in, **kwargs)[0]
        return (result,)

    def infer_from_request(
        self, request: PPOCRInferenceRequest
    ) -> Union[OCRInferenceResponse, List]:
        if type(request.image) is list:
            response = []
            request_copy = copy.copy(request)
            for image in request.image:
                request_copy.image = image
                response.append(self.single_request(request=request_copy))
            return response
        return self.single_request(request)

    def single_request(
        self, request: PPOCRInferenceRequest
    ) -> OCRInferenceResponse:
        t1 = perf_counter()
        pipeline_result, image_metadata = self.infer(**request.model_dump())
        detections = pipeline_result.detections
        # detections is None when the detection stage was disabled
        # (recognition-only pipeline) - no boxes to report then.
        boxes = detections.xyxy.tolist() if detections is not None else []
        confidences = detections.confidence.tolist() if detections is not None else []
        line_texts = pipeline_result.line_texts
        predictions = []
        for index, (box, confidence) in enumerate(zip(boxes, confidences)):
            text = line_texts[index] if index < len(line_texts) else ""
            predictions.append(
                ObjectDetectionPrediction(
                    **{
                        "x": box[0] + (box[2] - box[0]) / 2,
                        "y": box[1] + (box[3] - box[1]) / 2,
                        "width": box[2] - box[0],
                        "height": box[3] - box[1],
                        "confidence": float(confidence),
                        "class": text,
                        "class_id": 0,
                        "detection_id": str(uuid.uuid4()),
                    }
                )
            )
        return OCRInferenceResponse(
            result=pipeline_result.text,
            image=image_metadata,
            predictions=predictions,
            time=perf_counter() - t1,
        )
