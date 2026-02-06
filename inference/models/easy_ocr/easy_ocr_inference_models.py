import copy
import uuid
from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionPrediction,
)
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
)
from inference.core.models.base import Model
from inference.core.models.inference_models_adapters import (
    get_extra_weights_provider_headers,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image_bgr
from inference_models import AutoModel, Detections
from inference_models.models.easy_ocr.easy_ocr_torch import EasyOCRTorch


class InferenceModelsEasyOCRAdapter(Model):
    """Roboflow EasyOCR model implementation.

    This class is responsible for handling the EasyOCR model, including
    loading the model, preprocessing the input, and performing inference.
    """

    def __init__(
        self, model_id: str = "easy_ocr/english_g2", api_key: str = None, **kwargs
    ):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "ocr"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: EasyOCRTorch = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            extra_weights_provider_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def predict(self, image_in: np.ndarray, **kwargs) -> Tuple[str, Detections]:
        parsed_texts, parsed_structures = self._model.infer(images=image_in, **kwargs)
        parsed_text = parsed_texts[0]
        parsed_structure = parsed_structures[0]
        return parsed_text, parsed_structure

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions, preprocess_return_metadata

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, InferenceResponseImage]:
        image = load_image_bgr(image)
        return image, InferenceResponseImage(
            width=image.shape[1], height=image.shape[0]
        )

    def infer_from_request(
        self, request: EasyOCRInferenceRequest
    ) -> Union[OCRInferenceResponse, List]:
        if type(request.image) is list:
            response = []
            request_copy = copy.copy(request)
            for image in request.image:
                request_copy.image = image
                response.append(self.single_request(request=request_copy))
            return response
        return self.single_request(request)

    def single_request(self, request: EasyOCRInferenceRequest) -> OCRInferenceResponse:
        t1 = perf_counter()
        kwargs = request.dict()
        kwargs["confidence"] = 0.0
        prediction_result, image_metadata = self.infer(**kwargs)
        predictions_for_image = []
        for instance_id in range(prediction_result[1].xyxy.shape[0]):
            x_min, y_min, x_max, y_max = prediction_result[1].xyxy[instance_id].tolist()
            width = x_max - x_min
            height = y_max - y_min
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            predictions_for_image.append(
                ObjectDetectionPrediction(
                    # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                    **{
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height,
                        "confidence": 1.0,  # confidence is not returned by the model
                        "class": prediction_result[1].bboxes_metadata[instance_id][
                            "text"
                        ],
                        "class_id": 0,  # you can only prompt for one object at once
                        "detection_id": str(uuid.uuid4()),
                    }
                )
            )
        return OCRInferenceResponse(
            result=prediction_result[0],
            image=image_metadata,
            predictions=predictions_for_image,
            time=perf_counter() - t1,
        )
