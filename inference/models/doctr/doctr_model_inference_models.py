from copy import copy
from time import perf_counter
from typing import Any, List, Tuple, Union

import torch
from PIL import Image

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
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
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference_models import AutoModel
from inference_models.models.doctr.doctr_torch import DocTR


class InferenceModelsDocTRAdapter(Model):
    def __init__(
        self, model_id: str = "doctr_rec/crnn_vgg16_bn", api_key: str = None, **kwargs
    ):
        print("Initializing InferenceModelsDocTRAdapter", flush=True)
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "ocr"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: DocTR = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        DocTR pre-processes images as part of its inference pipeline.

        Thus, no preprocessing is required here.
        """
        pass

    def infer_from_request(
        self, request: DoctrOCRInferenceRequest
    ) -> Union[OCRInferenceResponse, List]:
        if type(request.image) is list:
            response = []
            request_copy = copy.copy(request)
            for image in request.image:
                request_copy.image = image
                response.append(self.single_request(request=request_copy))
            return response
        return self.single_request(request)

    def single_request(self, request: DoctrOCRInferenceRequest) -> OCRInferenceResponse:
        t1 = perf_counter()
        result = self.infer(**request.dict())
        if not isinstance(result, tuple):
            result = (result, None, None)
        # maintaining backwards compatibility with previous implementation
        if request.generate_bounding_boxes:
            return OCRInferenceResponse(
                result=result[0],
                image=result[1],
                predictions=result[2],
                time=perf_counter() - t1,
            )
        else:
            return OCRInferenceResponse(
                result=result[0],
                time=perf_counter() - t1,
            )

    def infer(
        self, image: Any, **kwargs
    ) -> Union[
        str, Tuple[str, InferenceResponseImage, List[ObjectDetectionPrediction]]
    ]:
        """
        Run inference on a provided image.
            - image: can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.

        Args:
            request (DoctrOCRInferenceRequest): The inference request.

        Returns:
            OCRInferenceResponse: The inference response.
        """

        img = load_image_bgr(image)
        detected_texts, parsed_structures = self._model.infer(images=img)
        detected_text = detected_texts[0]
        parsed_structure = parsed_structures[0]
        image_height, image_width = img.shape[:2]
        predictions_for_image = []
        classes = self._model.class_names
        for instance_id in range(parsed_structure.xyxy.shape[0]):
            x_min, y_min, x_max, y_max = parsed_structure.xyxy[instance_id].tolist()
            width = x_max - x_min
            height = y_max - y_min
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            class_id = parsed_structure.class_id[instance_id].item()
            predictions_for_image.append(
                ObjectDetectionPrediction(
                    # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                    **{
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height,
                        "confidence": 1.0,  # confidence is not returned by the model
                        "class": classes[class_id],
                        "class_id": class_id,  # you can only prompt for one object at once
                    }
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            detected_text,
            InferenceResponseImage(width=image_width, height=image_height),
            predictions_for_image,
        )
