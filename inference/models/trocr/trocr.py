import os

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, logging

from inference.core.env import MODEL_CACHE_DIR

cache_dir = os.path.join(MODEL_CACHE_DIR)
import os
from time import perf_counter
from typing import Any, List, Tuple, Union

import torch
from PIL import Image

from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import API_KEY, MODEL_CACHE_DIR  # TODO: Add version ID to env
from inference.core.models.base import PreprocessReturnMetadata
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb

from inference.core.logger import logger

logging.set_verbosity_error()


class TrOCR(RoboflowCoreModel):
    def __init__(self, *args, model_id, **kwargs):
        self.model_id = "microsoft/trocr-base-printed"
        logger.debug(f"TrOCR Model ID: {self.model_id}")

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).eval()

        self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        self.task_type = "ocr"

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[Image.Image, PreprocessReturnMetadata]:
        pil_image = Image.fromarray(load_image_rgb(image))

        return pil_image, PreprocessReturnMetadata({})

    def postprocess(
        self,
        predictions: Tuple[str],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions[0]

    def predict(self, image_in: Image.Image, **kwargs):
        model_inputs = self.processor(image_in, return_tensors="pt").to(
            self.model.device
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs)
            decoded = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        return (decoded,)

    def infer_from_request(
        self, request: TrOCRInferenceRequest
    ) -> OCRInferenceResponse:
        t1 = perf_counter()
        text = self.infer(**request.dict())
        t2 = perf_counter()
        response = OCRInferenceResponse(result=text[0], time=t2 - t1)
        return response


if __name__ == "__main__":
    import cv2

    path = input("Image path:")
    image = cv2.imread(path)
    trocr = TrOCR()
    print(trocr.infer(image))
