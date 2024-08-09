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

logging.set_verbosity_error()


class TrOCR(RoboflowCoreModel):
    def __init__(self, *args, model_id=f"microsoft/trocr-base-printed", **kwargs):
        # super().__init__(*args, model_id=model_id, **kwargs) TODO: Add model cache
        self.model_id = model_id
        self.endpoint = model_id
        self.cache_dir = os.path.join(MODEL_CACHE_DIR, self.endpoint + "/")
        self.cache_dir = model_id  # TODO: Remove (temp)

        self.model = VisionEncoderDecoderModel.from_pretrained(self.cache_dir).eval()

        self.processor = TrOCRProcessor.from_pretrained(self.cache_dir)
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
        response = OCRInferenceResponse(response=text)
        response.time = perf_counter() - t1
        return response

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        # TODO: Add model files to bucket
        # TODO: Add list of model files here
        return []


if __name__ == "__main__":
    import cv2

    path = input("Image path:")
    image = cv2.imread(path)
    trocr = TrOCR()
    print(trocr.infer(image))
