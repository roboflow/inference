from time import perf_counter
from typing import Any, Tuple

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, logging

from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import DEVICE
from inference.core.exceptions import InvalidModelIDError
from inference.core.models.base import PreprocessReturnMetadata
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.set_verbosity_error()

SUPPORTED_MODEL_IDS = {
    "microsoft/trocr-small-printed",
    "microsoft/trocr-base-printed",
    "microsoft/trocr-large-printed",
}


class TrOCR(RoboflowCoreModel):
    def __init__(self, *args, model_id, **kwargs):
        model_id = model_id.replace("trocr/", "microsoft/")
        if model_id not in SUPPORTED_MODEL_IDS:
            raise InvalidModelIDError(
                f"Requested TROCR in version: {model_id}. Supported versions: {SUPPORTED_MODEL_IDS}"
            )
        self.endpoint = model_id
        self.model_id = model_id
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(self.model_id).eval().to(DEVICE)
        )
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
        text = self.infer(**request.model_dump())
        t2 = perf_counter()
        response = OCRInferenceResponse(result=text[0], time=t2 - t1)
        return response
