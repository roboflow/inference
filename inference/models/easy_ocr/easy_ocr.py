import os
import shutil
from time import perf_counter
from typing import Any, List, Tuple, Union
import uuid

import cv2
import easyocr
import numpy as np
from pydantic import ValidationError
import torch
from PIL import Image

from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.responses.inference import InferenceResponse, ObjectDetectionPrediction
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import DEVICE, MODEL_CACHE_DIR
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.models.types import PreprocessReturnMetadata

if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"


def _to_bounding_box(bbox: List[List[int]]) -> List[int]:
    """Converts bounding boxes from corner points to [x_min, y_min, x_max, y_max] format.

    Args:
        boxes ([List[List[int]]): Bounding boxes in corner points format.

    Returns:
        [List[int]: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """

    x = bbox[0][0]
    y = bbox[0][1]
    width = bbox[2][0]-x
    height = bbox[2][1]-y
    return [x, y, width, height]


class EasyOCR(RoboflowCoreModel):
    """Roboflow EasyOCR model implementation.

    This class is responsible for handling the EasyOCR model, including
    loading the model, preprocessing the input, and performing inference.
    """

    def __init__(
        self,
        model_id: str = "easy_ocr/english_g2",
        device: str = DEVICE,
        *args,
        **kwargs,
    ):
        """Initializes EasyOCR with the given arguments and keyword arguments."""

        super().__init__(model_id=model_id.lower(), *args, **kwargs)
        self.device = device

        self.recognizer = model_id.split("/")[1]

        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/{model_id}/weights.pt",
            f"{MODEL_CACHE_DIR}/{model_id}/{self.recognizer}.pth",
        )

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        language_codes = kwargs.get("language_codes", ["en"])
        quantize = kwargs.get("quantize", False)

        reader = easyocr.Reader(
            language_codes,
            download_enabled=False,
            user_network_directory=f"/tmp/cache/easy_ocr/{self.recognizer}/",
            model_storage_directory=f"/tmp/cache/easy_ocr/{self.recognizer}/",
            detect_network="craft",
            recog_network=self.recognizer,
            detector=True,
            recognizer=True,
            gpu=True,
            quantize=quantize,
        )

        img = np.array(image_in[0]["value"])

        results = reader.readtext(img)

        # convert native EasyOCR results from numpy to standard python types
        results = [
            ([[x.item() for x in c] for c in res[0]], res[1], res[2].item())
            for res in results
        ]

        return (results,)

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions[0]

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        return image, kwargs

    def infer_from_request(
        self, request: EasyOCRInferenceRequest
    ) -> OCRInferenceResponse:
        t1 = perf_counter()
        result = self.infer(**request.dict())

        strings = [res[1] for res in result]

        return OCRInferenceResponse(
            result=" ".join(strings),
            predictions=[
                {
                    "x": box[0][0]+(box[2][0]-box[0][0])//2,
                    "y": box[0][1]+(box[2][1]-box[0][1])//2,
                    "width": box[2][0]-box[0][0],
                    "height": box[2][1]-box[0][1],
                    "confidence": float(confidence),
                    "class": string,
                    "class_id": 0,
                    "detection_id": str(uuid.uuid4()),
                } for box, string, confidence in result],
            time=perf_counter() - t1,
        )

    def make_response(
        self, *args, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        """Constructs an object detection response.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def get_infer_bucket_file_list(self) -> List[str]:
        return ["weights.pt", "craft_mlt_25k.pt"]
