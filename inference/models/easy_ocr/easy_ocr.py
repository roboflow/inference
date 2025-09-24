import os
import shutil
from time import perf_counter
from typing import Any, List, Tuple, Union

import easyocr
import numpy as np
import torch

from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.responses.easy_ocr import EasyOCRInferenceResponse
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import DEVICE, MODEL_CACHE_DIR
from inference.core.models.roboflow import RoboflowCoreModel
from PIL import Image

from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image

if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
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

        self.model_id = model_id
        self.version = model_id.split("/")[1]
        self.languages = kwargs.get("languages", ["en"])

        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/{model_id}/weights.pt",
            f"{MODEL_CACHE_DIR}/{model_id}/{self.version}.pth",
        )

        self.reader = easyocr.Reader(
            self.languages,
            download_enabled=False,
            user_network_directory=f'/tmp/cache/{self.model_id}/',
            model_storage_directory=f'/tmp/cache/{self.model_id}/',
            detect_network='craft',
            recog_network=self.version,
            detector=True,
            recognizer=True,
            gpu=True
            )

    def predict(self, image_in: Image.Image, prompt="", history=None, **kwargs):
        try:

            img = np.array(image_in[0]["value"])

            results = self.reader.readtext(img)

            # convert native EasyOCR results from numpy arrays to standard python types
            results = [([[int(x) for x in c] for c in res[0]], res[1], float(res[2])) for res in results]

            return (results,)
        except Exception as e:
            raise

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
    ) -> EasyOCRInferenceResponse:
        t1 = perf_counter()
        result = self.infer(**request.dict())
        return EasyOCRInferenceResponse(
            result=result,
            time=perf_counter() - t1,
        )

    def make_response(
        self, *args, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        return EasyOCRInferenceResponse(
                    result=result,
                    time=perf_counter() - t1,
                )

    def get_infer_bucket_file_list(self) -> List[str]:
        return ["weights.pt", "craft_mlt_25k.pth"]
