import copy
import os
import shutil
import tempfile
import uuid
from time import perf_counter
from typing import Any, List, Tuple, Union
from unittest import result

import easyocr
import numpy as np
import torch
from PIL import Image

from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponse,
    InferenceResponseImage,
    ObjectDetectionPrediction,
)
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import DEVICE
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image, load_image_with_inferred_type

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
        self.task_type = "ocr"
        self.recognizer = model_id.split("/")[1]

    def predict(self, image_in: np.ndarray, prompt="", history=None, **kwargs):
        language_codes = kwargs.get("language_codes", ["en"])
        quantize = kwargs.get("quantize", False)
        recognition_weights_path = self.cache_file("weights.pt")
        detector_weights_path = self.cache_file("craft_mlt_25k.pth")
        with tempfile.TemporaryDirectory(prefix="inference-easyocr-") as runtime_dir:
            _stage_runtime_weight(
                source_path=recognition_weights_path,
                destination_path=os.path.join(
                    runtime_dir,
                    f"{self.recognizer}.pth",
                ),
            )
            _stage_runtime_weight(
                source_path=detector_weights_path,
                destination_path=os.path.join(
                    runtime_dir,
                    "craft_mlt_25k.pth",
                ),
            )
            reader = easyocr.Reader(
                language_codes,
                download_enabled=False,
                user_network_directory=runtime_dir,
                model_storage_directory=runtime_dir,
                detect_network="craft",
                recog_network=self.recognizer,
                detector=True,
                recognizer=True,
                gpu=True,
                quantize=quantize,
            )
            results = reader.readtext(image_in)
        # convert native EasyOCR results from numpy to standard python types
        results = [
            (
                [
                    [x.item() if not isinstance(x, (int, float)) else x for x in c]
                    for c in res[0]
                ],
                res[1],
                res[2].item() if not isinstance(res[2], (int, float)) else res[2],
            )
            for res in results
        ]

        return results

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return predictions, preprocess_return_metadata

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        image = load_image(image)[0]
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
        prediction_result, image_metadata = self.infer(**request.dict())
        strings = [res[1] for res in prediction_result]
        return OCRInferenceResponse(
            result=" ".join(strings),
            image=image_metadata,
            predictions=[
                ObjectDetectionPrediction(
                    **{
                        "x": box[0][0] + (box[2][0] - box[0][0]) // 2,
                        "y": box[0][1] + (box[2][1] - box[0][1]) // 2,
                        "width": box[2][0] - box[0][0],
                        "height": box[2][1] - box[0][1],
                        "confidence": float(confidence),
                        "class": string,
                        "class_id": 0,
                        "detection_id": str(uuid.uuid4()),
                    }
                )
                for box, string, confidence in prediction_result
            ],
            time=perf_counter() - t1,
        )

    def get_infer_bucket_file_list(self) -> List[str]:
        return ["weights.pt", "craft_mlt_25k.pth"]


def _stage_runtime_weight(source_path: str, destination_path: str) -> None:
    """Expose a cached weight under the filename expected by EasyOCR.

    The cache itself may be mounted read-only. Prefer a temporary symlink so
    large recognition weights are not duplicated; copy only on platforms that
    cannot create the link.
    """

    if os.path.islink(source_path) or not os.path.isfile(source_path):
        raise FileNotFoundError(
            f"EasyOCR cached weight is missing or not a regular file: {source_path}"
        )
    try:
        os.symlink(os.path.abspath(source_path), destination_path)
    except (NotImplementedError, OSError):
        shutil.copyfile(source_path, destination_path)
