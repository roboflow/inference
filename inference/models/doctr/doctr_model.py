import base64
from io import BytesIO
from time import perf_counter
from typing import Any, List, Optional, Union

import numpy as np
import onnxruntime
import torch

from doctr.io import DocumentFile

from PIL import Image

from inference.core.data_models import (
    DoctrOCRInferenceRequest,
    DoctrOCRInferenceResponse,
)

from inference.core.models.roboflow import RoboflowCoreModel

from doctr.models import ocr_predictor


class DocTR(RoboflowCoreModel):
    """DocTR class for document Optical Character Recognition (OCR).

    Attributes:
        doctr: The DocTR model.
        ort_session: ONNX runtime inference session.
    """

    def __init__(self, *args, model_id: str = "db_resnet50", **kwargs):
        """Initializes the DocTR model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, model_id=model_id + "/1", **kwargs)

        self.doctr = ocr_predictor(
            det_arch=model_id, reco_arch="crnn_vgg16_bn", pretrained=True
        )

        self.sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

        self.ort_session = onnxruntime.InferenceSession(
            self.cache_file("decoder.onnx"),
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        DocTR pre-processes images as part of its inference pipeline.

        Thus, no preprocessing is required here.
        """
        pass

    def infer(self, request: DoctrOCRInferenceRequest):
        """
        Run inference on a provided image.

        Args:
            request (DoctrOCRInferenceRequest): The inference request.

        Returns:
            DoctrOCRInferenceResponse: The inference response.
        """
        t1 = perf_counter()

        img = self.load_image(request.image.type, request.image.value)

        bytes_img = self.pil_to_bytes(img)

        doc = DocumentFile.from_images(bytes_img)

        result = self.predictor(doc)
        t2 = perf_counter() - t1

        return DoctrOCRInferenceResponse(
            result=result,
            time=t2,
        )
