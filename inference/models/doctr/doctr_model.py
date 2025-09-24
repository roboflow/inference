import os
import shutil
import tempfile
from time import perf_counter
from typing import Any, Tuple

import torch
from doctr.io import DocumentFile
from doctr.models import crnn_vgg16_bn, db_resnet50, ocr_predictor
from PIL import Image

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import DEVICE, MODEL_CACHE_DIR
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image

if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"


def _geometry_to_bbox(page_dimensions: Tuple[int, int], geometry: dict) -> list[int]:
    """Convert a geometry dictionary to a bounding box.

    Args:
        geometry (dict): A dictionary containing the geometry of the detected text.

    Returns:
        list[int]: A list representing the bounding box in the format [x_min, y_min, x_max, y_max].
    """
    x_min = int(page_dimensions[1] * geometry[0][0])
    y_min = int(page_dimensions[0] * geometry[0][1])
    x_max = int(page_dimensions[1] * geometry[1][0])
    y_max = int(page_dimensions[0] * geometry[1][1])
    return [x_min, y_min, x_max, y_max]


class DocTR(RoboflowCoreModel):
    def __init__(self, *args, model_id: str = "doctr_rec/crnn_vgg16_bn", **kwargs):
        """Initializes the DocTR model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.api_key = kwargs.get("api_key")
        self.dataset_id = "doctr"
        self.version_id = "default"
        self.endpoint = model_id
        model_id = model_id.lower()

        self.det_model = DocTRDet(api_key=kwargs.get("api_key"))
        self.rec_model = DocTRRec(api_key=kwargs.get("api_key"))

        os.makedirs(f"{MODEL_CACHE_DIR}/doctr/models/", exist_ok=True)

        detector_weights_path = (
            f"{MODEL_CACHE_DIR}/doctr/models/{self.det_model.version_id}.pt"
        )
        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/doctr_det/{self.det_model.version_id}/model.pt",
            detector_weights_path,
        )
        recognizer_weights_path = (
            f"{MODEL_CACHE_DIR}/doctr/models/{self.rec_model.version_id}.pt"
        )
        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/doctr_rec/{self.rec_model.version_id}/model.pt",
            recognizer_weights_path,
        )

        det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
        det_model.load_state_dict(
            torch.load(detector_weights_path, map_location=DEVICE, weights_only=True)
        )

        reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
        reco_model.load_state_dict(
            torch.load(recognizer_weights_path, map_location=DEVICE, weights_only=True)
        )

        self.model = ocr_predictor(
            det_arch=det_model,
            reco_arch=reco_model,
            pretrained=False,
        )
        self.task_type = "ocr"

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        self.det_model.clear_cache(delete_from_disk=delete_from_disk)
        self.rec_model.clear_cache(delete_from_disk=delete_from_disk)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        DocTR pre-processes images as part of its inference pipeline.

        Thus, no preprocessing is required here.
        """
        pass

    def infer_from_request(
        self, request: DoctrOCRInferenceRequest
    ) -> OCRInferenceResponse:
        t1 = perf_counter()
        result = self.infer(**request.dict())
        return OCRInferenceResponse(
            result=result[0],
            objects=result[1],
            time=perf_counter() - t1,
        )

    def infer(self, image: Any, **kwargs):
        """
        Run inference on a provided image.
            - image: can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.

        Args:
            request (DoctrOCRInferenceRequest): The inference request.

        Returns:
            OCRInferenceResponse: The inference response.
        """

        img = load_image(image)

        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            image = Image.fromarray(img[0])

            image.save(f.name)

            doc = DocumentFile.from_images([f.name])

            result = self.model(doc).export()

            blocks = result["pages"][0]["blocks"]
            page_dimensions = result["pages"][0]["dimensions"]

            words = [
                word
                for block in blocks
                for line in block["lines"]
                for word in line["words"]
            ]

            result = " ".join([word["value"] for word in words])
            objects = [
                {
                    "bounding_box": _geometry_to_bbox(
                        page_dimensions, word["geometry"]
                    ),
                    "confidence": float(word["objectness_score"]),
                    "string": word["value"],
                }
                for word in words
            ]

            # previous implementation only returned result, so using "extended" option to maintain backwards compatibility
            if kwargs.get("generate_bounding_boxes", False):
                return result, objects
            else:
                return result

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return ["model.pt"]


class DocTRRec(RoboflowCoreModel):
    def __init__(self, *args, model_id: str = "doctr_rec/crnn_vgg16_bn", **kwargs):
        """Initializes the DocTR model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.get_infer_bucket_file_list()

        super().__init__(*args, model_id=model_id, **kwargs)

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        super().clear_cache(delete_from_disk=delete_from_disk)

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return ["model.pt"]


class DocTRDet(RoboflowCoreModel):
    """DocTR class for document Optical Character Recognition (OCR).

    Attributes:
        doctr: The DocTR model.
        ort_session: ONNX runtime inference session.
    """

    def __init__(self, *args, model_id: str = "doctr_det/db_resnet50", **kwargs):
        """Initializes the DocTR model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        self.get_infer_bucket_file_list()

        super().__init__(*args, model_id=model_id, **kwargs)

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        super().clear_cache(delete_from_disk=delete_from_disk)

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return ["model.pt"]
