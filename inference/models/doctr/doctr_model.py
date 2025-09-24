import os
import shutil
import tempfile
from time import perf_counter
from typing import Any

import torch

from doctr.io import DocumentFile
from doctr.models import ocr_predictor, db_resnet50, crnn_vgg16_bn

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

        os.environ["DOCTR_CACHE_DIR"] = os.path.join(MODEL_CACHE_DIR, "doctr")

        self.det_model = DocTRDet(api_key=kwargs.get("api_key"))
        self.rec_model = DocTRRec(api_key=kwargs.get("api_key"))

        print("===det_model===", self.det_model.version_id)
        print("===rec_model===", self.rec_model.version_id)

        os.makedirs(f"{MODEL_CACHE_DIR}/doctr/models/", exist_ok=True)

        detector_weights_path = f"{MODEL_CACHE_DIR}/doctr/models/{self.det_model.version_id}.pt"
        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/doctr_det/{self.det_model.version_id}/model.pt",
            detector_weights_path,
        )
        recognizer_weights_path = f"{MODEL_CACHE_DIR}/doctr/models/{self.rec_model.version_id}.pt"
        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/doctr_rec/{self.rec_model.version_id}/model.pt",
            recognizer_weights_path,
        )

        det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
        det_model.load_state_dict(torch.load(detector_weights_path, map_location=DEVICE, weights_only=True))
        #detector.from_pretrained(detector_weights_path, map_location=DEVICE)

        reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
        #recognizer.from_pretrained(recognizer_weights_path, map_location=DEVICE)
        reco_model.load_state_dict(torch.load(recognizer_weights_path, map_location=DEVICE, weights_only=True))

        #print("===detector===", detector)
        #print("===recognizer===", recognizer)

        self.model = ocr_predictor(
            det_arch=det_model,
            reco_arch=reco_model,
            pretrained=False,
        )
        #self.model = ocr_predictor(det_arch=detector, reco_arch=recognizer)
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
            result=result,
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

            print(result)

            result = result["pages"][0]["blocks"]

            result = [
                " ".join([word["value"] for word in line["words"]])
                for block in result
                for line in block["lines"]
            ]

            result = " ".join(result)

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
