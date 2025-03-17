import os
import shutil
import tempfile
from time import perf_counter
from typing import Any, List, Union

from doctr import models as models
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.env import MODEL_CACHE_DIR
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image


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

        os.environ["DOCTR_CACHE_DIR"] = os.path.join(MODEL_CACHE_DIR, "doctr_rec")

        self.det_model = DocTRDet(api_key=kwargs.get("api_key"))
        self.rec_model = DocTRRec(api_key=kwargs.get("api_key"))

        os.makedirs(f"{MODEL_CACHE_DIR}/doctr_rec/models/", exist_ok=True)
        os.makedirs(f"{MODEL_CACHE_DIR}/doctr_det/models/", exist_ok=True)

        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/doctr_det/db_resnet50/model.pt",
            f"{MODEL_CACHE_DIR}/doctr_det/models/db_resnet50-ac60cadc.pt",
        )
        shutil.copyfile(
            f"{MODEL_CACHE_DIR}/doctr_rec/crnn_vgg16_bn/model.pt",
            f"{MODEL_CACHE_DIR}/doctr_rec/models/crnn_vgg16_bn-9762b0b0.pt",
        )

        self.model = ocr_predictor(
            det_arch=self.det_model.version_id,
            reco_arch=self.rec_model.version_id,
            pretrained=True,
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
