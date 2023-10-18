from groundingdino.util.inference import Model

import numpy as np
import onnxruntime
import torch
from time import perf_counter


from PIL import Image
import urllib.request

from inference.core.data_models import (
    CVInferenceRequest,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    InferenceResponseImage,
)

from inference.core.utils.image_utils import xyxy_to_xywh
from inference.core.models.roboflow import RoboflowCoreModel

import os

from inference.core.env import MODEL_CACHE_DIR


class GroundingDINO(RoboflowCoreModel):
    """GroundingDINO class for zero-shot object detection.

    Attributes:
        model: The GroundingDINO model.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the GroundingDINO model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, model_id="groundingdino/groundingdino_swint_ogc", **kwargs)

        self._load_model()

    def _load_model(self):
        GROUDNING_DINO_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, "groundingdino")

        GROUNDING_DINO_CONFIG_PATH = os.path.join(
            GROUDNING_DINO_CACHE_DIR, "GroundingDINO_SwinT_OGC.py"
        )
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
            GROUDNING_DINO_CACHE_DIR, "groundingdino_swint_ogc.pth"
        )

        if not os.path.exists(GROUDNING_DINO_CACHE_DIR):
            os.makedirs(GROUDNING_DINO_CACHE_DIR)

        if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
            url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)

        self.get_infer_bucket_file_list()

        # TODO: move weights if required here

        self.model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        pass

    def infer(self, request: CVInferenceRequest):
        """
        Run inference on a provided image.

        Args:
            request (CVInferenceRequest): The inference request.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            GroundingDINOInferenceRequest: The inference response.
        """
        t1 = perf_counter()

        image = self.load_image(request.image.type, request.image.value)
        img_dims = image.shape

        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=[request.classes],
            box_threshold=0.5,
            text_threshold=0.5,
        )

        xywh_bboxes = [xyxy_to_xywh(detection.xyxy) for detection in detections.xyxy]

        t2 = perf_counter() - t1

        # detections has .xyxy, confidence, and class_id

        responses = ObjectDetectionInferenceResponse(
            predictions=[
                ObjectDetectionPrediction(
                    **{
                        "x": xywh_bboxes[i][0],
                        "y": xywh_bboxes[i][1],
                        "width": xywh_bboxes[i][2],
                        "height": xywh_bboxes[i][3],
                        "confidence": pred.confidence[i],
                        "class": self.class_names[int(pred.class_id[i])],
                        "class_id": int(pred.class_id[i]),
                    }
                )
                for pred, i in enumerate(detections)
                if not request.class_filter
                or self.class_names[int(pred[6])] in request.class_filter
            ],
            image=InferenceResponseImage(width=img_dims[1], height=img_dims[0]),
            time=t2
        )
        return responses

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return ["groundingdino_swint_ogc.pth"]
