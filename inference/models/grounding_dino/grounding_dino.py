import os
import urllib.request
from time import perf_counter
from typing import Any, List

import torch
from groundingdino.util.inference import Model

from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import CLASS_AGNOSTIC_NMS, MODEL_CACHE_DIR
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_bgr, xyxy_to_xywh


class GroundingDINO(RoboflowCoreModel):
    """GroundingDINO class for zero-shot object detection.

    Attributes:
        model: The GroundingDINO model.
    """

    def __init__(
        self, *args, model_id="grounding_dino/groundingdino_swint_ogc", **kwargs
    ):
        """Initializes the GroundingDINO model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, model_id=model_id, **kwargs)

        GROUNDING_DINO_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, model_id)

        GROUNDING_DINO_CONFIG_PATH = os.path.join(
            GROUNDING_DINO_CACHE_DIR, "GroundingDINO_SwinT_OGC.py"
        )

        if not os.path.exists(GROUNDING_DINO_CACHE_DIR):
            os.makedirs(GROUNDING_DINO_CACHE_DIR)

        if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
            url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)

        self.model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=os.path.join(
                GROUNDING_DINO_CACHE_DIR, "groundingdino_swint_ogc.pth"
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.task_type = "object-detection"

    def preproc_image(self, image: Any):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        np_image = load_image_bgr(image)
        return np_image

    def infer_from_request(
        self,
        request: GroundingDINOInferenceRequest,
    ) -> ObjectDetectionInferenceResponse:
        """
        Perform inference based on the details provided in the request, and return the associated responses.
        """
        result = self.infer(**request.dict())
        return result

    def infer(
        self,
        image: InferenceRequestImage,
        text: List[str] = None,
        class_filter: list = None,
        box_threshold=0.5,
        text_threshold=0.5,
        class_agnostic_nms=CLASS_AGNOSTIC_NMS,
        **kwargs
    ):
        """
        Run inference on a provided image.
            - image: can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.

        Args:
            request (CVInferenceRequest): The inference request.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            GroundingDINOInferenceRequest: The inference response.
        """
        t1 = perf_counter()
        image = self.preproc_image(image)
        img_dims = image.shape

        detections = self.model.predict_with_classes(
            image=image,
            classes=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        self.class_names = text

        if class_agnostic_nms:
            detections = detections.with_nms(class_agnostic=True)
        else:
            detections = detections.with_nms()

        xywh_bboxes = [xyxy_to_xywh(detection) for detection in detections.xyxy]

        t2 = perf_counter() - t1

        responses = ObjectDetectionInferenceResponse(
            predictions=[
                ObjectDetectionPrediction(
                    **{
                        "x": xywh_bboxes[i][0],
                        "y": xywh_bboxes[i][1],
                        "width": xywh_bboxes[i][2],
                        "height": xywh_bboxes[i][3],
                        "confidence": detections.confidence[i],
                        "class": self.class_names[int(detections.class_id[i])],
                        "class_id": int(detections.class_id[i]),
                    }
                )
                for i, pred in enumerate(detections.xyxy)
                if not class_filter
                or self.class_names[int(pred[6])] in class_filter
                and detections.class_id[i] is not None
            ],
            image=InferenceResponseImage(width=img_dims[1], height=img_dims[0]),
            time=t2,
        )
        return responses

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return ["groundingdino_swint_ogc.pth"]
