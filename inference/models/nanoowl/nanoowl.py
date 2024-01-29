import os
import urllib.request
from time import perf_counter
from typing import Any

import torch
from nanoowl.owl_predictor import OwlPredictor

from inference.core.entities.requests.nanoowl import NanoOwlInferenceRequest
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import MODEL_CACHE_DIR
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb, xyxy_to_xywh


class NanoOwl(RoboflowCoreModel):
    """NanoOwl class for zero-shot object detection.

    Attributes:
        model: The NanoOwl model.
    """

    def __init__(
        self, *args, model_id="google/owlvit-base-patch32", **kwargs
    ):
        """Initializes the NanoOwl model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, model_id=model_id, **kwargs)

        NANOOWL_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, model_id)

        if not os.path.exists(NANOOWL_CACHE_DIR):
            os.makedirs(NANOOWL_CACHE_DIR)

        self.model = OwlPredictor(
            "google/owlvit-base-patch32",
            device="cpu",
            image_encoder_engine="data/owlvit-base-patch32-image-encoder.engine",
        )

    def preproc_image(self, image: Any):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        np_image = load_image_rgb(image)
        return np_image

    def infer_from_request(
        self,
        request: NanoOwlInferenceRequest,
    ) -> ObjectDetectionInferenceResponse:
        """
        Perform inference based on the details provided in the request, and return the associated responses.
        """
        result = self.infer(**request.dict())
        return result

    def infer(
        self, image: Any = None, text: list = None, class_filter: list = None, **kwargs
    ):
        """
        Run inference on a provided image.

        Args:
            request (CVInferenceRequest): The inference request.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            GroundingDINOInferenceRequest: The inference response.
        """
        t1 = perf_counter()
        image = self.preproc_image(image)
        img_dims = image.shape

        print(text)

        detections = self.model.predict(image=image, text=text, threshold=0.1)

        print(detections)

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
                if not class_filter or self.class_names[int(pred[6])] in class_filter
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
        return []