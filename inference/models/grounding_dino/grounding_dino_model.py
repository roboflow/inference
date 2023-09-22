from groundingdino.util.inference import Model

import numpy as np
import onnxruntime
import torch
from time import perf_counter


from PIL import Image

from inference.core.data_models import (
    GroundingDINOInferenceRequest,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    InferenceResponseImage,
)

from inference.core.utils.image_utils import xyxy_to_xywh
from inference.core.models.roboflow import RoboflowCoreModel


class GroundingDINO(RoboflowCoreModel):
    """GroundingDINO class for zero-shot object detection.

    Attributes:
        doctr: The GroundingDINO model.
        ort_session: ONNX runtime inference session.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the GroundingDINO model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, model_id="groundingdino/1", **kwargs)

        self.groundingdino = Model

        self.sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

        self.ort_session = onnxruntime.InferenceSession(
            self.cache_file("decoder.onnx"),
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        pass

    def infer(self, request: GroundingDINOInferenceRequest):
        """
        Run inference on a provided image.

        Args:
            request (GroundingDINOInferenceRequest): The inference request.
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
            text_threshold=0.5
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
        )
        return responses
