from time import perf_counter
from typing import Any, Optional

import numpy as np
from ultralytics import YOLO

from inference.core.cache import cache
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import DEFAULT_CLASS_AGNOSTIC_NMS, DEFAULT_MAX_CANDIDATES
from inference.core.models.defaults import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESH,
    DEFAUlT_MAX_DETECTIONS,
)
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.hash import get_string_list_hash
from inference.core.utils.image_utils import load_image_rgb


class YOLOWorld(RoboflowCoreModel):
    """GroundingDINO class for zero-shot object detection.

    Attributes:
        model: The GroundingDINO model.
    """

    task_type = "object-detection"

    def __init__(self, *args, model_id="yolo_world/l", **kwargs):
        """Initializes the YOLO-World model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, model_id=model_id, **kwargs)

        self.model = YOLO(self.cache_file("yolo-world.pt"))
        self.class_names = None

    def preproc_image(self, image: Any):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        np_image = load_image_rgb(image)
        return np_image[:, :, ::-1]

    def infer_from_request(
        self,
        request: YOLOWorldInferenceRequest,
    ) -> ObjectDetectionInferenceResponse:
        """
        Perform inference based on the details provided in the request, and return the associated responses.
        """
        result = self.infer(**request.dict())
        return result

    def infer(
        self,
        image: Any = None,
        text: list = None,
        confidence: float = DEFAULT_CONFIDENCE,
        max_detections: Optional[int] = DEFAUlT_MAX_DETECTIONS,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        class_agnostic_nms=DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
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

        if text is not None and text != self.class_names:
            self.set_classes(text)
        if self.class_names is None:
            raise ValueError(
                "Class names not set and not provided in the request. Must set class names before inference or provide them via the argument `text`."
            )
        results = self.model.predict(
            image,
            conf=confidence,
            verbose=False,
        )[0]

        t2 = perf_counter() - t1

        if len(results) > 0:
            bbox_array = np.array([box.xywh.tolist()[0] for box in results.boxes])
            conf_array = np.array([[float(box.conf)] for box in results.boxes])
            cls_array = np.array(
                [self.get_cls_conf_array(int(box.cls)) for box in results.boxes]
            )

            pred_array = np.concatenate([bbox_array, conf_array, cls_array], axis=1)
            pred_array = np.expand_dims(pred_array, axis=0)
            pred_array = w_np_non_max_suppression(
                pred_array,
                conf_thresh=confidence,
                iou_thresh=iou_threshold,
                class_agnostic=class_agnostic_nms,
                max_detections=max_detections,
                max_candidate_detections=max_candidates,
                box_format="xywh",
            )[0]
        else:
            pred_array = []
        predictions = []

        for i, pred in enumerate(pred_array):
            predictions.append(
                ObjectDetectionPrediction(
                    **{
                        "x": (pred[0] + pred[2]) / 2,
                        "y": (pred[1] + pred[3]) / 2,
                        "width": pred[2] - pred[0],
                        "height": pred[3] - pred[1],
                        "confidence": pred[4],
                        "class": self.class_names[int(pred[6])],
                        "class_id": int(pred[6]),
                    }
                )
            )

        responses = ObjectDetectionInferenceResponse(
            predictions=predictions,
            image=InferenceResponseImage(width=img_dims[1], height=img_dims[0]),
            time=t2,
        )
        return responses

    def set_classes(self, text: list):
        """Set the class names for the model.

        Args:
            text (list): The class names.
        """
        text_hash = get_string_list_hash(text)
        cached_embeddings = cache.get_numpy(text_hash)
        if cached_embeddings is not None:
            self.model.model.txt_feats = cached_embeddings
            self.model.model.model[-1].nc = len(text)
        else:
            self.model.set_classes(text)
            cache.set_numpy(text_hash, self.model.model.txt_feats, expire=300)
        self.class_names = text

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["model.pt"].
        """
        return ["yolo-world.pt"]

    def get_cls_conf_array(self, class_id) -> list:
        arr = [0] * len(self.class_names)
        arr[class_id] = 1
        return arr
