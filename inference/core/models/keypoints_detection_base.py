from typing import List, Optional, Tuple

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
)
from inference.core.exceptions import ModelArtefactError
from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.keypoints import model_keypoints_to_response
from inference.core.models.utils.validate import (
    get_num_classes_from_model_prediction_shape,
)
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import post_process_bboxes, post_process_keypoints

DEFAULT_CONFIDENCE = 0.4
DEFAULT_IOU_THRESH = 0.3
DEFAULT_CLASS_AGNOSTIC_NMS = False
DEFAUlT_MAX_DETECTIONS = 300
DEFAULT_MAX_CANDIDATES = 3000


class KeypointsDetectionBaseOnnxRoboflowInferenceModel(
    ObjectDetectionBaseOnnxRoboflowInferenceModel
):
    """Roboflow ONNX Object detection model. This class implements an object detection specific infer method."""

    task_type = "keypoint-detection"

    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)

    def get_infer_bucket_file_list(self) -> list:
        """Returns the list of files to be downloaded from the inference bucket for ONNX model.

        Returns:
            list: A list of filenames specific to ONNX models.
        """
        return ["environment.json", "class_names.txt", "keypoints_metadata.json"]

    def postprocess(
        self,
        predictions: Tuple[np.ndarray],
        preproc_return_metadata: PreprocessReturnMetadata,
        class_agnostic_nms=DEFAULT_CLASS_AGNOSTIC_NMS,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        """Postprocesses the object detection predictions.

        Args:
            predictions (np.ndarray): Raw predictions from the model.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_agnostic_nms (bool): Whether to apply class-agnostic non-max suppression. Default is False.
            confidence (float): Confidence threshold for filtering detections. Default is 0.5.
            iou_threshold (float): IoU threshold for non-max suppression. Default is 0.5.
            max_candidates (int): Maximum number of candidate detections. Default is 3000.
            max_detections (int): Maximum number of final detections. Default is 300.

        Returns:
            List[KeypointsDetectionInferenceResponse]: The post-processed predictions.
        """
        predictions = predictions[0]
        number_of_classes = len(self.get_class_names)
        num_masks = predictions.shape[2] - 5 - number_of_classes
        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            class_agnostic=class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
            num_masks=num_masks,
        )

        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preproc_return_metadata["img_dims"]
        predictions = post_process_bboxes(
            predictions=predictions,
            infer_shape=infer_shape,
            img_dims=img_dims,
            preproc=self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        predictions = post_process_keypoints(
            predictions=predictions,
            keypoints_start_index=-num_masks,
            infer_shape=infer_shape,
            img_dims=img_dims,
            preproc=self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        return self.make_response(predictions, img_dims, **kwargs)

    def make_response(
        self,
        predictions: List[List[float]],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        """Constructs object detection response objects based on predictions.

        Args:
            predictions (List[List[float]]): The list of predictions.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            List[KeypointsDetectionInferenceResponse]: A list of response objects containing keypoints detection predictions.
        """
        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]
        keypoint_confidence_threshold = 0.0
        if "request" in kwargs:
            keypoint_confidence_threshold = kwargs["request"].keypoint_confidence
        responses = [
            KeypointsDetectionInferenceResponse(
                predictions=[
                    KeypointsPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[6])],
                            "class_id": int(pred[6]),
                            "keypoints": model_keypoints_to_response(
                                keypoints_metadata=self.keypoints_metadata,
                                keypoints=pred[7:],
                                predicted_object_class_id=int(pred[6]),
                                keypoint_confidence_threshold=keypoint_confidence_threshold,
                            ),
                        }
                    )
                    for pred in batch_predictions
                    if not class_filter
                    or self.class_names[int(pred[6])] in class_filter
                ],
                image=InferenceResponseImage(
                    width=img_dims[ind][1], height=img_dims[ind][0]
                ),
            )
            for ind, batch_predictions in enumerate(predictions)
        ]
        return responses

    def keypoints_count(self) -> int:
        raise NotImplementedError

    def validate_model_classes(self) -> None:
        num_keypoints = self.keypoints_count()
        output_shape = self.get_model_output_shape()
        num_classes = get_num_classes_from_model_prediction_shape(
            len_prediction=output_shape[2], keypoints=num_keypoints
        )
        if num_classes != self.num_classes:
            raise ValueError(
                f"Number of classes in model ({num_classes}) does not match the number of classes in the environment ({self.num_classes})"
            )
