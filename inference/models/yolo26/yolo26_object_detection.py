from inference.models.yolov11.yolov11_object_detection import YOLOv11ObjectDetection
import numpy as np
from typing import Tuple, List, Optional
from inference.core.utils.onnx import ImageMetaType, run_session_via_iobinding
from inference.core.models.defaults import (
    DEFAULT_CLASS_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESH,
    DEFAULT_MAX_CANDIDATES,
    DEFAUlT_MAX_DETECTIONS,
)

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.postprocess import post_process_bboxes

class YOLO26ObjectDetection(YOLOv11ObjectDetection):
    def predict(self, img_in: ImageMetaType, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class confidence scores.
        """
        with self._session_lock:
            predictions = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )[0]
        boxes = predictions[:, :, :4]
        confs = predictions[:, :, 4:5]
        class_indices = predictions[:, :, 5:6]
        predictions = np.concatenate([boxes, confs, class_indices], axis=2)

        return (predictions,)

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preproc_return_metadata: PreprocessReturnMetadata,
        class_agnostic_nms=DEFAULT_CLASS_AGNOSTIC_NMS,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
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
            List[ObjectDetectionInferenceResponse]: The post-processed predictions.
        """
        predictions = predictions[0]
        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preproc_return_metadata["img_dims"]
        new_predictions = []
        for batch_predictions in predictions:
            keep_confidences = batch_predictions[:, 4] > confidence
            batch_predictions = batch_predictions[keep_confidences]
            new_predictions.append(batch_predictions)

        predictions = new_predictions
        predictions = post_process_bboxes(
            predictions,
            infer_shape,
            img_dims,
            self.preproc,
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
    ) -> List[ObjectDetectionInferenceResponse]:
        """Constructs object detection response objects based on predictions.

        Args:
            predictions (List[List[float]]): The list of predictions.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            List[ObjectDetectionInferenceResponse]: A list of response objects containing object detection predictions.
        """

        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]

        predictions = predictions[
            : len(img_dims)
        ]  # If the batch size was fixed we have empty preds at the end

        responses = [
            ObjectDetectionInferenceResponse(
                predictions=[
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[5])],
                            "class_id": int(pred[5]),
                        }
                    )
                    for pred in batch_predictions
                    if not class_filter
                    or self.class_names[int(pred[5])] in class_filter
                ],
                image=InferenceResponseImage(
                    width=img_dims[ind][1], height=img_dims[ind][0]
                ),
            )
            for ind, batch_predictions in enumerate(predictions)
        ]
        return responses


    def validate_model_classes(self) -> None:
        pass