import os
from typing import List, Tuple

import numpy as np

from inference.core.models.defaults import (
    DEFAULT_CLASS_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESH,
    DEFAULT_MAX_CANDIDATES,
    DEFAUlT_MAX_DETECTIONS,
)

from inference.core.logger import logger
from typing import Any

from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
    ObjectDetectionInferenceResponse,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.onnx import ImageMetaType, run_session_via_iobinding

class RFDETRObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection model (Implements an object detection specific infer method).

    This class is responsible for performing object detection using the RFDETR model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs object detection on the given image using the ONNX session.
    """

    preprocess_means = [0.485, 0.456, 0.406]
    preprocess_stds = [0.229, 0.224, 0.225]

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the RFDETR model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return os.path.join(self.cache_dir, "weights.onnx")

    def preprocess(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        fix_batch_size: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        self.resize_method = "Stretch to"
        return super().preprocess(image, disable_preproc_auto_orient, disable_preproc_contrast, disable_preproc_grayscale, disable_preproc_static_crop, fix_batch_size, **kwargs)
    
    def predict(
        self, img_in: ImageMetaType, threshold=0.5, **kwargs
    ) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.
            threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.5.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class IDs.
        """
        predictions = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )

        bboxes = predictions[0]
        logits = predictions[1]

        return (bboxes, logits)

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
            preproc_return_metadata (PreprocessReturnMetadata): Metadata from preprocessing.
            class_agnostic_nms (bool): Whether to apply class-agnostic non-max suppression.
            confidence (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for non-max suppression.
            max_candidates (int): Maximum number of candidate detections.
            max_detections (int): Maximum number of final detections.

        Returns:
            List[ObjectDetectionInferenceResponse]: The post-processed predictions.
        """
        bboxes = predictions[0]
        logits = predictions[1]

        logits = 1 / (1 + np.exp(-logits))

        img_dims = preproc_return_metadata["img_dims"]

        processed_predictions = []

        for batch_idx in range(bboxes.shape[0]):
            orig_h, orig_w = img_dims[batch_idx]

            boxes = bboxes[batch_idx]
            image_logits = logits[batch_idx]

            scores = np.max(image_logits, axis=1)
            labels = np.argmax(image_logits, axis=1)

            confidence_mask = scores > confidence
            filtered_boxes = boxes[confidence_mask]
            filtered_scores = scores[confidence_mask]
            filtered_labels = labels[confidence_mask]

            converted_boxes = []
            for box in filtered_boxes:
                cx, cy, w, h = box
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                scale_fct = np.array([orig_w, orig_h, orig_w, orig_h])
                converted_box = np.array([x1, y1, x2, y2]) * scale_fct

                converted_boxes.append(converted_box)

            converted_boxes = np.array(converted_boxes)

            if len(converted_boxes) > 0:
                batch_predictions = np.column_stack(
                    (
                        converted_boxes,
                        filtered_scores.reshape(-1, 1),
                        np.zeros((len(filtered_scores), 1)),
                        filtered_labels.reshape(-1, 1),
                    )
                )
            else:
                batch_predictions = np.zeros((0, 7))

            processed_predictions.append(batch_predictions)

        return self.make_response(processed_predictions, img_dims, **kwargs)

    def validate_model_classes(self) -> None:
        pass
