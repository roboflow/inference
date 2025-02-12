from typing import Tuple

import numpy as np

from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.utils.onnx import run_session_via_iobinding


class YOLONASObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    box_format = "xyxy"

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLO-NAS model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class confidence scores.
        """
        predictions = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )
        boxes = predictions[0]
        class_confs = predictions[1]
        confs = np.expand_dims(np.max(class_confs, axis=2), axis=2)
        predictions = np.concatenate([boxes, confs, class_confs], axis=2)
        return (predictions,)
