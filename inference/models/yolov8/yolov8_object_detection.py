import time
from typing import Tuple, Union

import numpy as np
import onnxruntime as ort

from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.utils.onnx import ImageMetaType, run_session_via_iobinding


class YOLOv8ObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection model (Implements an object detection specific infer method).

    This class is responsible for performing object detection using the YOLOv8 model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs object detection on the given image using the ONNX session.
    """

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv8 model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def predict(self, img_in: ImageMetaType, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class confidence scores.
        """
        predictions = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )[0]
        predictions = predictions.transpose(0, 2, 1)
        boxes = predictions[:, :, :4]
        class_confs = predictions[:, :, 4:]
        confs = np.expand_dims(np.max(class_confs, axis=2), axis=2)
        predictions = np.concatenate([boxes, confs, class_confs], axis=2)
        return (predictions,)
