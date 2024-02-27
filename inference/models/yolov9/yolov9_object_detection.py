from typing import Tuple

import numpy as np

from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)


class YOLOv9ObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection model (Implements an object detection specific infer method).

    This class is responsible for performing object detection using the YOLOv9 model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.
    """

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv9 model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions.
        """
        # (b x 8 x 8000)
        predictions = self.onnx_session.run(None, {self.input_name: img_in})[0]
        predictions = predictions.transpose(0, 2, 1)
        boxes = predictions[:, :, :4]
        class_confs = predictions[:, :, 4:]
        confs = np.expand_dims(np.max(class_confs, axis=2), axis=2)
        predictions = np.concatenate([boxes, confs, class_confs], axis=2)
        return (predictions,)
