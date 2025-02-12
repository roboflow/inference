from typing import Tuple

import numpy as np

from inference.core.exceptions import ModelArtefactError
from inference.core.models.keypoints_detection_base import (
    KeypointsDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.models.utils.keypoints import superset_keypoints_count
from inference.core.utils.onnx import run_session_via_iobinding


class YOLOv8KeypointsDetection(KeypointsDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX keypoints detection model (Implements an object detection specific infer method).

    This class is responsible for performing keypoints detection using the YOLOv8 model
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

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
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
        number_of_classes = len(self.get_class_names)
        class_confs = predictions[:, :, 4 : 4 + number_of_classes]
        keypoints_detections = predictions[:, :, 4 + number_of_classes :]
        confs = np.expand_dims(np.max(class_confs, axis=2), axis=2)
        bboxes_predictions = np.concatenate(
            [boxes, confs, class_confs, keypoints_detections], axis=2
        )
        return (bboxes_predictions,)

    def keypoints_count(self) -> int:
        """Returns the number of keypoints in the model."""
        if self.keypoints_metadata is None:
            raise ModelArtefactError("Keypoints metadata not available.")
        return superset_keypoints_count(self.keypoints_metadata)
