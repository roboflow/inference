from typing import Tuple

import numpy as np

from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)


class YOLOv10ObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection model (Implements an object detection specific infer method).

    This class is responsible for performing object detection using the YOLOv10 model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs object detection on the given image using the ONNX session.
    """

    box_format = "xyxy"

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv10 model.

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
        predictions = self.onnx_session.run(None, {self.input_name: img_in})[0]
        boxes = predictions[:, :, :4]
        batch_size = boxes.shape[0]
        num_boxes = boxes.shape[1]
        class_confs = np.zeros(
            (batch_size, num_boxes, self.num_classes), dtype=np.float32
        )
        for batch in range(batch_size):
            for box in range(num_boxes):
                class_id = int(predictions[batch, box, 5])
                confidence = predictions[batch, box, 4]
                class_confs[batch, box, class_id] = confidence

        confs = predictions[:, :, 4:5]
        predictions = np.concatenate([boxes, confs, class_confs], axis=2)

        return (predictions,)
