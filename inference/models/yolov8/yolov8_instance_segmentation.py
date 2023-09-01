from typing import List, Tuple

import numpy as np

from inference.core.models.instance_segmentation_base import (
    InstanceSegmentationBaseOnnxRoboflowInferenceModel,
)


class YOLOv8InstanceSegmentation(InstanceSegmentationBaseOnnxRoboflowInferenceModel):
    """YOLOv8 Instance Segmentation ONNX Inference Model.

    This class is responsible for performing instance segmentation using the YOLOv8 model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        infer_onnx: Performs inference on the given image using the ONNX session.
    """

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv8 model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def infer_onnx(self, img_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing two NumPy arrays representing the predictions and protos. The predictions include boxes, confidence scores, class confidence scores, and masks.
        """
        predictions = self.onnx_session.run(None, {self.input_name: img_in})
        protos = predictions[1]
        predictions = predictions[0]
        predictions = predictions.transpose(0, 2, 1)
        boxes = predictions[:, :, :4]
        class_confs = predictions[:, :, 4:-32]
        confs = np.expand_dims(np.max(class_confs, axis=2), axis=2)
        masks = predictions[:, :, -32:]
        predictions = np.concatenate([boxes, confs, class_confs, masks], axis=2)
        return predictions, protos
