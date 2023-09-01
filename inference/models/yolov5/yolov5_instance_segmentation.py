from typing import List, Tuple

import numpy as np

from inference.core.models.instance_segmentation_base import (
    InstanceSegmentationBaseOnnxRoboflowInferenceModel,
)


class YOLOv5InstanceSegmentation(InstanceSegmentationBaseOnnxRoboflowInferenceModel):
    """YOLOv5 Instance Segmentation ONNX Inference Model.

    This class is responsible for performing instance segmentation using the YOLOv5 model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.
    """

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv5 model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "yolov5s_weights.onnx"

    def infer_onnx(self, img_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing two NumPy arrays representing the predictions.
        """
        predictions = self.onnx_session.run(None, {self.input_name: img_in})
        return predictions[0], predictions[1]
