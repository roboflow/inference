from typing import List, Tuple

import numpy as np

from inference.core.models.instance_segmentation_base import (
    InstanceSegmentationBaseOnnxRoboflowInferenceModel,
)
from inference.core.utils.onnx import run_session_via_iobinding


class YOLOv7InstanceSegmentation(InstanceSegmentationBaseOnnxRoboflowInferenceModel):
    """YOLOv7 Instance Segmentation ONNX Inference Model.

    This class is responsible for performing instance segmentation using the YOLOv7 model
    with ONNX runtime.

    Methods:
        predict: Performs inference on the given image using the ONNX session.
    """

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing two NumPy arrays representing the predictions and protos.
        """
        predictions = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )
        protos = predictions[4]
        predictions = predictions[0]
        return predictions, protos
