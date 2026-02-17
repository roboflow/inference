from typing import Tuple

import numpy as np

from inference.core.models.semantic_segmentation_base import SemanticSegmentationBaseOnnxRoboflowInferenceModel

from inference.core.utils.onnx import run_session_via_iobinding

class DeepLabV3PlusSemanticSegmentation(SemanticSegmentationBaseOnnxRoboflowInferenceModel):
    """DeepLabV3Plus Semantic Segmentation ONNX Inference Model.

    This class is responsible for performing semantic segmentation using the DeepLabV3Plus model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs inference on the given image using the ONNX session.
    """

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the DeepLabV3Plus model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"
    
    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing two NumPy arrays representing the predictions and protos. The predictions include boxes, confidence scores, class confidence scores, and masks.
        """
        with self._session_lock:
            predictions, protos = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )
        return predictions, protos