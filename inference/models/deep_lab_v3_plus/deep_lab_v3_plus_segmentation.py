from typing import Any, List, Tuple

import numpy as np
import torch

from inference.core.models.semantic_segmentation_base import (
    SemanticSegmentationBaseOnnxRoboflowInferenceModel,
    SemanticSegmentationModelOutput,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    SemanticSegmentationInferenceResponse,
    SemanticSegmentationPrediction,
)
from inference.core.utils.onnx import run_session_via_iobinding


class DeepLabV3PlusSemanticSegmentation(
    SemanticSegmentationBaseOnnxRoboflowInferenceModel
):
    """DeepLabV3Plus Semantic Segmentation ONNX Inference Model.

    This class is responsible for performing semantic segmentation using the DeepLabV3Plus model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs inference on the given image using the ONNX session.
    """

    preprocess_means = [0.485, 0.456, 0.406]
    preprocess_stds = [0.229, 0.224, 0.225]

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the DeepLabV3Plus model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        img_in, img_dims = self.load_image(image)

        img_in /= 255.0
        # borrowed from classification_base.py
        mean = self.preprocess_means
        std = self.preprocess_stds
        img_in[:, 0, :, :] = (img_in[:, 0, :, :] - mean[0]) / std[0]
        img_in[:, 1, :, :] = (img_in[:, 1, :, :] - mean[1]) / std[1]
        img_in[:, 2, :, :] = (img_in[:, 2, :, :] - mean[2]) / std[2]

        return img_in, PreprocessReturnMetadata(
            {
                "img_dims": img_dims,
                "im_shape": img_in.shape,
            }
        )

    def predict(self, img_in: np.ndarray, **kwargs) -> SemanticSegmentationModelOutput:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            SemanticSegmentationPredictions: Tuple containing one NumPy array representing the predictions.
        """
        with self._session_lock:
            predictions = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )
        return predictions

    def postprocess(
        self,
        predictions: SemanticSegmentationModelOutput,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs
    ) -> List[SemanticSegmentationInferenceResponse]:

        img_dims = preprocess_return_metadata["img_dims"]
        predictions = [torch.tensor(p) for p in predictions[0]]
        responses = []

        for pred, img_dim in zip(predictions, img_dims):
            if pred.size == 0:
                continue

            class_probs = torch.nn.functional.softmax(pred, dim=0)
            confidence, class_ids = torch.max(class_probs, dim=0)
            # stretch to img_dim
            confidence = torch.nn.functional.interpolate(
                confidence.unsqueeze(dim=0).unsqueeze(dim=0),
                size=img_dim,
                mode="nearest",
            ).squeeze()
            class_ids = (
                torch.nn.functional.interpolate(
                    class_ids.unsqueeze(dim=0).unsqueeze(dim=0).to(torch.float),
                    size=img_dim,
                    mode="nearest",
                )
                .squeeze()
                .to(torch.long)
            )

            response_predictions = SemanticSegmentationPrediction(
                segmentation_map=class_ids, class_confidence=confidence
            )
            responses.append(
                SemanticSegmentationInferenceResponse(
                    predictions=response_predictions,
                    image=InferenceResponseImage(width=img_dim[1], height=img_dim[0]),
                )
            )

        return responses
