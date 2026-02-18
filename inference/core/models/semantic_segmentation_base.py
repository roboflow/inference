import base64
import io
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    SemanticSegmentationInferenceResponse,
    SemanticSegmentationPrediction,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.onnx import run_session_via_iobinding

SemanticSegmentationRawPredictions = Tuple[np.ndarray]


class SemanticSegmentationBaseOnnxRoboflowInferenceModel(OnnxRoboflowInferenceModel):

    task_type = "semantic-segmentation"

    preprocess_means = [0.5, 0.5, 0.5]
    preprocess_stds = [0.5, 0.5, 0.5]

    @property
    def class_map(self):
        # match inference-internal/blob/main/deploy/helpers/helpers.py#L107-L128
        return {str(k): v for k, v in enumerate(self.class_names)}

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        img_in, img_dims = self.load_image(image)

        # NB: range scaling and normalization are not automatically applied in load_image
        # see classification_base.py
        img_in /= 255.0

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

    def predict(
        self, img_in: np.ndarray, **kwargs
    ) -> SemanticSegmentationRawPredictions:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            SemanticSegmentationRawPredictions: Tuple containing a NumPy array representing the raw predictions.
            Raw predictions are a (C x H x W) array of logits per class per pixel.
        """
        with self._session_lock:
            predictions = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )  # List[np.ndarray]
        return tuple(predictions)

    def postprocess(
        self,
        predictions: SemanticSegmentationRawPredictions,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs
    ) -> List[SemanticSegmentationInferenceResponse]:
        img_dims = preprocess_return_metadata["img_dims"]
        predictions = predictions[0]
        return self.make_response(predictions, img_dims, **kwargs)

    def make_response(
        self, batch_predictions, img_dims, **kwargs
    ) -> List[SemanticSegmentationInferenceResponse]:
        # (N,C,H,W)
        batch_predictions = torch.tensor(batch_predictions)
        batch_class_probs = torch.nn.functional.softmax(batch_predictions, dim=1)
        # (N,H,W)
        batch_confidence, batch_class_ids = torch.max(batch_class_probs, dim=1)

        responses = []
        for confidence, class_ids, img_dim in zip(
            batch_confidence, batch_class_ids, img_dims
        ):
            # resize to img_dim
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
                .to(torch.uint8)
            )

            # pack up
            response_image = InferenceResponseImage(width=img_dim[1], height=img_dim[0])

            response_predictions = SemanticSegmentationPrediction(
                segmentation_mask=self.img_to_b64_str(class_ids),
                class_map=self.class_map,
                image=dict(response_image),
            )

            response = SemanticSegmentationInferenceResponse(
                predictions=response_predictions,
                image=response_image,
            )
            responses.append(response)

            return responses

    def img_to_b64_str(self, img: torch.Tensor) -> str:
        img = Image.fromarray(img.numpy())
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        return img_str
