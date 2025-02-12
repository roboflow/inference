from typing import List, Tuple

import numpy as np

from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse
from inference.core.models.defaults import DEFAULT_CONFIDENCE, DEFAUlT_MAX_DETECTIONS
from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.onnx import run_session_via_iobinding
from inference.core.utils.postprocess import post_process_bboxes


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
        predictions = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )[0]

        return (predictions,)

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preproc_return_metadata: PreprocessReturnMetadata,
        confidence: float = DEFAULT_CONFIDENCE,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        """Postprocesses the object detection predictions.

        Args:
            predictions (np.ndarray): Raw predictions from the model.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            confidence (float): Confidence threshold for filtering detections. Default is 0.5.
            max_detections (int): Maximum number of final detections. Default is 300.

        Returns:
            List[ObjectDetectionInferenceResponse]: The post-processed predictions.
        """
        predictions = predictions[0]
        predictions = np.append(predictions, predictions[..., 5:], axis=-1)
        predictions[..., 5] = predictions[..., 4]

        mask = predictions[..., 4] > confidence
        predictions = [
            p[mask[idx]][:max_detections] for idx, p in enumerate(predictions)
        ]

        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preproc_return_metadata["img_dims"]
        predictions = post_process_bboxes(
            predictions,
            infer_shape,
            img_dims,
            self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        return self.make_response(predictions, img_dims, **kwargs)

    def validate_model_classes(self) -> None:
        pass
