from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
import torch

from inference_models.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)


@dataclass
class Detections:
    xyxy: torch.Tensor  # (n_boxes, 4)
    class_id: torch.Tensor  # (n_boxes, )
    confidence: torch.Tensor  # (n_boxes, )
    image_metadata: Optional[dict] = None
    bboxes_metadata: Optional[List[dict]] = (
        None  # if given, list of size equal to # of bboxes
    )

    def to_supervision(self) -> sv.Detections:
        """Convert detections to Supervision Detections format.

        Converts the PyTorch tensor-based detections to Supervision's NumPy-based
        format for visualization and analysis. This enables use of Supervision's
        rich ecosystem of annotators, trackers, and utilities.

        Returns:
            sv.Detections: Supervision Detections object with:

                - xyxy: Bounding boxes as NumPy array (N, 4) in [x1, y1, x2, y2] format

                - class_id: Class IDs as NumPy array (N,)

                - confidence: Confidence scores as NumPy array (N,)

        Examples:
            Convert and visualize detections:

            >>> import cv2
            >>> import supervision as sv
            >>> from inference_models import AutoModel
            >>>
            >>> model = AutoModel.from_pretrained("yolov8n-640")
            >>> image = cv2.imread("image.jpg")
            >>> predictions = model(image)
            >>>
            >>> # Convert to Supervision format
            >>> detections = predictions[0].to_supervision()
            >>>
            >>> # Use Supervision annotators
            >>> annotator = sv.BoxAnnotator()
            >>> annotated = annotator.annotate(image.copy(), detections)

            Filter by confidence:

            >>> detections = predictions[0].to_supervision()
            >>> high_conf = detections[detections.confidence > 0.7]

        See Also:
            - Supervision documentation: https://supervision.roboflow.com
        """
        return sv.Detections(
            xyxy=self.xyxy.cpu().numpy(),
            class_id=self.class_id.cpu().numpy(),
            confidence=self.confidence.cpu().numpy(),
        )


class ObjectDetectionModel(
    ABC, Generic[PreprocessedInputs, PreprocessingMetadata, RawPrediction]
):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "ObjectDetectionModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[Detections]:
        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, pre_processing_meta, **kwargs)

    @abstractmethod
    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        pass

    @abstractmethod
    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> RawPrediction:
        pass

    @abstractmethod
    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessingMetadata,
        **kwargs,
    ) -> List[Detections]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[Detections]:
        return self.infer(images, **kwargs)


class OpenVocabularyObjectDetectionModel(
    ABC, Generic[PreprocessedInputs, PreprocessingMetadata, RawPrediction]
):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "OpenVocabularyObjectDetectionModel":
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        classes: Union[str, List[str]],
        **kwargs,
    ) -> List[Detections]:
        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, classes, **kwargs)
        return self.post_process(model_results, pre_processing_meta, **kwargs)

    @abstractmethod
    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        pass

    @abstractmethod
    def forward(
        self,
        pre_processed_images: PreprocessedInputs,
        classes: List[str],
        **kwargs,
    ) -> RawPrediction:
        pass

    @abstractmethod
    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessingMetadata,
        **kwargs,
    ) -> List[Detections]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        classes: List[str],
        **kwargs,
    ) -> List[Detections]:
        return self.infer(images=images, classes=classes, **kwargs)
