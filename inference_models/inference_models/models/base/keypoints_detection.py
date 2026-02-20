from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
import torch

from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)


@dataclass
class KeyPoints:
    xy: torch.Tensor  # (instances, instance_key_points, 2)
    class_id: torch.Tensor  # (instances, )
    confidence: torch.Tensor  # (instances, instance_key_points)
    image_metadata: Optional[dict] = None
    key_points_metadata: Optional[List[dict]] = (
        None  # if given, list of size equal to # of instances
    )

    def to_supervision(self) -> sv.KeyPoints:
        """Convert keypoints to Supervision KeyPoints format.

        Converts the PyTorch tensor-based keypoints to Supervision's NumPy-based
        format for visualization and analysis. This enables use of Supervision's
        keypoint annotators and skeleton visualization tools.

        Returns:
            sv.KeyPoints: Supervision KeyPoints object with:

                - xy: Keypoint coordinates as NumPy array (N, K, 2) where N is number
                  of instances and K is number of keypoints per instance

                - class_id: Class IDs as NumPy array (N,)

                - confidence: Keypoint confidence scores as NumPy array (N, K)

        Examples:
            Convert and visualize keypoints:

            >>> import cv2
            >>> import supervision as sv
            >>> from inference_models import AutoModel
            >>>
            >>> model = AutoModel.from_pretrained("yolov8n-pose-640")
            >>> image = cv2.imread("image.jpg")
            >>> results = model(image)
            >>> key_points_list, detections_list = results
            >>>
            >>> # Convert to Supervision format
            >>> key_points = key_points_list[0].to_supervision()
            >>>
            >>> # Use Supervision annotators
            >>> vertex_annotator = sv.VertexAnnotator()
            >>> edge_annotator = sv.EdgeAnnotator(edges=model.skeletons[0])
            >>> annotated = edge_annotator.annotate(image.copy(), key_points)
            >>> annotated = vertex_annotator.annotate(annotated, key_points)

            Filter by class:

            >>> key_points = key_points_list[0].to_supervision()
            >>> person_mask = key_points.class_id == 0
            >>> person_keypoints = key_points[person_mask]

        See Also:
            - Supervision documentation: https://supervision.roboflow.com
        """
        return sv.KeyPoints(
            xy=self.xy.cpu().numpy(),
            class_id=self.class_id.cpu().numpy(),
            confidence=self.confidence.cpu().numpy(),
        )


class KeyPointsDetectionModel(
    ABC, Generic[PreprocessedInputs, PreprocessingMetadata, RawPrediction]
):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "KeyPointsDetectionModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def key_points_classes(self) -> List[List[str]]:
        pass

    @property
    @abstractmethod
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
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
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        return self.infer(images, **kwargs)
