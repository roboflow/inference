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
    confidence: torch.Tensor  # per-keypoint confidence (instances, instance_key_points)
    image_metadata: Optional[dict] = None
    key_points_metadata: Optional[List[dict]] = (
        None  # if given, list of size equal to # of instances
    )
    covariance: Optional[torch.Tensor] = (
        None  # if given, pixel-space per-keypoint covariance (instances, instance_key_points, 2, 2)
    )
    detection_confidence: Optional[torch.Tensor] = (
        None  # if given, per-instance object confidence (instances, )
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

                - keypoint_confidence: Per-keypoint confidence scores as NumPy array (N, K)

                - visible: Per-keypoint visibility as boolean NumPy array (N, K),
                  derived from ``keypoint_confidence > 0``

                - detection_confidence: Per-instance object confidence as NumPy array (N,),
                  only present when the model provides it (e.g. RF-DETR)

                - data["covariance"]: Pixel-space per-keypoint covariance matrices
                  as NumPy array (N, K, 2, 2), only present when the model predicts
                  keypoint localization uncertainty (e.g. RF-DETR). Consumed by
                  Supervision's covariance ellipse annotators.

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
        confidence_array = self.confidence.cpu().numpy()
        kwargs = {
            "xy": self.xy.cpu().numpy(),
            "class_id": self.class_id.cpu().numpy(),
            "keypoint_confidence": confidence_array,
            "visible": confidence_array > 0,
        }
        if self.detection_confidence is not None:
            kwargs["detection_confidence"] = self.detection_confidence.cpu().numpy()
        if self.covariance is not None:
            kwargs["data"] = {"covariance": self.covariance.cpu().numpy()}
        return sv.KeyPoints(**kwargs)


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
