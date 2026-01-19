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
class InstanceDetections:
    xyxy: torch.Tensor  # (n_boxes, 4)
    class_id: torch.Tensor  # (n_boxes, )
    confidence: torch.Tensor  # (n_boxes, )
    mask: torch.Tensor  # (n_boxes, mask_height, mask_width)
    image_metadata: Optional[dict] = None
    bboxes_metadata: Optional[List[dict]] = (
        None  # if given, list of size equal to # of bboxes
    )

    def to_supervision(self) -> sv.Detections:
        """Convert instance segmentation detections to Supervision Detections format.

        Converts the PyTorch tensor-based instance segmentation results to Supervision's
        NumPy-based format. This includes both bounding boxes and segmentation masks,
        enabling use of Supervision's mask annotators and analysis tools.

        Returns:
            sv.Detections: Supervision Detections object with:

                - xyxy: Bounding boxes as NumPy array (N, 4) in [x1, y1, x2, y2] format

                - class_id: Class IDs as NumPy array (N,)

                - confidence: Confidence scores as NumPy array (N,)

                - mask: Segmentation masks as NumPy array (N, H, W) with boolean values

        Examples:
            Convert and visualize instance segmentation:

            >>> import cv2
            >>> import supervision as sv
            >>> from inference_models import AutoModel
            >>>
            >>> model = AutoModel.from_pretrained("yolov8n-seg-640")
            >>> image = cv2.imread("image.jpg")
            >>> predictions = model(image)
            >>>
            >>> # Convert to Supervision format
            >>> detections = predictions[0].to_supervision()
            >>>
            >>> # Use Supervision mask annotator
            >>> mask_annotator = sv.MaskAnnotator()
            >>> annotated = mask_annotator.annotate(image.copy(), detections)

            Access masks:

            >>> detections = predictions[0].to_supervision()
            >>> print(f"Masks shape: {detections.mask.shape}")  # (N, H, W)
            >>> print(f"First mask: {detections.mask[0]}")  # Boolean array

        See Also:
            - Supervision documentation: https://supervision.roboflow.com
        """
        return sv.Detections(
            xyxy=self.xyxy.cpu().numpy(),
            class_id=self.class_id.cpu().numpy(),
            confidence=self.confidence.cpu().numpy(),
            mask=self.mask.cpu().numpy(),
        )


class InstanceSegmentationModel(
    ABC, Generic[PreprocessedInputs, PreprocessingMetadata, RawPrediction]
):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "InstanceSegmentationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[InstanceDetections]:
        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, pre_processing_meta, **kwargs)

    @abstractmethod
    def pre_process(
        self, images: Union[torch.Tensor, List[torch.Tensor]], **kwargs
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
        pre_processing_meta: PreprocessedInputs,
        **kwargs,
    ) -> List[InstanceDetections]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[InstanceDetections]:
        return self.infer(images, **kwargs)
