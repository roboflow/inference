from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

from inference.v1.models.base.object_detection import Detections


@dataclass
class KeyPoints:
    xy: torch.Tensor  # (instances, instance_key_points, 2)
    class_id: torch.Tensor  # (instances, instance_key_points)
    confidence: torch.Tensor  # (instances, instance_key_points)
    image_metadata: Optional[dict] = None
    key_points_metadata: Optional[List[dict]] = None  # if given, list of size equal to # of instances


class KeyPointsDetectionModel(ABC):

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args, **kwargs) -> "KeyPointsDetectionModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(self, images: torch.Tensor, *args, **kwargs) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        pre_processed_images = self.pre_process(images, *args, **kwargs)
        model_results = self.forward(pre_processed_images, *args, **kwargs)
        return self.post_process(model_results, *args, **kwargs)

    @abstractmethod
    def pre_process(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, pre_processed_images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def post_process(self, model_results: torch.Tensor, *args, **kwargs) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        pass

    def __call__(self, images: torch.Tensor, *args, **kwargs) -> Tuple[List[KeyPoints], Optional[List[Detections]]]:
        return self.infer(images, *args, **kwargs)
