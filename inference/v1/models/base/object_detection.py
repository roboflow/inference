from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch


@dataclass
class Detections:
    xyxy: torch.Tensor  # (n_boxes, 4)
    class_ids: torch.Tensor  # (n_boxes, )
    confidence: torch.Tensor  # (n_boxes, )
    image_metadata: Optional[dict] = None
    bboxes_metadata: Optional[List[dict]] = (
        None  # if given, list of size equal to # of bboxes
    )


class ObjectDetectionModel(ABC):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, *args, **kwargs
    ) -> "ObjectDetectionModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self, images: Union[torch.Tensor, List[torch.Tensor]], **kwargs
    ) -> List[Detections]:
        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, pre_processing_meta, **kwargs)

    @abstractmethod
    def pre_process(
        self, images: Union[torch.Tensor, List[torch.Tensor]], **kwargs
    ) -> Tuple[torch.Tensor, Any]:
        pass

    @abstractmethod
    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def post_process(
        self, model_results: torch.Tensor, pre_processing_meta: Any, **kwargs
    ) -> List[Detections]:
        pass

    def __call__(self, images: torch.Tensor, **kwargs) -> List[Detections]:
        return self.infer(images, **kwargs)
