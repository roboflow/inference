from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch


@dataclass
class InstanceDetections:
    xyxy: torch.Tensor  # (n_boxes, 4)
    class_ids: torch.Tensor  # (n_boxes, )
    confidence: torch.Tensor  # (n_boxes, )
    masks: torch.Tensor  # (n_boxes, mask_height, mask_width)
    image_metadata: Optional[dict] = None
    bboxes_metadata: Optional[List[dict]] = (
        None  # if given, list of size equal to # of bboxes
    )


class InstanceSegmentationModel(ABC):

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, *args, **kwargs
    ) -> "InstanceSegmentationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self, images: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs
    ) -> List[InstanceDetections]:
        pre_processed_images, pre_processing_meta = self.pre_process(
            images, *args, **kwargs
        )
        model_results = self.forward(pre_processed_images, *args, **kwargs)
        return self.post_process(model_results, pre_processing_meta, *args, **kwargs)

    @abstractmethod
    def pre_process(
        self, images: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs
    ) -> Tuple[torch.Tensor, Any]:
        pass

    @abstractmethod
    def forward(
        self, pre_processed_images: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def post_process(
        self, model_results: torch.Tensor, pre_processing_meta: Any, *args, **kwargs
    ) -> List[InstanceDetections]:
        pass

    def __call__(
        self, images: torch.Tensor, *args, **kwargs
    ) -> List[InstanceDetections]:
        return self.infer(images, *args, **kwargs)
